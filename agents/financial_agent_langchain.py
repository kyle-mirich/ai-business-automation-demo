"""
Enhanced Financial Agent with Tool-Based Reasoning
Features:
- Tool-based architecture (simpler than full LangChain agents)
- RAG system for querying sales data with citations
- Interactive chatbot for Q&A
- Visible tool usage
- Prophet forecasting integration
"""

import calendar
import difflib
import re

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import google.generativeai as genai

from utils.cost_calculator import estimate_tokens, calculate_gemini_cost
from utils.data_loader import load_sales_data, validate_data


class Tool:
    """Simple tool wrapper"""
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description


class FinancialAgentLangChain:
    """
    Tool-powered Financial Agent with RAG and interactive chat

    This agent can:
    1. Load and index sales data for RAG
    2. Answer questions about sales data with citations
    3. Perform complex analysis using tools
    4. Generate forecasts with Prophet
    5. Show tool usage and reasoning transparently
    """

    def __init__(self, api_key: str, data_path: str):
        """Initialize the Financial Agent"""
        self.api_key = api_key
        self.data_path = data_path
        self.df = None
        self.summary_stats = {}
        self.total_tokens = 0
        self.chat_history = []

        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel('gemini-2.5-flash')

        # Create tools
        self.tools = self._create_tools()
        self.agent_executor = None

    def load_data(self) -> Dict:
        """Load and validate sales data"""
        df, error = load_sales_data(self.data_path)
        if error:
            return {'success': False, 'message': error}

        is_valid, error = validate_data(df)
        if not is_valid:
            return {'success': False, 'message': error}

        self.df = df

        # Calculate summary statistics
        self.summary_stats = {
            'row_count': len(df),
            'date_range': (df['date'].min().strftime('%Y-%m-%d'),
                          df['date'].max().strftime('%Y-%m-%d')),
            'total_revenue': df['revenue'].sum(),
            'total_cost': df['cost'].sum(),
            'profit': df['revenue'].sum() - df['cost'].sum(),
            'unique_products': df['product'].nunique(),
            'unique_categories': df['category'].nunique(),
            'unique_segments': df['customer_segment'].nunique()
        }

        return {
            'success': True,
            'message': 'Data loaded successfully',
            **self.summary_stats
        }

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""

        def query_sales_data(query: str) -> str:
            """Query sales data and return results with citations"""
            if self.df is None:
                return "Error: Data not loaded yet"

            try:
                query_lower = query.lower()
                month_numbers = self._extract_months_from_query(query_lower)
                filtered_df = (
                    self.df[self.df['date'].dt.month.isin(month_numbers)].copy()
                    if month_numbers else self.df
                )

                if month_numbers and filtered_df.empty:
                    return "No sales data found for the requested month(s)."

                active_df = filtered_df if month_numbers else self.df
                results = []
                citations = []

                month_scope = self._format_month_scope(month_numbers)

                # Revenue queries
                if "revenue" in query_lower or "sales" in query_lower:
                    total_rev = active_df['revenue'].sum()
                    total_cost = active_df['cost'].sum()
                    profit = total_rev - total_cost
                    transactions = len(active_df)
                    if month_scope:
                        results.append(f"Total revenue for {month_scope}: ${total_rev:,.2f}")
                    else:
                        results.append(f"Total revenue: ${total_rev:,.2f}")
                    results.append(f"Total cost: ${total_cost:,.2f}")
                    results.append(f"Profit: ${profit:,.2f}")
                    results.append(f"Transactions analyzed: {transactions}")

                    date_min = active_df['date'].min().strftime('%Y-%m-%d')
                    date_max = active_df['date'].max().strftime('%Y-%m-%d')
                    citations.append(
                        f"Revenue scope ({month_scope or 'full dataset'}): {transactions} transactions "
                        f"between {date_min} and {date_max}"
                    )

                # Product queries
                if "product" in query_lower or "top" in query_lower:
                    top_products = (
                        active_df.groupby('product')['revenue']
                        .sum()
                        .sort_values(ascending=False)
                        .head(5)
                    )
                    if month_scope:
                        results.append(f"\nTop 5 products by revenue ({month_scope}):")
                    else:
                        results.append("\nTop 5 products by revenue:")
                    for product, rev in top_products.items():
                        results.append(f"  - {product}: ${rev:,.2f}")
                        product_df = active_df[active_df['product'] == product]
                        transaction_count = len(product_df)
                        first_txn = product_df['date'].min().strftime('%Y-%m-%d')
                        last_txn = product_df['date'].max().strftime('%Y-%m-%d')
                        citations.append(
                            f"Product '{product}': {transaction_count} transactions "
                            f"({first_txn} to {last_txn})"
                        )

                # Category queries
                if "category" in query_lower or "categories" in query_lower:
                    cat_revenue = (
                        active_df.groupby('category')['revenue']
                        .sum()
                        .sort_values(ascending=False)
                    )
                    if month_scope:
                        results.append(f"\nRevenue by category ({month_scope}):")
                    else:
                        results.append("\nRevenue by category:")
                    for cat, rev in cat_revenue.items():
                        results.append(f"  - {cat}: ${rev:,.2f}")
                        category_df = active_df[active_df['category'] == cat]
                        cat_transactions = len(category_df)
                        first_txn = category_df['date'].min().strftime('%Y-%m-%d')
                        last_txn = category_df['date'].max().strftime('%Y-%m-%d')
                        citations.append(
                            f"Category '{cat}': {cat_transactions} transactions "
                            f"({first_txn} to {last_txn})"
                        )

                # Product trend queries
                if "product" in query_lower and ("declining" in query_lower or "growing" in query_lower or "trend" in query_lower):
                    monthly_product_revenue = active_df.groupby([active_df['date'].dt.strftime('%B'), 'product'])['revenue'].sum().unstack(fill_value=0)
                    
                    # Ensure months are in order
                    month_order = ['July', 'August', 'September'] # Assuming Q3
                    monthly_product_revenue = monthly_product_revenue.reindex(month_order).fillna(0)

                    declining_products = []
                    growing_products = []

                    for product in monthly_product_revenue.columns:
                        revenue_series = monthly_product_revenue[product]
                        if len(revenue_series) > 1:
                            # Simple trend: compare first and last non-zero revenue
                            non_zero_rev = revenue_series[revenue_series > 0]
                            if len(non_zero_rev) > 1:
                                trend_is_declining = non_zero_rev.iloc[-1] < non_zero_rev.iloc[0]
                                trend_is_growing = non_zero_rev.iloc[-1] > non_zero_rev.iloc[0]

                                if trend_is_declining or trend_is_growing:
                                    # Create a string showing all monthly revenues for context
                                    monthly_str = " | ".join([f"{month[:3]}: ${rev:,.2f}" for month, rev in revenue_series.items()])
                                    
                                    if trend_is_declining:
                                        declining_products.append(f"- {product}: ({monthly_str})")
                                    elif trend_is_growing:
                                        growing_products.append(f"- {product}: ({monthly_str})")

                    if "declining" in query_lower and declining_products:
                        results.append("\nProducts with declining revenue trend:")
                        results.extend(declining_products)
                    
                    if "growing" in query_lower and growing_products:
                        results.append("\nProducts with growing revenue trend:")
                        results.extend(growing_products)

                # Time-based queries
                if month_numbers or "month" in query_lower or "trend" in query_lower or "growth" in query_lower:
                    monthly_source_df = filtered_df if month_numbers else self.df
                    monthly_data = monthly_source_df.copy()
                    monthly_data['month_number'] = monthly_data['date'].dt.month
                    monthly_data['month_name'] = monthly_data['date'].dt.strftime('%B')
                    monthly_summary = monthly_data.groupby(['month_number', 'month_name']).agg({
                        'revenue': 'sum',
                        'cost': 'sum',
                        'quantity': 'sum',
                        'product': 'count'
                    }).rename(columns={
                        'revenue': 'total_revenue',
                        'cost': 'total_cost',
                        'quantity': 'total_quantity',
                        'product': 'transaction_count'
                    }).reset_index().sort_values('month_number')

                    # Ensure all months in Q3 are present, filling missing ones with 0
                    all_q3_months = pd.DataFrame({
                        'month_number': [7, 8, 9],
                        'month_name': ['July', 'August', 'September']
                    })
                    monthly_summary = pd.merge(all_q3_months, monthly_summary, on=['month_number', 'month_name'], how='left').fillna(0)

                    if month_numbers:
                        monthly_summary = monthly_summary[monthly_summary['month_number'].isin(month_numbers)]

                    if not monthly_summary.empty:
                        header = "\nMonthly revenue:"
                        if month_scope:
                            header = f"\nMonthly revenue ({month_scope}):"
                        results.append(header)

                        for _, row in monthly_summary.iterrows():
                            profit_value = row['total_revenue'] - row['total_cost']
                            results.append(
                                f"  - {row['month_name']}: ${row['total_revenue']:,.2f} revenue | "
                                f"${profit_value:,.2f} profit | {int(row['transaction_count'])} transactions"
                            )
                            month_rows = monthly_data[monthly_data['month_number'] == row['month_number']]
                            month_date_min = month_rows['date'].min().strftime('%Y-%m-%d')
                            month_date_max = month_rows['date'].max().strftime('%Y-%m-%d')
                            citation_text = (
                                f"{row['month_name']} scope: {int(row['transaction_count'])} transactions "
                                f"({month_date_min} to {month_date_max})"
                            )
                            if citation_text not in citations:
                                citations.append(citation_text)

                # Customer segment queries
                if "customer" in query_lower or "segment" in query_lower:
                    seg_revenue = (
                        active_df.groupby('customer_segment')['revenue']
                        .sum()
                        .sort_values(ascending=False)
                    )
                    if month_scope:
                        results.append(f"\nRevenue by customer segment ({month_scope}):")
                    else:
                        results.append("\nRevenue by customer segment:")
                    for seg, rev in seg_revenue.items():
                        results.append(f"  - {seg}: ${rev:,.2f}")
                        segment_df = active_df[active_df['customer_segment'] == seg]
                        seg_transactions = len(segment_df)
                        first_txn = segment_df['date'].min().strftime('%Y-%m-%d')
                        last_txn = segment_df['date'].max().strftime('%Y-%m-%d')
                        citations.append(
                            f"Segment '{seg}': {seg_transactions} transactions "
                            f"({first_txn} to {last_txn})"
                        )

                if not results:
                    results.append("Query processed. Data available for: revenue, products, categories, monthly trends, customer segments.")

                response = "\n".join(results)
                response += "\n\n📚 CITATIONS:\n" + "\n".join([f"  [{i+1}] {cit}" for i, cit in enumerate(citations)])

                return response

            except Exception as e:
                return f"Error querying data: {str(e)}"

        def calculate_statistics(metric: str) -> str:
            """Calculate specific statistical metrics"""
            if self.df is None:
                return "Error: Data not loaded yet"

            try:
                metric_lower = metric.lower()

                if "average" in metric_lower or "mean" in metric_lower:
                    avg_revenue = self.df['revenue'].mean()
                    avg_quantity = self.df['quantity'].mean()
                    return f"Average transaction revenue: ${avg_revenue:.2f}\nAverage quantity per transaction: {avg_quantity:.2f}"
                elif "median" in metric_lower:
                    med_revenue = self.df['revenue'].median()
                    return f"Median transaction revenue: ${med_revenue:.2f}"
                elif "std" in metric_lower:
                    std_revenue = self.df['revenue'].std()
                    return f"Standard deviation of revenue: ${std_revenue:.2f}"
                else:
                    return f"Total revenue: ${self.df['revenue'].sum():,.2f}\nTotal transactions: {len(self.df)}"

            except Exception as e:
                return f"Error calculating statistics: {str(e)}"

        def find_specific_data(criteria: str) -> str:
            """Find specific transactions matching criteria"""
            if self.df is None:
                return "Error: Data not loaded yet"

            try:
                criteria_lower = criteria.lower()
                filtered = self.df.copy()
                month_numbers = self._extract_months_from_query(criteria_lower)

                if month_numbers:
                    filtered = filtered[filtered['date'].dt.month.isin(month_numbers)]

                # Filter by product
                for product in self.df['product'].unique():
                    if product.lower() in criteria_lower:
                        filtered = filtered[filtered['product'] == product]
                        break

                # Filter by category
                for category in self.df['category'].unique():
                    if category.lower() in criteria_lower:
                        filtered = filtered[filtered['category'] == category]
                        break

                if len(filtered) == 0:
                    return "No data found matching criteria"

                # Return sample with row numbers
                sample = filtered.head(10)
                result = f"Found {len(filtered)} matching transactions. Showing first 10:\n\n"

                for idx, row in sample.iterrows():
                    result += f"Row {idx}: {row['date'].strftime('%Y-%m-%d')} | {row['product']} | "
                    result += f"Qty: {row['quantity']} | Rev: ${row['revenue']:.2f} | {row['customer_segment']}\n"

                if len(filtered) > 10:
                    result += f"\n... and {len(filtered) - 10} more transactions"

                return result

            except Exception as e:
                return f"Error finding data: {str(e)}"

        def forecast_revenue(periods: int = 90) -> str:
            """Use Prophet to forecast future revenue"""
            if self.df is None:
                return "Error: Data not loaded yet"

            try:
                # Prepare data for Prophet
                daily_revenue = self.df.groupby('date')['revenue'].sum().reset_index()
                daily_revenue.columns = ['ds', 'y']

                # Train Prophet model
                model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                model.fit(daily_revenue)

                # Make forecast
                future = model.make_future_dataframe(periods=periods, freq='D')
                forecast = model.predict(future)

                # Extract forecast period results
                forecast_period = forecast.tail(periods)
                predicted_total = forecast_period['yhat'].sum()
                lower_bound = forecast_period['yhat_lower'].sum()
                upper_bound = forecast_period['yhat_upper'].sum()

                # Calculate growth
                historical_total = daily_revenue['y'].sum()
                growth_rate = ((predicted_total - historical_total) / historical_total) * 100

                result = f"Prophet Forecast for next {periods} days:\n"
                result += f"  Predicted Revenue: ${predicted_total:,.2f}\n"
                result += f"  Confidence Interval: ${lower_bound:,.2f} - ${upper_bound:,.2f}\n"
                result += f"  Growth vs Historical: {growth_rate:+.1f}%\n"
                result += f"\n📊 Model trained on {len(daily_revenue)} days of historical data"

                return result

            except Exception as e:
                return f"Error forecasting: {str(e)}"

        def calculator(expression: str) -> str:
            """
            A simple calculator to evaluate mathematical expressions.
            Supports +, -, *, /, and parentheses.
            Example: "100 * (1 + 0.25)"
            """
            try:
                # Sanitize expression
                allowed_chars = "0123456789.+-*/() "
                if any(c not in allowed_chars for c in expression):
                    return f"Error: Invalid characters in expression: {expression}"

                # Avoid security issues with eval
                if "__" in expression:
                    return "Error: Invalid expression."

                result = eval(expression, {'__builtins__': None}, {})
                return f"Result of '{expression}': {result}"
            except Exception as e:
                return f"Error calculating '{expression}': {str(e)}"

        # Create tool list
        tools = [
            Tool(
                name="QuerySalesData",
                func=query_sales_data,
                description="Query the sales database for information about revenue, products, categories, monthly trends, or customer segments. Returns data with citations."
            ),
            Tool(
                name="CalculateStatistics",
                func=calculate_statistics,
                description="Calculate statistical metrics like average, median, standard deviation on sales data."
            ),
            Tool(
                name="FindSpecificData",
                func=find_specific_data,
                description="Find specific transactions matching criteria (product, category, etc.). Returns actual row numbers."
            ),
            Tool(
                name="ForecastRevenue",
                func=forecast_revenue,
                description="Use Prophet ML model to forecast future revenue. Default is 90 days (Q4 forecast)."
            ),
            Tool(
                name="Calculator",
                func=calculator,
                description="A simple calculator to evaluate mathematical expressions. Use for calculations like percentage change, ratios, etc. Example: '(120-100)/100 * 100' for percentage increase."
            )
        ]

        return tools

    def _extract_months_from_query(self, query_lower: str) -> List[int]:
        """
        Return month numbers (1-12) mentioned in the query.

        Uses full month names, common abbreviations, and handles "sept".
        """
        months_found: List[int] = []
        month_lookup = {
            calendar.month_name[idx].lower(): idx
            for idx in range(1, 13)
        }
        abbr_lookup = {
            calendar.month_abbr[idx].lower(): idx
            for idx in range(1, 13)
            if calendar.month_abbr[idx]
        }

        for name, idx in month_lookup.items():
            if re.search(rf"\b{name}\b", query_lower):
                months_found.append(idx)

        for abbr, idx in abbr_lookup.items():
            if re.search(rf"\b{abbr}\b", query_lower):
                months_found.append(idx)

        tokens = re.findall(r"[a-z]+", query_lower)
        for token in tokens:
            if token in month_lookup:
                months_found.append(month_lookup[token])
                continue
            if token in abbr_lookup:
                months_found.append(abbr_lookup[token])
                continue

            close_full = difflib.get_close_matches(token, month_lookup.keys(), n=1, cutoff=0.72)
            if close_full:
                months_found.append(month_lookup[close_full[0]])
                continue

            close_abbr = difflib.get_close_matches(token, abbr_lookup.keys(), n=1, cutoff=0.9)
            if close_abbr:
                months_found.append(abbr_lookup[close_abbr[0]])

        if any(token.startswith("sept") for token in tokens) and 9 not in months_found:
            months_found.append(9)

        # Handle month ranges like "July to September"
        if len(set(months_found)) == 2 and (' to ' in query_lower or '-' in query_lower):
            start_month, end_month = min(months_found), max(months_found)
            if start_month < end_month:
                months_found.extend(range(start_month + 1, end_month))

        return sorted(set(months_found))

    def _format_month_scope(self, month_numbers: List[int]) -> str:
        """Return a human-readable month scope label."""
        month_names = [
            calendar.month_name[m] for m in month_numbers
            if 1 <= m <= 12
        ]

        if not month_names:
            return ""
        if len(month_names) == 1:
            return month_names[0]
        if len(month_names) == 2:
            return " and ".join(month_names)
        return ", ".join(month_names[:-1]) + f", and {month_names[-1]}"

    def initialize_agent(self):
        """Initialize the agent"""
        self.agent_executor = True

    def get_rows_data(self, row_indices: List[int]) -> pd.DataFrame:
        """
        Get specific rows from the dataset

        Args:
            row_indices: List of row indices to retrieve

        Returns:
            DataFrame with the requested rows
        """
        if self.df is None:
            return pd.DataFrame()

        try:
            # Get rows that exist
            valid_indices = [idx for idx in row_indices if idx in self.df.index]
            return self.df.loc[valid_indices]
        except Exception as e:
            return pd.DataFrame()

    def execute_pandas_code(self, code: str) -> Dict[str, Any]:
        """
        Execute pandas code safely and return results

        Args:
            code: Pandas code to execute (e.g., "df.groupby('product')['revenue'].sum()")

        Returns:
            Dictionary with success status and results
        """
        if self.df is None:
            return {
                'success': False,
                'error': 'Data not loaded',
                'result': None
            }

        try:
            # Create a safe namespace with only necessary objects
            namespace = {
                'df': self.df.copy(),
                'pd': pd,
                'np': np
            }

            # Execute the code
            result = eval(code, {"__builtins__": {}}, namespace)

            # Convert result to serializable format
            if isinstance(result, pd.DataFrame):
                return {
                    'success': True,
                    'result': result,
                    'result_type': 'dataframe',
                    'code': code
                }
            elif isinstance(result, pd.Series):
                return {
                    'success': True,
                    'result': result.to_frame(),
                    'result_type': 'series',
                    'code': code
                }
            else:
                return {
                    'success': True,
                    'result': result,
                    'result_type': 'value',
                    'code': code
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'result': None,
                'code': code
            }

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Interactive chat with tool usage and structured citations"""
        if self.agent_executor is None:
            self.initialize_agent()

        try:
            # Determine which tools to use
            tools_used = []
            tool_outputs = []
            citation_dataframe = None  # Will hold the aggregated data that answers the question

            query_lower = user_message.lower()
            month_numbers = self._extract_months_from_query(query_lower)
            scoped_df = self.df[self.df['date'].dt.month.isin(month_numbers)] if month_numbers else self.df

            # Decide which tools to call based on keywords
            if any(word in query_lower for word in ['top', 'product', 'revenue', 'category', 'segment', 'month', 'trend', 'percentage', 'increase', 'decrease', 'difference', 'ratio', 'vs', 'compare', 'growth']):
                tool_result = self.tools[0].func(user_message)
                tools_used.append({'tool': 'QuerySalesData', 'input': user_message, 'output': tool_result})
                tool_outputs.append(tool_result)

                # Create citation dataframe based on query type
                if ('declining' in query_lower or 'growing' in query_lower) and 'product' in query_lower:
                    # Show monthly product breakdown
                    if not scoped_df.empty:
                        monthly_product_revenue = scoped_df.groupby([scoped_df['date'].dt.strftime('%B'), 'product'])['revenue'].sum().unstack(fill_value=0)
                        # Ensure months are in order
                        month_order = ['July', 'August', 'September'] # Assuming Q3
                        citation_dataframe = monthly_product_revenue.reindex(month_order).fillna(0)
                        # Format all columns as currency
                        for col in citation_dataframe.columns:
                            citation_dataframe[col] = citation_dataframe[col].apply(lambda x: f"${x:,.2f}")
                elif 'product' in query_lower or 'top' in query_lower:
                    # Show aggregated product revenue
                    if not scoped_df.empty:
                        citation_dataframe = scoped_df.groupby('product').agg({
                            'revenue': 'sum',
                            'quantity': 'sum',
                            'date': ['min', 'max']
                        }).round(2)
                        citation_dataframe.columns = ['Total Revenue', 'Total Quantity', 'First Sale', 'Last Sale']
                        citation_dataframe = citation_dataframe.sort_values('Total Revenue', ascending=False).head(10)
                        citation_dataframe['First Sale'] = citation_dataframe['First Sale'].dt.strftime('%Y-%m-%d')
                        citation_dataframe['Last Sale'] = citation_dataframe['Last Sale'].dt.strftime('%Y-%m-%d')
                        citation_dataframe['Total Revenue'] = citation_dataframe['Total Revenue'].apply(lambda x: f"${x:,.2f}")

                elif 'category' in query_lower or 'categories' in query_lower:
                    # Show category breakdown
                    if not scoped_df.empty:
                        citation_dataframe = scoped_df.groupby('category').agg({
                            'revenue': 'sum',
                            'quantity': 'sum',
                            'product': 'count'
                        }).round(2)
                        citation_dataframe.columns = ['Total Revenue', 'Total Quantity', 'Transaction Count']
                        citation_dataframe['Total Revenue'] = citation_dataframe['Total Revenue'].apply(lambda x: f"${x:,.2f}")

                elif 'segment' in query_lower or 'customer' in query_lower:
                    # Show customer segment breakdown
                    if not scoped_df.empty:
                        citation_dataframe = scoped_df.groupby('customer_segment').agg({
                            'revenue': 'sum',
                            'quantity': 'sum',
                            'product': 'count'
                        }).round(2)
                        citation_dataframe.columns = ['Total Revenue', 'Total Quantity', 'Transaction Count']
                        citation_dataframe['Total Revenue'] = citation_dataframe['Total Revenue'].apply(lambda x: f"${x:,.2f}")

                elif 'month' in query_lower or 'trend' in query_lower:
                    # Show monthly breakdown
                    monthly_source_df = scoped_df if month_numbers else self.df
                    if not monthly_source_df.empty:
                        monthly_data = monthly_source_df.copy()
                        monthly_data['month_number'] = monthly_data['date'].dt.month
                        monthly_data['month_name'] = monthly_data['date'].dt.strftime('%B')
                        citation_dataframe = monthly_data.groupby(['month_number', 'month_name']).agg({
                            'revenue': 'sum',
                            'quantity': 'sum',
                            'product': 'count'
                        }).round(2)
                        citation_dataframe.columns = ['Total Revenue', 'Total Quantity', 'Transaction Count']
                        citation_dataframe['Total Revenue'] = citation_dataframe['Total Revenue'].apply(lambda x: f"${x:,.2f}")
                        citation_dataframe = citation_dataframe.sort_index(level='month_number')
                        citation_dataframe.index = citation_dataframe.index.droplevel('month_number')

            if any(word in query_lower for word in ['average', 'mean', 'median', 'statistics', 'calculate']):
                tool_result = self.tools[1].func(user_message)
                tools_used.append({'tool': 'CalculateStatistics', 'input': user_message, 'output': tool_result})
                tool_outputs.append(tool_result)

            if any(word in query_lower for word in ['find', 'show', 'list', 'transactions', 'specific']):
                tool_result = self.tools[2].func(user_message)
                tools_used.append({'tool': 'FindSpecificData', 'input': user_message, 'output': tool_result})
                tool_outputs.append(tool_result)

                # For specific data requests, show actual matching transactions
                # Try to extract criteria from the query
                for product in self.df['product'].unique():
                    if product.lower() in query_lower:
                        citation_dataframe = scoped_df[scoped_df['product'] == product].head(20).copy()
                        # Format for display
                        citation_dataframe['revenue'] = citation_dataframe['revenue'].apply(lambda x: f"${x:.2f}")
                        citation_dataframe['cost'] = citation_dataframe['cost'].apply(lambda x: f"${x:.2f}")
                        citation_dataframe['date'] = citation_dataframe['date'].dt.strftime('%Y-%m-%d')
                        break

            if any(word in query_lower for word in ['forecast', 'predict', 'future', 'q4']):
                tool_result = self.tools[3].func(90)
                tools_used.append({'tool': 'ForecastRevenue', 'input': '90 days (Q4)', 'output': tool_result})
                tool_outputs.append(tool_result)

            # Decide if a calculation is needed
            if any(word in query_lower for word in ['percentage', 'increase', 'decrease', 'difference', 'ratio', 'vs', 'compare', 'growth']):
                # The other tools already provide a lot of data.
                # We can ask the LLM to formulate a calculation based on the user query and the data we've already fetched.
                calc_context = "\n\n".join(tool_outputs)
                if calc_context: # Only try to calculate if there is context
                    calc_prompt = f"""Based on the user query "{user_message}" and the following data, what is the specific mathematical expression to calculate the answer?

Data:
{calc_context}

- Only return the mathematical expression.
- If no calculation is needed or possible, return "N/A".
- Example: to find percentage increase from 100 to 120, return "(120 - 100) / 100 * 100".
- Use the numbers directly from the data provided.

Expression:"""
                    calc_response = self.llm.generate_content(calc_prompt)
                    calc_expression = getattr(calc_response, 'text', 'N/A').strip().replace('`', '')

                    if calc_expression and calc_expression.lower() != "n/a":
                        # Find calculator tool
                        calculator_tool = next((t for t in self.tools if t.name == "Calculator"), None)
                        if calculator_tool:
                            tool_result = calculator_tool.func(calc_expression)
                            tools_used.append({'tool': 'Calculator', 'input': calc_expression, 'output': tool_result})
                            tool_outputs.append(tool_result)

            # Check if user wants to see specific data visualization
            pandas_code = None
            if any(word in query_lower for word in ['show me', 'display', 'visualize', 'breakdown', 'detailed']):
                # Ask LLM to generate pandas code
                code_prompt = f"""Generate a single line of pandas code to answer: "{user_message}"

Available DataFrame: df
Columns: date, product, quantity, revenue, cost, category, customer_segment

Return ONLY the pandas code, nothing else. Examples:
- df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(10)
- df[df['category']=='Electronics'].head(20)
- df.groupby('customer_segment').agg({{'revenue':'sum', 'quantity':'sum'}})

Code:"""

                code_response = self.llm.generate_content(code_prompt)
                pandas_code = code_response.text.strip().replace('```python', '').replace('```', '').strip()

            # Build context for LLM
            context = "\n\n".join(tool_outputs) if tool_outputs else "No specific tool data was retrieved for this query."

            # Build prompt
            prompt = f"""You are a financial data analyst assistant. The user asked: "{user_message}"

Here is the data retrieved from our analysis tools:

{context}

Based on this data, provide a clear, comprehensive, and well-structured answer to the user's question using Markdown.
- Use headings, subheadings, lists, and tables to structure the information.
- Highlight key metrics and insights using bolding or other Markdown emphasis.
- Ensure the response is easy to read and process by Streamlit's `st.markdown()` function.
- Include specific numbers and metrics from the tool output.
- Focus on the relevant timeframe or filters implied by the data.
- Do not include bracketed citations like [1] or [2] in the response.

Your response should be in Markdown format:"""

            # Generate response
            response = self.llm.generate_content(prompt)
            final_response = getattr(response, 'text', '') or ''
            # The response is expected to be in Markdown, so minimal cleaning is needed.
            # Remove any bracketed citations as they are handled separately.
            final_response = re.sub(r'\s*\[\d+\]', '', final_response).strip()

            # Update chat history
            self.chat_history.append({'role': 'user', 'content': user_message})
            self.chat_history.append({'role': 'assistant', 'content': final_response})

            return {
                'success': True,
                'response': final_response,
                'intermediate_steps': tools_used,
                'chat_history': self.chat_history,
                'citation_dataframe': citation_dataframe,  # Aggregated dataframe showing source data
                'pandas_code': pandas_code  # Generated code if applicable
            }

        except Exception as e:
            return {
                'success': False,
                'response': f"Error: {str(e)}",
                'intermediate_steps': [],
                'chat_history': self.chat_history,
                'citation_dataframe': None,
                'pandas_code': None
            }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive financial report"""
        if self.agent_executor is None:
            self.initialize_agent()

        report_query = """Analyze the Q3 2025 sales data and provide a comprehensive financial report including:

1. Overall revenue performance and key metrics
2. Top performing products with specific numbers
3. Category breakdown and analysis
4. Customer segment analysis
5. Monthly trends and growth patterns
6. Q4 revenue forecast
7. Key insights and recommendations

Provide specific numbers, citations, and data-driven insights."""

        return self.chat(report_query)

    def generate_visualizations(self) -> List[go.Figure]:
        """Generate interactive Plotly visualizations"""
        if self.df is None:
            return []

        figures = []

        # 1. Daily Revenue Trend
        daily_rev = self.df.groupby('date')['revenue'].sum().reset_index()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=daily_rev['date'], y=daily_rev['revenue'],
            mode='lines+markers', name='Daily Revenue',
            line=dict(color='#2563eb', width=2), marker=dict(size=4)
        ))
        fig1.update_layout(title='Q3 2025 Daily Revenue Trend', xaxis_title='Date',
                          yaxis_title='Revenue ($)', hovermode='x unified', template='plotly_white')
        figures.append(fig1)

        # 2. Top Products
        top_products = self.df.groupby('product')['revenue'].sum().sort_values(ascending=True).tail(10)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(y=top_products.index, x=top_products.values,
                              orientation='h', marker=dict(color='#2563eb')))
        fig2.update_layout(title='Top 10 Products by Revenue', xaxis_title='Revenue ($)',
                          yaxis_title='Product', template='plotly_white')
        figures.append(fig2)

        # 3. Category Pie Chart
        cat_revenue = self.df.groupby('category')['revenue'].sum()
        fig3 = go.Figure(data=[go.Pie(labels=cat_revenue.index, values=cat_revenue.values,
                                      hole=0.3, marker=dict(colors=['#2563eb', '#3b82f6', '#60a5fa']))])
        fig3.update_layout(title='Revenue by Category', template='plotly_white')
        figures.append(fig3)

        # 4. Customer Segments
        seg_revenue = self.df.groupby('customer_segment')['revenue'].sum()
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=seg_revenue.index, y=seg_revenue.values, marker=dict(color='#2563eb')))
        fig4.update_layout(title='Revenue by Customer Segment', xaxis_title='Customer Segment',
                          yaxis_title='Revenue ($)', template='plotly_white')
        figures.append(fig4)

        return figures
