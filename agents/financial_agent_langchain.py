"""
Enhanced Financial Agent with Tool-Based Reasoning
Features:
- Tool-based architecture (simpler than full LangChain agents)
- RAG system for querying sales data with citations
- Interactive chatbot for Q&A
- Visible tool usage
- Prophet forecasting integration
"""

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
                results = []
                citations = []

                # Revenue queries
                if "revenue" in query_lower or "sales" in query_lower:
                    total_rev = self.df['revenue'].sum()
                    results.append(f"Total revenue: ${total_rev:,.2f}")
                    citations.append(f"Source: Aggregated from {len(self.df)} transactions")

                # Product queries
                if "product" in query_lower or "top" in query_lower:
                    top_products = self.df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(5)
                    results.append("\nTop 5 products by revenue:")
                    for product, rev in top_products.items():
                        results.append(f"  - {product}: ${rev:,.2f}")
                        # Get sample row indices for this product
                        sample_rows = self.df[self.df['product'] == product].head(3).index.tolist()
                        citations.append(f"Product '{product}': Rows {','.join(map(str, sample_rows))}")

                # Category queries
                if "category" in query_lower or "categories" in query_lower:
                    cat_revenue = self.df.groupby('category')['revenue'].sum().sort_values(ascending=False)
                    results.append("\nRevenue by category:")
                    for cat, rev in cat_revenue.items():
                        results.append(f"  - {cat}: ${rev:,.2f}")
                        citations.append(f"Category '{cat}': {len(self.df[self.df['category'] == cat])} transactions")

                # Time-based queries
                if "month" in query_lower or "trend" in query_lower or "growth" in query_lower:
                    self.df['month'] = self.df['date'].dt.month_name()
                    monthly_rev = self.df.groupby('month')['revenue'].sum()
                    results.append("\nMonthly revenue:")
                    for month, rev in monthly_rev.items():
                        results.append(f"  - {month}: ${rev:,.2f}")
                    citations.append(f"Monthly breakdown from {self.summary_stats['date_range'][0]} to {self.summary_stats['date_range'][1]}")

                # Customer segment queries
                if "customer" in query_lower or "segment" in query_lower:
                    seg_revenue = self.df.groupby('customer_segment')['revenue'].sum().sort_values(ascending=False)
                    results.append("\nRevenue by customer segment:")
                    for seg, rev in seg_revenue.items():
                        results.append(f"  - {seg}: ${rev:,.2f}")
                        citations.append(f"Segment '{seg}': {len(self.df[self.df['customer_segment'] == seg])} transactions")

                if not results:
                    results.append("Query processed. Data available for: revenue, products, categories, monthly trends, customer segments.")

                # Format response with citations
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
            )
        ]

        return tools

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

            # Decide which tools to call based on keywords
            if any(word in query_lower for word in ['top', 'product', 'revenue', 'category', 'segment', 'month', 'trend']):
                tool_result = self.tools[0].func(user_message)
                tools_used.append({'tool': 'QuerySalesData', 'input': user_message, 'output': tool_result})
                tool_outputs.append(tool_result)

                # Create citation dataframe based on query type
                if 'product' in query_lower or 'top' in query_lower:
                    # Show aggregated product revenue
                    citation_dataframe = self.df.groupby('product').agg({
                        'revenue': 'sum',
                        'quantity': 'sum',
                        'date': ['min', 'max']
                    }).round(2)
                    citation_dataframe.columns = ['Total Revenue', 'Total Quantity', 'First Sale', 'Last Sale']
                    citation_dataframe = citation_dataframe.sort_values('Total Revenue', ascending=False).head(10)
                    citation_dataframe['Total Revenue'] = citation_dataframe['Total Revenue'].apply(lambda x: f"${x:,.2f}")

                elif 'category' in query_lower or 'categories' in query_lower:
                    # Show category breakdown
                    citation_dataframe = self.df.groupby('category').agg({
                        'revenue': 'sum',
                        'quantity': 'sum',
                        'product': 'count'
                    }).round(2)
                    citation_dataframe.columns = ['Total Revenue', 'Total Quantity', 'Transaction Count']
                    citation_dataframe['Total Revenue'] = citation_dataframe['Total Revenue'].apply(lambda x: f"${x:,.2f}")

                elif 'segment' in query_lower or 'customer' in query_lower:
                    # Show customer segment breakdown
                    citation_dataframe = self.df.groupby('customer_segment').agg({
                        'revenue': 'sum',
                        'quantity': 'sum',
                        'product': 'count'
                    }).round(2)
                    citation_dataframe.columns = ['Total Revenue', 'Total Quantity', 'Transaction Count']
                    citation_dataframe['Total Revenue'] = citation_dataframe['Total Revenue'].apply(lambda x: f"${x:,.2f}")

                elif 'month' in query_lower or 'trend' in query_lower:
                    # Show monthly breakdown
                    self.df['month_name'] = self.df['date'].dt.strftime('%B')
                    citation_dataframe = self.df.groupby('month_name').agg({
                        'revenue': 'sum',
                        'quantity': 'sum',
                        'product': 'count'
                    }).round(2)
                    citation_dataframe.columns = ['Total Revenue', 'Total Quantity', 'Transaction Count']
                    citation_dataframe['Total Revenue'] = citation_dataframe['Total Revenue'].apply(lambda x: f"${x:,.2f}")
                    # Reorder by actual month order
                    month_order = ['July', 'August', 'September']
                    citation_dataframe = citation_dataframe.reindex([m for m in month_order if m in citation_dataframe.index])

            if any(word in query_lower for word in ['average', 'mean', 'median', 'statistics', 'calculate']):
                tool_result = self.tools[1].func(user_message)
                tools_used.append({'tool': 'CalculateStatistics', 'input': user_message, 'output': tool_result})
                tool_outputs.append(tool_result)

            if any(word in query_lower for word in ['find', 'show', 'list', 'transactions', 'specific']):
                tool_result = self.tools[2].func(user_message)
                tools_used.append({'tool': 'FindSpecificData', 'input': user_message, 'output': tool_result})
                tool_outputs.append(tool_result)

                # For specific data requests, show actual matching transactions
                import re
                # Try to extract criteria from the query
                for product in self.df['product'].unique():
                    if product.lower() in query_lower:
                        citation_dataframe = self.df[self.df['product'] == product].head(20).copy()
                        # Format for display
                        citation_dataframe['revenue'] = citation_dataframe['revenue'].apply(lambda x: f"${x:.2f}")
                        citation_dataframe['cost'] = citation_dataframe['cost'].apply(lambda x: f"${x:.2f}")
                        citation_dataframe['date'] = citation_dataframe['date'].dt.strftime('%Y-%m-%d')
                        break

            if any(word in query_lower for word in ['forecast', 'predict', 'future', 'q4']):
                tool_result = self.tools[3].func(90)
                tools_used.append({'tool': 'ForecastRevenue', 'input': '90 days (Q4)', 'output': tool_result})
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

Based on this data, provide a clear, comprehensive answer to the user's question.
- Include specific numbers and metrics
- Cite the sources (mention row numbers, transaction counts, etc.) from the tool outputs
- Be concise but thorough
- If citations are provided in the tool output, reference them in your answer

Your response:"""

            # Generate response
            response = self.llm.generate_content(prompt)
            final_response = response.text

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
