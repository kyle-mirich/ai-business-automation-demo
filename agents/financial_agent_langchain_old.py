"""
Enhanced Financial Agent using LangChain framework
Features:
- LangChain-based architecture with tools and chains
- RAG system for querying sales data with citations
- Interactive chatbot for Q&A
- Tool-based reasoning (visible to user)
- Prophet forecasting integration
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from utils.cost_calculator import estimate_tokens, calculate_gemini_cost
from utils.data_loader import load_sales_data, validate_data


class FinancialAgentLangChain:
    """
    LangChain-powered Financial Agent with RAG and interactive chat

    This agent can:
    1. Load and index sales data for RAG
    2. Answer questions about sales data with citations
    3. Perform complex analysis using tools
    4. Generate forecasts with Prophet
    5. Show tool usage and reasoning transparently
    """

    def __init__(self, api_key: str, data_path: str):
        """
        Initialize the LangChain Financial Agent

        Args:
            api_key: Google Gemini API key
            data_path: Path to sales CSV file
        """
        self.api_key = api_key
        self.data_path = data_path
        self.df = None
        self.summary_stats = {}
        self.total_tokens = 0
        self.chat_history = []

        # Initialize LangChain LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3,
            streaming=True
        )

        # Create tools
        self.tools = self._create_tools()

        # Create agent
        self.agent = None
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
        """
        Create LangChain tools for the agent to use
        These tools will be visible to the user when the agent uses them
        """

        def query_sales_data(query: str) -> str:
            """Query sales data and return results with citations"""
            if self.df is None:
                return "Error: Data not loaded yet"

            try:
                # Parse the query and perform relevant analysis
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
                        sample_rows = self.df[self.df['product'] == product].head(2).index.tolist()
                        citations.append(f"Product '{product}': See rows {sample_rows}")

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
            """Calculate specific statistical metrics on the sales data"""
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

                elif "std" in metric_lower or "deviation" in metric_lower:
                    std_revenue = self.df['revenue'].std()
                    return f"Standard deviation of revenue: ${std_revenue:.2f}"

                elif "min" in metric_lower and "max" in metric_lower:
                    min_rev = self.df['revenue'].min()
                    max_rev = self.df['revenue'].max()
                    return f"Revenue range: ${min_rev:.2f} to ${max_rev:.2f}"

                else:
                    return f"Total revenue: ${self.df['revenue'].sum():,.2f}\nTotal transactions: {len(self.df)}"

            except Exception as e:
                return f"Error calculating statistics: {str(e)}"

        def find_specific_data(criteria: str) -> str:
            """Find specific transactions or data points matching criteria"""
            if self.df is None:
                return "Error: Data not loaded yet"

            try:
                # Parse criteria and filter data
                criteria_lower = criteria.lower()
                filtered = self.df.copy()

                # Extract product name if mentioned
                for product in self.df['product'].unique():
                    if product.lower() in criteria_lower:
                        filtered = filtered[filtered['product'] == product]
                        break

                # Extract category if mentioned
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
                description="Query the sales database for information about revenue, products, categories, monthly trends, or customer segments. Returns data with citations to specific rows/transactions."
            ),
            Tool(
                name="CalculateStatistics",
                func=calculate_statistics,
                description="Calculate statistical metrics like average, median, standard deviation, min/max on sales data."
            ),
            Tool(
                name="FindSpecificData",
                func=find_specific_data,
                description="Find specific transactions or data points matching given criteria (product name, category, date range, etc.). Returns actual row numbers from the dataset."
            ),
            Tool(
                name="ForecastRevenue",
                func=forecast_revenue,
                description="Use Prophet machine learning model to forecast future revenue for a specified number of days. Default is 90 days (Q4 forecast)."
            )
        ]

        return tools

    def initialize_agent(self):
        """Initialize the LangChain agent with tools"""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial data analyst AI assistant with access to Q3 2025 sales data.

You have access to these tools:
- QuerySalesData: Get information about revenue, products, categories, trends, segments WITH CITATIONS
- CalculateStatistics: Calculate statistical metrics
- FindSpecificData: Find specific transactions with row numbers
- ForecastRevenue: Generate ML-based forecasts using Prophet

IMPORTANT:
1. Always use tools to access data - never make up numbers
2. When you use a tool, explain what you're doing and why
3. Always include citations from tool responses in your answers
4. Be specific and reference actual row numbers and data points
5. For complex questions, use multiple tools and synthesize the results

Your goal is to provide accurate, data-driven insights with transparent sourcing."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )

    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Interactive chat with the agent

        Args:
            user_message: User's question or request

        Returns:
            Dictionary with:
                - response: Agent's final answer
                - intermediate_steps: List of tools used and their outputs
                - chat_history: Conversation history
        """
        if self.agent_executor is None:
            self.initialize_agent()

        try:
            # Execute agent
            result = self.agent_executor.invoke({
                "input": user_message,
                "chat_history": self.chat_history
            })

            # Extract response and steps
            response = result.get('output', 'No response generated')
            intermediate_steps = result.get('intermediate_steps', [])

            # Update chat history
            self.chat_history.append(HumanMessage(content=user_message))
            self.chat_history.append(AIMessage(content=response))

            # Format intermediate steps for display
            steps_formatted = []
            for step in intermediate_steps:
                action, observation = step
                steps_formatted.append({
                    'tool': action.tool,
                    'input': action.tool_input,
                    'output': observation
                })

            return {
                'success': True,
                'response': response,
                'intermediate_steps': steps_formatted,
                'chat_history': self.chat_history
            }

        except Exception as e:
            return {
                'success': False,
                'response': f"Error: {str(e)}",
                'intermediate_steps': [],
                'chat_history': self.chat_history
            }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive financial report using the agent
        This simulates the automated analysis workflow
        """
        if self.agent_executor is None:
            self.initialize_agent()

        report_query = """Analyze the Q3 2025 sales data and provide a comprehensive financial report including:

1. Overall revenue performance and key metrics
2. Top performing products (with specific numbers and citations)
3. Category breakdown and analysis
4. Customer segment analysis
5. Monthly trends and growth patterns
6. Q4 revenue forecast using Prophet
7. Key insights and recommendations

Use all available tools to gather data and provide specific citations."""

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
            x=daily_rev['date'],
            y=daily_rev['revenue'],
            mode='lines+markers',
            name='Daily Revenue',
            line=dict(color='#2563eb', width=2),
            marker=dict(size=4)
        ))
        fig1.update_layout(
            title='Q3 2025 Daily Revenue Trend',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        figures.append(fig1)

        # 2. Top Products
        top_products = self.df.groupby('product')['revenue'].sum().sort_values(ascending=True).tail(10)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=top_products.index,
            x=top_products.values,
            orientation='h',
            marker=dict(color='#2563eb')
        ))
        fig2.update_layout(
            title='Top 10 Products by Revenue',
            xaxis_title='Revenue ($)',
            yaxis_title='Product',
            template='plotly_white'
        )
        figures.append(fig2)

        # 3. Category Pie Chart
        cat_revenue = self.df.groupby('category')['revenue'].sum()
        fig3 = go.Figure(data=[go.Pie(
            labels=cat_revenue.index,
            values=cat_revenue.values,
            hole=0.3,
            marker=dict(colors=['#2563eb', '#3b82f6', '#60a5fa'])
        )])
        fig3.update_layout(title='Revenue by Category', template='plotly_white')
        figures.append(fig3)

        # 4. Customer Segments
        seg_revenue = self.df.groupby('customer_segment')['revenue'].sum()
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=seg_revenue.index,
            y=seg_revenue.values,
            marker=dict(color='#2563eb')
        ))
        fig4.update_layout(
            title='Revenue by Customer Segment',
            xaxis_title='Customer Segment',
            yaxis_title='Revenue ($)',
            template='plotly_white'
        )
        figures.append(fig4)

        return figures
