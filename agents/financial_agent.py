"""
Financial Agent for analyzing sales data and forecasting revenue
Uses Gemini API for insights and Prophet for forecasting
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import google.generativeai as genai
from typing import Dict, List, Tuple, Generator
from pathlib import Path
import time

from utils.cost_calculator import estimate_tokens, calculate_gemini_cost
from utils.data_loader import load_sales_data, validate_data


class FinancialAgent:
    """
    Agent for financial analysis and forecasting

    This agent:
    1. Loads and validates sales data
    2. Analyzes trends using Gemini API
    3. Forecasts future revenue using Prophet
    4. Generates interactive visualizations
    5. Tracks API costs
    """

    def __init__(self, api_key: str, data_path: str):
        """
        Initialize the Financial Agent

        Args:
            api_key: Google Gemini API key
            data_path: Path to sales CSV file
        """
        self.api_key = api_key
        self.data_path = data_path
        self.df = None
        self.summary_stats = {}
        self.total_tokens = 0

        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def load_data(self) -> Dict:
        """
        Load and validate sales data

        Returns:
            Dictionary with:
                - success: bool
                - message: str
                - row_count: int
                - date_range: tuple
                - total_revenue: float
                - total_cost: float
                - profit: float
        """
        # Load data
        df, error = load_sales_data(self.data_path)

        if error:
            return {
                'success': False,
                'message': error
            }

        # Validate data
        is_valid, error = validate_data(df)

        if not is_valid:
            return {
                'success': False,
                'message': error
            }

        self.df = df

        # Calculate summary statistics
        self.summary_stats = {
            'row_count': len(df),
            'date_range': (df['date'].min().strftime('%Y-%m-%d'), df['date'].max().strftime('%Y-%m-%d')),
            'total_revenue': df['revenue'].sum(),
            'total_cost': df['cost'].sum(),
            'profit': df['revenue'].sum() - df['cost'].sum()
        }

        return {
            'success': True,
            'message': 'Data loaded successfully',
            **self.summary_stats
        }

    def _prepare_analysis_data(self) -> Dict:
        """
        Prepare aggregated data for analysis

        Returns:
            Dictionary with aggregated data by product, category, segment, month
        """
        # Revenue by product
        product_revenue = self.df.groupby('product').agg({
            'revenue': 'sum',
            'quantity': 'sum'
        }).sort_values('revenue', ascending=False)

        # Revenue by category
        category_revenue = self.df.groupby('category')['revenue'].sum().sort_values(ascending=False)

        # Revenue by customer segment
        segment_revenue = self.df.groupby('customer_segment')['revenue'].sum().sort_values(ascending=False)

        # Revenue by month
        self.df['month'] = self.df['date'].dt.month
        self.df['month_name'] = self.df['date'].dt.strftime('%B')
        month_revenue = self.df.groupby(['month', 'month_name'])['revenue'].sum().reset_index()

        # Daily revenue trend
        daily_revenue = self.df.groupby('date')['revenue'].sum().reset_index()

        return {
            'product_revenue': product_revenue,
            'category_revenue': category_revenue,
            'segment_revenue': segment_revenue,
            'month_revenue': month_revenue,
            'daily_revenue': daily_revenue
        }

    def _build_analysis_prompt(self, agg_data: Dict) -> str:
        """
        Build detailed prompt for Gemini analysis

        Args:
            agg_data: Aggregated data dictionary

        Returns:
            Formatted prompt string
        """
        # Top products
        top_products = agg_data['product_revenue'].head(5)
        products_text = "\n".join([
            f"  - {product}: ${revenue:,.2f} ({qty} units)"
            for product, (revenue, qty) in top_products.iterrows()
        ])

        # Categories
        categories_text = "\n".join([
            f"  - {cat}: ${revenue:,.2f}"
            for cat, revenue in agg_data['category_revenue'].items()
        ])

        # Customer segments
        segments_text = "\n".join([
            f"  - {seg}: ${revenue:,.2f}"
            for seg, revenue in agg_data['segment_revenue'].items()
        ])

        # Monthly trend
        months_text = "\n".join([
            f"  - {row['month_name']}: ${row['revenue']:,.2f}"
            for _, row in agg_data['month_revenue'].iterrows()
        ])

        prompt = f"""You are a financial analyst reviewing Q3 2025 sales data for an e-commerce company.

**Dataset Overview:**
- Total Revenue: ${self.summary_stats['total_revenue']:,.2f}
- Total Cost: ${self.summary_stats['total_cost']:,.2f}
- Profit Margin: {(self.summary_stats['profit'] / self.summary_stats['total_revenue'] * 100):.1f}%
- Date Range: {self.summary_stats['date_range'][0]} to {self.summary_stats['date_range'][1]}
- Total Transactions: {self.summary_stats['row_count']}

**Top 5 Products by Revenue:**
{products_text}

**Revenue by Category:**
{categories_text}

**Revenue by Customer Segment:**
{segments_text}

**Monthly Revenue Trend:**
{months_text}

**Your Task:**
Provide a comprehensive analysis covering:

1. **Revenue Trends**: Analyze the month-over-month growth from July to September. Is revenue growing, declining, or stable? What's the percentage change?

2. **Top Performers**: Identify the top 3 products and explain why they might be performing well. Include specific dollar amounts.

3. **Category Performance**: Compare the three categories (Electronics, Accessories, Office). Which is strongest? Which needs attention?

4. **Customer Segments**: Analyze the distribution across Consumer, Business, and Education segments. What does this tell us about the customer base?

5. **Areas of Concern**: Are there any products with low sales? Any declining trends? Any categories underperforming?

6. **Q4 Recommendations**: Based on Q3 performance, what should the business focus on for Q4? Should they invest in certain products, target specific segments, or adjust pricing?

Provide your analysis in a clear, concise format with specific numbers and actionable insights. Keep it professional but accessible."""

        return prompt

    def analyze_with_gemini(self) -> Generator[str, None, Tuple[str, str, int]]:
        """
        Analyze sales data using Gemini API with streaming

        Yields:
            Chunks of the streaming response

        Returns (final):
            Tuple of (full_response, prompt_used, token_count)
        """
        # Prepare aggregated data
        agg_data = self._prepare_analysis_data()

        # Build prompt
        prompt = self._build_analysis_prompt(agg_data)

        # Stream response from Gemini
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                ),
                stream=True
            )

            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            # Estimate tokens
            tokens = estimate_tokens(prompt + full_response)
            self.total_tokens += tokens

            return full_response, prompt, tokens

        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            yield error_msg
            return error_msg, prompt, 0

    def forecast_with_prophet(self) -> Dict:
        """
        Forecast Q4 revenue using Prophet

        Returns:
            Dictionary with:
                - success: bool
                - message: str
                - q4_forecast: float (total predicted Q4 revenue)
                - q4_lower: float (lower confidence bound)
                - q4_upper: float (upper confidence bound)
                - growth_rate: float (Q4 vs Q3 growth %)
                - forecast_df: DataFrame (Prophet predictions)
        """
        try:
            # Prepare data for Prophet (daily revenue)
            daily_revenue = self.df.groupby('date')['revenue'].sum().reset_index()
            daily_revenue.columns = ['ds', 'y']

            # Initialize and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )

            model.fit(daily_revenue)

            # Create future dataframe for 90 days (Q4)
            future = model.make_future_dataframe(periods=90, freq='D')

            # Predict
            forecast = model.predict(future)

            # Extract Q4 predictions (last 90 days)
            q4_forecast = forecast.tail(90)

            # Calculate metrics
            q4_total = q4_forecast['yhat'].sum()
            q4_lower = q4_forecast['yhat_lower'].sum()
            q4_upper = q4_forecast['yhat_upper'].sum()
            q3_total = self.summary_stats['total_revenue']
            growth_rate = ((q4_total - q3_total) / q3_total) * 100

            return {
                'success': True,
                'message': 'Forecast completed successfully',
                'q4_forecast': q4_total,
                'q4_lower': q4_lower,
                'q4_upper': q4_upper,
                'growth_rate': growth_rate,
                'forecast_df': forecast,
                'q3_total': q3_total
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"Error in Prophet forecasting: {str(e)}"
            }

    def generate_visualizations(self, forecast_data: Dict) -> List[go.Figure]:
        """
        Generate 5 interactive Plotly visualizations

        Args:
            forecast_data: Output from forecast_with_prophet()

        Returns:
            List of 5 Plotly figures
        """
        figures = []

        # Prepare aggregated data
        agg_data = self._prepare_analysis_data()

        # 1. Revenue Trend Line Chart (Q3 Daily Revenue)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=agg_data['daily_revenue']['date'],
            y=agg_data['daily_revenue']['revenue'],
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

        # 2. Top 10 Products Horizontal Bar Chart
        top_products = agg_data['product_revenue'].head(10).sort_values('revenue')
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=top_products.index,
            x=top_products['revenue'],
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

        # 3. Category Performance Pie Chart
        fig3 = go.Figure(data=[go.Pie(
            labels=agg_data['category_revenue'].index,
            values=agg_data['category_revenue'].values,
            hole=0.3,
            marker=dict(colors=['#2563eb', '#3b82f6', '#60a5fa'])
        )])
        fig3.update_layout(
            title='Revenue by Category',
            template='plotly_white'
        )
        figures.append(fig3)

        # 4. Customer Segment Bar Chart
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=agg_data['segment_revenue'].index,
            y=agg_data['segment_revenue'].values,
            marker=dict(color='#2563eb')
        ))
        fig4.update_layout(
            title='Revenue by Customer Segment',
            xaxis_title='Customer Segment',
            yaxis_title='Revenue ($)',
            template='plotly_white'
        )
        figures.append(fig4)

        # 5. Q4 Forecast Chart with Confidence Bands
        if forecast_data['success']:
            forecast_df = forecast_data['forecast_df']

            # Separate Q3 actual and Q4 forecast
            q3_end = self.df['date'].max()
            q3_data = forecast_df[forecast_df['ds'] <= q3_end]
            q4_data = forecast_df[forecast_df['ds'] > q3_end]

            fig5 = go.Figure()

            # Q3 Actual
            fig5.add_trace(go.Scatter(
                x=q3_data['ds'],
                y=q3_data['yhat'],
                mode='lines',
                name='Q3 Actual',
                line=dict(color='#2563eb', width=2)
            ))

            # Q4 Forecast
            fig5.add_trace(go.Scatter(
                x=q4_data['ds'],
                y=q4_data['yhat'],
                mode='lines',
                name='Q4 Forecast',
                line=dict(color='#10b981', width=2, dash='dash')
            ))

            # Confidence interval
            fig5.add_trace(go.Scatter(
                x=q4_data['ds'].tolist() + q4_data['ds'].tolist()[::-1],
                y=q4_data['yhat_upper'].tolist() + q4_data['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(16, 185, 129, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True
            ))

            fig5.update_layout(
                title='Q3 Actual vs Q4 Forecast',
                xaxis_title='Date',
                yaxis_title='Revenue ($)',
                hovermode='x unified',
                template='plotly_white'
            )
            figures.append(fig5)

        return figures

    def calculate_total_cost(self) -> float:
        """
        Calculate total API cost for the analysis

        Returns:
            Total cost in USD
        """
        return calculate_gemini_cost(self.total_tokens, "gemini-2.5-flash")
