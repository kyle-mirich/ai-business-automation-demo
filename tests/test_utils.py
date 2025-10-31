"""
Unit tests for utility functions
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cost_calculator import estimate_tokens, calculate_gemini_cost
from utils.data_loader import load_sales_data, load_tickets, load_inventory


class TestCostCalculator:
    """Tests for cost calculation utilities"""

    def test_estimate_tokens_basic(self):
        """Test basic token estimation"""
        text = "Hello world"
        tokens = estimate_tokens(text)
        assert tokens == 3  # 2 words * 1.5 = 3 tokens

    def test_estimate_tokens_empty(self):
        """Test token estimation with empty string"""
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_estimate_tokens_long_text(self):
        """Test token estimation with longer text"""
        text = " ".join(["word"] * 100)  # 100 words
        tokens = estimate_tokens(text)
        assert tokens == 150  # 100 * 1.5

    def test_calculate_gemini_cost_input_only(self):
        """Test Gemini cost calculation with input tokens only"""
        result = calculate_gemini_cost(input_tokens=1_000_000, output_tokens=0)
        assert result['input_cost'] == 0.10  # $0.10 per 1M tokens
        assert result['output_cost'] == 0.0
        assert result['total_cost'] == 0.10

    def test_calculate_gemini_cost_output_only(self):
        """Test cost calculation with output tokens only"""
        result = calculate_gemini_cost(input_tokens=0, output_tokens=1_000_000)
        assert result['output_cost'] == 0.40  # $0.40 per 1M tokens
        assert result['input_cost'] == 0.0
        assert result['total_cost'] == 0.40

    def test_calculate_gemini_cost_mixed(self):
        """Test cost calculation with both input and output tokens"""
        result = calculate_gemini_cost(input_tokens=500_000, output_tokens=500_000)
        assert result['input_cost'] == 0.05   # 500K * $0.10/1M
        assert result['output_cost'] == 0.20  # 500K * $0.40/1M
        assert result['total_cost'] == 0.25

    def test_calculate_gemini_cost_rounding(self):
        """Test that cost is properly rounded"""
        result = calculate_gemini_cost(input_tokens=1234, output_tokens=5678)
        assert isinstance(result['total_cost'], float)
        # Should be rounded to 6 decimal places
        assert len(str(result['total_cost']).split('.')[-1]) <= 6


class TestDataLoader:
    """Tests for data loading utilities"""

    def test_load_sales_data(self):
        """Test loading sales data"""
        df = load_sales_data()

        assert df is not None
        assert len(df) > 0

        # Check required columns exist
        required_cols = ['date', 'product', 'revenue', 'quantity']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_tickets(self):
        """Test loading support tickets"""
        tickets = load_tickets()

        assert tickets is not None
        assert isinstance(tickets, list)
        assert len(tickets) > 0

        # Check first ticket has required fields
        ticket = tickets[0]
        required_fields = ['id', 'subject', 'description']
        for field in required_fields:
            assert field in ticket, f"Missing field: {field}"

    def test_load_inventory(self):
        """Test loading inventory data"""
        df = load_inventory()

        assert df is not None
        assert len(df) > 0

        # Check required columns
        required_cols = ['sku', 'product_name', 'current_stock']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_sales_data_types(self):
        """Test that sales data has correct types"""
        df = load_sales_data()

        # Date should be datetime
        assert df['date'].dtype == 'datetime64[ns]'

        # Revenue and quantity should be numeric
        assert df['revenue'].dtype in ['float64', 'int64']
        assert df['quantity'].dtype in ['float64', 'int64']

    def test_inventory_stock_positive(self):
        """Test that inventory stock levels are non-negative"""
        df = load_inventory()
        assert (df['current_stock'] >= 0).all(), "Found negative stock levels"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
