"""
Generate realistic sales data for Q3 2025 (July 1 - September 30)
This script creates the sales_2025_q3.csv file with 247 rows of realistic e-commerce data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Products with realistic pricing
PRODUCTS = [
    {"name": "Wireless Headphones", "category": "Electronics", "price": 79.99, "cost": 40.00},
    {"name": "USB-C Cable", "category": "Accessories", "price": 12.99, "cost": 3.50},
    {"name": "Laptop Stand", "category": "Office", "price": 49.99, "cost": 20.00},
    {"name": "Bluetooth Mouse", "category": "Electronics", "price": 34.99, "cost": 15.00},
    {"name": "Phone Case", "category": "Accessories", "price": 19.99, "cost": 5.00},
    {"name": "Desk Organizer", "category": "Office", "price": 24.99, "cost": 10.00},
    {"name": "Portable Charger", "category": "Electronics", "price": 39.99, "cost": 18.00},
    {"name": "HDMI Cable", "category": "Accessories", "price": 15.99, "cost": 4.00},
    {"name": "Ergonomic Keyboard", "category": "Office", "price": 89.99, "cost": 45.00},
    {"name": "Webcam HD", "category": "Electronics", "price": 69.99, "cost": 35.00},
]

CUSTOMER_SEGMENTS = ["Consumer", "Business", "Education"]

# Date range: July 1 - September 30, 2025 (92 days)
start_date = datetime(2025, 7, 1)
end_date = datetime(2025, 9, 30)
date_range = pd.date_range(start_date, end_date, freq='D')

# Generate sales data with intentional growth trend
sales_data = []

for i, date in enumerate(date_range):
    # Determine month for growth trend
    month = date.month

    # Growth factor: July (1.0) -> August (1.15) -> September (1.35)
    if month == 7:
        growth_factor = 1.0
    elif month == 8:
        growth_factor = 1.15
    else:  # September
        growth_factor = 1.35

    # Generate 13-18 transactions per day for realistic revenue (~$124K total)
    num_transactions = np.random.choice([13, 14, 15, 16, 17, 18], p=[0.15, 0.20, 0.25, 0.20, 0.12, 0.08])

    for _ in range(num_transactions):
        # Select random product with weighted probability (some products more popular)
        product_weights = [0.15, 0.12, 0.10, 0.13, 0.08, 0.07, 0.11, 0.09, 0.08, 0.07]
        product = np.random.choice(PRODUCTS, p=product_weights)

        # Determine quantity (1-8 items, mostly 1-3)
        quantity = np.random.choice([1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8], p=[0.35, 0.20, 0.15, 0.12, 0.08, 0.04, 0.025, 0.015, 0.01, 0.005, 0.005])

        # Apply growth factor to quantity for trending products
        if product['name'] in ['Wireless Headphones', 'Laptop Stand', 'Ergonomic Keyboard']:
            quantity = max(1, int(quantity * growth_factor))

        # Calculate revenue and cost
        revenue = product['price'] * quantity
        cost = product['cost'] * quantity

        # Select customer segment (Business more likely to buy in bulk)
        if quantity >= 3:
            segment = np.random.choice(CUSTOMER_SEGMENTS, p=[0.2, 0.6, 0.2])
        else:
            segment = np.random.choice(CUSTOMER_SEGMENTS, p=[0.6, 0.3, 0.1])

        sales_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'product': product['name'],
            'quantity': quantity,
            'revenue': round(revenue, 2),
            'cost': round(cost, 2),
            'category': product['category'],
            'customer_segment': segment
        })

# Create DataFrame
df = pd.DataFrame(sales_data)

# Shuffle to make it look more realistic
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

# Save to CSV
output_path = 'data/sales_2025_q3.csv'
df.to_csv(output_path, index=False)

# Print summary statistics
print(f"Sales data generated successfully!")
print(f"File saved to: {output_path}")
print(f"\nSummary Statistics:")
print(f"  Total rows: {len(df)}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Total revenue: ${df['revenue'].sum():,.2f}")
print(f"  Total cost: ${df['cost'].sum():,.2f}")
print(f"  Total profit: ${(df['revenue'].sum() - df['cost'].sum()):,.2f}")
print(f"\nRevenue by month:")
df['month'] = pd.to_datetime(df['date']).dt.month
monthly = df.groupby('month')['revenue'].sum()
print(f"  July (7): ${monthly.get(7, 0):,.2f}")
print(f"  August (8): ${monthly.get(8, 0):,.2f}")
print(f"  September (9): ${monthly.get(9, 0):,.2f}")
print(f"\nTop 5 products by revenue:")
top_products = df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(5)
for product, revenue in top_products.items():
    print(f"  {product}: ${revenue:,.2f}")
