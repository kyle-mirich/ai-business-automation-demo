"""
Quick test script to verify the setup
"""

import sys
from pathlib import Path

print("Testing AI Business Automation Demo Platform setup...\n")

# Test 1: Check Python version
print(f"[OK] Python version: {sys.version.split()[0]}")

# Test 2: Check data file exists
data_path = Path("data/sales_2025_q3.csv")
if data_path.exists():
    print(f"[OK] Sales data file exists: {data_path}")
else:
    print(f"[ERROR] Sales data file missing: {data_path}")
    sys.exit(1)

# Test 3: Check environment file
env_path = Path(".env")
if env_path.exists():
    print(f"[OK] Environment file exists: {env_path}")
    with open(env_path) as f:
        content = f.read()
        if "GOOGLE_API_KEY" in content and "your_api_key_here" not in content:
            print("[OK] API key configured")
        else:
            print("[WARN] API key not configured - add your Gemini API key to .env")
else:
    print(f"[ERROR] Environment file missing: {env_path}")

# Test 4: Test imports
print("\nTesting imports...")
try:
    import pandas as pd
    print("[OK] pandas")
except ImportError:
    print("[ERROR] pandas - run: pip install pandas")

try:
    import numpy as np
    print("[OK] numpy")
except ImportError:
    print("[ERROR] numpy - run: pip install numpy")

try:
    import plotly
    print("[OK] plotly")
except ImportError:
    print("[ERROR] plotly - run: pip install plotly")

try:
    import streamlit
    print("[OK] streamlit")
except ImportError:
    print("[ERROR] streamlit - run: pip install streamlit")

try:
    import prophet
    print("[OK] prophet")
except ImportError:
    print("[ERROR] prophet - run: pip install prophet")

try:
    import google.generativeai as genai
    print("[OK] google-generativeai")
except ImportError:
    print("[ERROR] google-generativeai - run: pip install google-generativeai")

try:
    import langgraph
    print("[OK] langgraph")
except ImportError:
    print("[ERROR] langgraph - run: pip install langgraph")

try:
    import langchain
    print("[OK] langchain")
except ImportError:
    print("[ERROR] langchain - run: pip install langchain")

try:
    from dotenv import load_dotenv
    print("[OK] python-dotenv")
except ImportError:
    print("[ERROR] python-dotenv - run: pip install python-dotenv")

# Test 5: Test agent import
print("\nTesting agent imports...")
try:
    from agents.financial_agent_langchain import FinancialAgentLangChain
    print("[OK] FinancialAgentLangChain imported successfully")
except ImportError as e:
    print(f"[ERROR] FinancialAgentLangChain import failed: {e}")

try:
    from agents.support_agents import SupportAgentOrchestrator
    print("[OK] SupportAgentOrchestrator imported successfully")
except ImportError as e:
    print(f"[ERROR] SupportAgentOrchestrator import failed: {e}")

try:
    from agents.inventory_agent import InventoryAgent
    print("[OK] InventoryAgent imported successfully")
except ImportError as e:
    print(f"[ERROR] InventoryAgent import failed: {e}")

# Test 6: Test utility imports
try:
    from utils.data_loader import load_sales_data
    from utils.cost_calculator import calculate_gemini_cost
    print("[OK] Utility functions imported successfully")
except ImportError as e:
    print(f"[ERROR] Utility import failed: {e}")

print("\n" + "="*50)
print("Setup test complete!")
print("="*50)
print("\nTo run the app:")
print("  streamlit run Home.py")
