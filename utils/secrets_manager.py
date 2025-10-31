"""
Secrets management utility for Streamlit apps

Handles API keys from multiple sources:
1. Streamlit secrets (st.secrets) - for Streamlit Cloud deployment
2. Environment variables (.env) - for local development
"""

import os
from typing import Optional
import streamlit as st
from dotenv import load_dotenv


def get_api_key(key_name: str = "GOOGLE_API_KEY") -> Optional[str]:
    """
    Get API key from Streamlit secrets or environment variables

    Priority:
    1. st.secrets (Streamlit Cloud)
    2. Environment variables (.env file)

    Args:
        key_name: Name of the API key to retrieve

    Returns:
        API key string or None if not found
    """
    # Try Streamlit secrets first (for deployed apps)
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            api_key = st.secrets[key_name]
            if api_key and api_key != "your_api_key_here":
                return api_key
    except Exception:
        # st.secrets might not be available in all contexts
        pass

    # Fall back to environment variables (for local development)
    load_dotenv()
    api_key = os.getenv(key_name)

    if api_key and api_key != "your_api_key_here":
        return api_key

    return None


def check_api_key_configured(key_name: str = "GOOGLE_API_KEY") -> tuple[bool, str]:
    """
    Check if API key is properly configured

    Args:
        key_name: Name of the API key to check

    Returns:
        Tuple of (is_configured: bool, source: str)
        source can be: "streamlit_secrets", "environment", or "not_found"
    """
    # Check Streamlit secrets
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            api_key = st.secrets[key_name]
            if api_key and api_key != "your_api_key_here":
                return (True, "streamlit_secrets")
    except Exception:
        pass

    # Check environment variables
    load_dotenv()
    api_key = os.getenv(key_name)
    if api_key and api_key != "your_api_key_here":
        return (True, "environment")

    return (False, "not_found")


def display_api_key_error():
    """
    Display helpful error message when API key is not configured
    """
    st.error("⚠️ Google Gemini API key not configured!")

    # Check which setup method to recommend
    is_cloud = os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud"

    if is_cloud:
        st.info("""
        **Streamlit Cloud Setup:**
        1. Go to your app settings
        2. Navigate to "Secrets" section
        3. Add the following:
        ```toml
        GOOGLE_API_KEY = "your_actual_api_key_here"
        ```
        4. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        """)
    else:
        st.info("""
        **Local Development Setup:**

        **Option 1: Using .env file (recommended for local)**
        1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Add it to the `.env` file: `GOOGLE_API_KEY=your_key_here`
        3. Restart the Streamlit app

        **Option 2: Using Streamlit secrets (for testing deployment locally)**
        1. Create `.streamlit/secrets.toml` file
        2. Add: `GOOGLE_API_KEY = "your_key_here"`
        3. Restart the Streamlit app

        **Note:** Make sure `.env` and `.streamlit/secrets.toml` are in your `.gitignore`!
        """)
