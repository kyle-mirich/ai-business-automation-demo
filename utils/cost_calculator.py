"""
Cost calculation utilities for tracking API usage
"""

def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text using word count heuristic

    Args:
        text: Input text string

    Returns:
        Estimated token count (word_count * 1.5)
    """
    if not text:
        return 0
    word_count = len(text.split())
    return int(word_count * 1.5)


def calculate_gemini_cost(tokens: int, model: str = "gemini-2.5-flash-lite") -> float:
    """
    Calculate API cost based on token usage

    Args:
        tokens: Number of tokens used
        model: Model name (gemini-2.5-flash-lite or gemini-1.5-flash)

    Returns:
        Cost in USD
    """
    # Pricing per 1K tokens (approximate)
    pricing = {
        "gemini-2.5-flash-lite": 0.001,      # $0.001 per 1K tokens
        "gemini-2.5-flash-lite": 0.0005,   # $0.0005 per 1K tokens
    }

    rate = pricing.get(model, 0.001)
    cost = (tokens / 1000) * rate
    return round(cost, 4)
