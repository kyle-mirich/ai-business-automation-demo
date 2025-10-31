"""
Cost calculation utilities for tracking API usage

Pricing based on Google Gemini API (as of 2024)
Standard pricing per 1 million tokens
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


def calculate_gemini_cost(
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "gemini-2.5-flash",
    use_thinking: bool = False
) -> dict:
    """
    Calculate API cost based on token usage using accurate Gemini pricing

    Pricing (per 1 million tokens):
    - Input (text/image/video): $0.10
    - Output: $0.40
    - Audio input: $0.30
    - Context caching: $0.01 (text/image/video), $0.03 (audio)

    Args:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used (includes thinking tokens)
        model: Model name (for future model-specific pricing)
        use_thinking: Whether this includes thinking tokens (doesn't affect price)

    Returns:
        Dictionary with cost breakdown:
        {
            'input_cost': float,
            'output_cost': float,
            'total_cost': float,
            'input_tokens': int,
            'output_tokens': int
        }
    """
    # Pricing per 1 million tokens (standard paid tier)
    INPUT_PRICE_PER_MILLION = 0.10   # $0.10 per 1M tokens
    OUTPUT_PRICE_PER_MILLION = 0.40  # $0.40 per 1M tokens (includes thinking)

    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost

    return {
        'input_cost': round(input_cost, 6),
        'output_cost': round(output_cost, 6),
        'total_cost': round(total_cost, 6),
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'model': model
    }


def format_cost_breakdown(cost_info: dict) -> str:
    """
    Format cost information as a human-readable string

    Args:
        cost_info: Dictionary from calculate_gemini_cost()

    Returns:
        Formatted string with cost breakdown
    """
    return f"""
💰 **Cost Breakdown**
- Input: {cost_info['input_tokens']:,} tokens → ${cost_info['input_cost']:.6f}
- Output: {cost_info['output_tokens']:,} tokens → ${cost_info['output_cost']:.6f}
- **Total: ${cost_info['total_cost']:.6f}**
- Model: {cost_info['model']}
    """.strip()


# Legacy function for backward compatibility
def calculate_gemini_cost_legacy(tokens: int, model: str = "gemini-2.5-flash") -> float:
    """
    Legacy cost calculation (deprecated - use calculate_gemini_cost instead)

    This assumes all tokens are output tokens for backward compatibility

    Args:
        tokens: Number of tokens used
        model: Model name

    Returns:
        Cost in USD
    """
    cost_info = calculate_gemini_cost(input_tokens=0, output_tokens=tokens, model=model)
    return cost_info['total_cost']
