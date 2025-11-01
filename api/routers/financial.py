"""
Financial Report API Router
Provides endpoints for the Financial Agent with chat, report generation, and visualizations
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, AsyncGenerator
import os
from pathlib import Path
import json

from agents.financial_agent_langchain import FinancialAgentLangChain

router = APIRouter()

# Initialize financial agent (singleton)
_financial_agent = None


class ChatMessage(BaseModel):
    role: str
    content: str


class FinancialChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[ChatMessage]] = []


class FinancialChatResponse(BaseModel):
    response: str
    tools_used: List[Dict[str, Any]]
    citation_dataframe: Optional[Dict[str, Any]] = None
    pandas_code: Optional[str] = None
    cost_info: Dict[str, Any]


class ReportGenerationResponse(BaseModel):
    response: str
    intermediate_steps: List[Dict[str, Any]]
    cost_info: Dict[str, Any]


def get_financial_agent() -> FinancialAgentLangChain:
    """Get or create financial agent instance"""
    global _financial_agent
    if _financial_agent is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

        data_path = os.path.join(os.path.dirname(__file__), "../../data/sales_2025_q3.csv")
        _financial_agent = FinancialAgentLangChain(api_key=api_key, data_path=data_path)

        # Auto-load data
        try:
            result = _financial_agent.load_data()
            if not result.get('success'):
                print(f"Warning: Could not preload financial data: {result.get('message')}")
        except Exception as e:
            print(f"Warning: Could not preload financial data: {str(e)}")

    return _financial_agent


@router.post("/load")
async def load_data():
    """Load sales data"""
    try:
        agent = get_financial_agent()
        result = agent.load_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@router.get("/stats")
async def get_stats():
    """Get summary statistics"""
    try:
        agent = get_financial_agent()
        if agent.df is None:
            return {"loaded": False, "message": "Data not loaded"}

        return {
            "loaded": True,
            **agent.summary_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@router.post("/chat", response_model=FinancialChatResponse)
async def chat(request: FinancialChatRequest):
    """
    Chat with the financial agent
    Returns answer with tool usage and citations
    """
    try:
        agent = get_financial_agent()

        # Ensure data is loaded
        if agent.df is None:
            load_result = agent.load_data()
            if not load_result.get('success'):
                raise HTTPException(status_code=500, detail=f"Could not load data: {load_result.get('message')}")

        # Convert chat history
        agent.chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.chat_history
        ]

        # Get response from agent
        result = agent.chat(request.message)

        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('response', 'Error processing chat'))

        # Calculate cost
        from utils.cost_calculator import estimate_tokens, calculate_gemini_cost
        input_tokens = estimate_tokens(request.message)
        output_tokens = estimate_tokens(result['response'])
        cost_info = calculate_gemini_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model="gemini-2.5-flash"
        )

        # Format citation dataframe if available
        citation_df = result.get('citation_dataframe')
        citation_dict = None
        if citation_df is not None and not citation_df.empty:
            citation_dict = citation_df.to_dict(orient='split')

        return FinancialChatResponse(
            response=result['response'],
            tools_used=result.get('intermediate_steps', []),
            citation_dataframe=citation_dict,
            pandas_code=result.get('pandas_code'),
            cost_info=cost_info
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@router.post("/generate-report", response_model=ReportGenerationResponse)
async def generate_report():
    """Generate a comprehensive financial report"""
    try:
        agent = get_financial_agent()

        # Ensure data is loaded
        if agent.df is None:
            load_result = agent.load_data()
            if not load_result.get('success'):
                raise HTTPException(status_code=500, detail=f"Could not load data: {load_result.get('message')}")

        # Generate report
        result = agent.generate_comprehensive_report()

        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('response', 'Error generating report'))

        # Calculate cost
        from utils.cost_calculator import estimate_tokens, calculate_gemini_cost
        report_prompt = "Analyze the sales data and generate a comprehensive financial report with insights, trends, and recommendations."
        input_tokens = estimate_tokens(report_prompt)
        output_tokens = estimate_tokens(result['response'])
        cost_info = calculate_gemini_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model="gemini-2.5-flash"
        )

        return ReportGenerationResponse(
            response=result['response'],
            intermediate_steps=result.get('intermediate_steps', []),
            cost_info=cost_info
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@router.get("/visualizations")
async def get_visualizations():
    """Generate visualizations"""
    try:
        agent = get_financial_agent()

        # Ensure data is loaded
        if agent.df is None:
            load_result = agent.load_data()
            if not load_result.get('success'):
                raise HTTPException(status_code=500, detail=f"Could not load data: {load_result.get('message')}")

        # Generate visualizations
        figures = agent.generate_visualizations()

        # Convert Plotly figures to JSON
        fig_jsons = [fig.to_json() for fig in figures]

        return {"figures": fig_jsons}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {str(e)}")


@router.get("/health")
async def health():
    """Check if financial agent is initialized"""
    try:
        agent = get_financial_agent()
        loaded = agent.df is not None
        return {
            "status": "healthy" if loaded else "data_not_loaded",
            "data_loaded": loaded,
            "row_count": agent.summary_stats.get('row_count', 0) if loaded else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
