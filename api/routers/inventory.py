"""
Inventory Optimization API Router
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import os
import pandas as pd
from io import StringIO

from api.models import (
    InventoryAnalysisRequest,
    InventoryAnalysisResponse,
    InventoryItem,
    InventoryRecommendation
)
from agents.inventory_agent import InventoryAgent
from utils.cost_calculator import calculate_gemini_cost

router = APIRouter()


def get_inventory_agent() -> InventoryAgent:
    """Create inventory agent instance"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

    return InventoryAgent(
        api_key=api_key,
        model="gemini-2.5-flash"
    )


@router.post("/analyze", response_model=InventoryAnalysisResponse)
async def analyze_inventory(request: InventoryAnalysisRequest):
    """
    Analyze inventory data and generate recommendations

    Accepts either CSV data as string or file path
    """
    try:
        agent = get_inventory_agent()

        # Load data
        if request.csv_data:
            df = pd.read_csv(StringIO(request.csv_data))
        elif request.file_path:
            if not os.path.exists(request.file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
            df = pd.read_csv(request.file_path)
        else:
            # Use default data file
            default_path = os.path.join(
                os.path.dirname(__file__),
                "../../data/inventory.csv"
            )
            if not os.path.exists(default_path):
                raise HTTPException(
                    status_code=400,
                    detail="No data provided and default file not found"
                )
            df = pd.read_csv(default_path)

        # Run analysis
        result = agent.run(df)

        # Format items
        items = []
        for _, row in result["analyzed_data"].iterrows():
            items.append(InventoryItem(
                sku=row["sku"],
                product_name=row["product_name"],
                current_stock=int(row["current_stock"]),
                reorder_point=int(row["reorder_point"]),
                lead_time_days=int(row["lead_time_days"]),
                cost_per_unit=float(row["cost_per_unit"]),
                last_30_days_sales=int(row["last_30_days_sales"]),
                category=row["category"],
                status=row["status"],
                forecast_30d=float(row.get("forecast_30d", 0)),
                forecast_60d=float(row.get("forecast_60d", 0)),
                forecast_90d=float(row.get("forecast_90d", 0))
            ))

        # Format recommendations
        recommendations = []
        for rec in result.get("recommendations", []):
            recommendations.append(InventoryRecommendation(
                sku=rec.get("sku", ""),
                product_name=rec.get("product_name", ""),
                action=rec.get("action", ""),
                reasoning=rec.get("reasoning", ""),
                urgency=rec.get("urgency", "medium"),
                estimated_cost_impact=rec.get("estimated_cost_impact")
            ))

        # Calculate cost
        tokens_used = result.get("token_usage", {}).get("total_tokens", 0)
        cost_info = calculate_gemini_cost(output_tokens=tokens_used, model="gemini-2.5-flash")
        cost_usd = cost_info['total_cost']

        return InventoryAnalysisResponse(
            summary=result.get("summary", {}),
            items=items,
            recommendations=recommendations,
            ai_insights=result.get("ai_insights", ""),
            tokens_used=tokens_used,
            cost_usd=cost_usd
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing inventory: {str(e)}")


@router.post("/analyze/upload")
async def analyze_inventory_upload(file: UploadFile = File(...)):
    """
    Analyze inventory from uploaded CSV file
    """
    try:
        # Read uploaded file
        contents = await file.read()
        csv_data = contents.decode("utf-8")

        # Use the analyze endpoint
        request = InventoryAnalysisRequest(csv_data=csv_data)
        return await analyze_inventory(request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@router.get("/health")
async def health():
    """Check if inventory agent can be initialized"""
    try:
        agent = get_inventory_agent()
        return {
            "status": "healthy",
            "model": agent.model
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
