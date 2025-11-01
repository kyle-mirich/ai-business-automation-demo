"""
Pydantic models for API request/response validation
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ========== RAG Chatbot Models ==========
class RAGQueryRequest(BaseModel):
    """Request model for RAG chatbot queries"""
    query: str = Field(..., description="User query", min_length=1)
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=[],
        description="Previous conversation messages"
    )
    top_k: Optional[int] = Field(default=3, description="Number of documents to retrieve", ge=1, le=10)


class SourceDocument(BaseModel):
    """Source document metadata"""
    content: str
    metadata: Dict[str, Any]
    page: Optional[int] = None
    source: Optional[str] = None


class RAGQueryResponse(BaseModel):
    """Response model for RAG chatbot queries"""
    answer: str
    sources: List[SourceDocument]
    tokens_used: Optional[int] = None


# ========== Support Ticket Models ==========
class SupportTicketRequest(BaseModel):
    """Request model for support ticket processing"""
    ticket_id: str = Field(..., description="Unique ticket identifier")
    subject: str = Field(..., description="Ticket subject", min_length=1)
    description: str = Field(..., description="Ticket description", min_length=1)
    customer_email: str = Field(..., description="Customer email")
    customer_name: Optional[str] = None


class SupportTicketResponse(BaseModel):
    """Response model for support ticket processing"""
    ticket_id: str
    category: str
    category_confidence: float
    priority: str
    priority_score: float
    priority_reasoning: str
    department: str
    response: str
    steps: List[Dict[str, Any]]
    tokens_used: int
    cost_usd: float


# ========== Inventory Models ==========
class InventoryAnalysisRequest(BaseModel):
    """Request model for inventory analysis"""
    # Optional: can pass CSV data directly or use file upload
    csv_data: Optional[str] = Field(None, description="CSV data as string")
    file_path: Optional[str] = Field(None, description="Path to inventory CSV file")


class InventoryItem(BaseModel):
    """Single inventory item with recommendations"""
    sku: str
    product_name: str
    current_stock: int
    reorder_point: int
    lead_time_days: int
    cost_per_unit: float
    last_30_days_sales: int
    category: str
    status: str
    forecast_30d: Optional[float] = None
    forecast_60d: Optional[float] = None
    forecast_90d: Optional[float] = None


class InventoryRecommendation(BaseModel):
    """AI-generated recommendation for inventory item"""
    sku: str
    product_name: str
    action: str
    reasoning: str
    urgency: str
    estimated_cost_impact: Optional[float] = None


class InventoryAnalysisResponse(BaseModel):
    """Response model for inventory analysis"""
    summary: Dict[str, Any]
    items: List[InventoryItem]
    recommendations: List[InventoryRecommendation]
    ai_insights: str
    tokens_used: int
    cost_usd: float


# ========== Health Check Models ==========
class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
