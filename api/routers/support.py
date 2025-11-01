"""
Support Ticket API Router
"""

from fastapi import APIRouter, HTTPException
import os

from api.models import SupportTicketRequest, SupportTicketResponse
from agents.support_agents import SupportAgentOrchestrator
from utils.cost_calculator import calculate_gemini_cost

router = APIRouter()

# Initialize support orchestrator (singleton)
_support_orchestrator = None


def get_support_orchestrator() -> SupportAgentOrchestrator:
    """Get or create support orchestrator instance"""
    global _support_orchestrator
    if _support_orchestrator is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

        _support_orchestrator = SupportAgentOrchestrator(
            api_key=api_key,
            model="gemini-2.5-flash"
        )
    return _support_orchestrator


@router.post("/triage", response_model=SupportTicketResponse)
async def triage_ticket(request: SupportTicketRequest):
    """
    Process and triage a support ticket

    Returns classification, priority, routing, and AI-generated response
    """
    try:
        orchestrator = get_support_orchestrator()

        # Convert request to ticket dict with proper format
        ticket = {
            "ticket_id": request.ticket_id,
            "subject": request.subject,
            "message": f"{request.subject}\n\n{request.description}",  # Combine subject and description
            "customer": {
                "name": request.customer_name or "Valued Customer",
                "email": request.customer_email,
                "tier": "standard"
            }
        }

        # Process ticket
        result = orchestrator.run(ticket)

        # Calculate cost
        tokens_used = result.get("tokens_used", 0)
        cost_info = calculate_gemini_cost(output_tokens=tokens_used, model="gemini-2.5-flash")
        cost_usd = cost_info['total_cost']

        return SupportTicketResponse(
            ticket_id=result.get("ticket", {}).get("ticket_id", request.ticket_id),
            category=result.get("category", "unknown"),
            category_confidence=result.get("category_confidence", 0.0),
            priority=result.get("priority", "medium"),
            priority_score=result.get("priority_score", 0.5),
            priority_reasoning=result.get("priority_reasoning", ""),
            department=result.get("department", "general"),
            response=result.get("response", ""),
            steps=result.get("steps", []),
            tokens_used=tokens_used,
            cost_usd=cost_usd
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ticket: {str(e)}")


@router.get("/health")
async def health():
    """Check if support orchestrator is initialized"""
    try:
        orchestrator = get_support_orchestrator()
        return {
            "status": "healthy",
            "model": orchestrator.model
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
