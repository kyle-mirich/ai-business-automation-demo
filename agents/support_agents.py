"""
Support ticket orchestration with LangGraph and Gemini

This module defines a SupportAgentOrchestrator that coordinates four agents:
- Classifier Agent (LLM)
- Prioritizer Agent (LLM)
- Router Agent (rule based)
- Response Agent (LLM)
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph

from utils.cost_calculator import calculate_gemini_cost, estimate_tokens


class TicketState(TypedDict, total=False):
    """Graph state for a support ticket run."""

    ticket: Dict[str, Any]
    category: str
    category_confidence: float
    priority: str
    priority_score: float
    priority_reasoning: str
    department: str
    response: str
    steps: List[Dict[str, Any]]
    tokens_used: int


@dataclass
class TokenUsage:
    """Aggregate token usage across all agent calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self, model: str) -> Dict[str, Any]:
        # Use new accurate cost calculator
        cost_info = calculate_gemini_cost(
            input_tokens=self.prompt_tokens,
            output_tokens=self.completion_tokens,
            model=model
        )
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": cost_info['total_cost'],
        }


class SupportAgentOrchestrator:
    """Coordinates support ticket processing using LangGraph."""

    _allowed_categories = {
        "SHIPPING_INQUIRY",
        "REFUND_REQUEST",
        "ADDRESS_CHANGE",
        "BULK_SALES",
        "TECHNICAL_ISSUE",
        "PRODUCT_QUESTION",
        "ACCOUNT_ISSUE",
        "GENERAL_INQUIRY",
    }

    _priority_levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    _department_map = {
        "SHIPPING_INQUIRY": "Logistics",
        "REFUND_REQUEST": "Billing & Refunds",
        "ADDRESS_CHANGE": "Customer Success",
        "BULK_SALES": "Sales",
        "TECHNICAL_ISSUE": "Engineering Support",
        "PRODUCT_QUESTION": "Product Education",
        "ACCOUNT_ISSUE": "Account Services",
        "GENERAL_INQUIRY": "Customer Success",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
    ):
        if not api_key:
            raise ValueError("Google Gemini API key is required for SupportAgentOrchestrator.")

        self.model = model  # Add model attribute for API compatibility
        self.model_name = model
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            convert_system_message_to_human=True,
        )

        self.usage = TokenUsage()
        self.graph = self._build_graph()
        self.run_history: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def reset_usage(self) -> None:
        """Reset aggregated usage metrics."""
        self.usage = TokenUsage()
        self.run_history.clear()

    def process_ticket(self, ticket: Dict[str, Any]) -> TicketState:
        """Run a single ticket through the LangGraph workflow."""
        initial_state: TicketState = {
            "ticket": ticket,
            "steps": [],
            "tokens_used": 0,
        }

        result: TicketState = self.graph.invoke(initial_state)
        result["ticket"] = ticket
        step_tokens = 0
        for step in result.get("steps", []) or []:
            usage = step.get("token_usage", {})
            step_tokens += usage.get("total_tokens", 0)
        result["tokens_used"] = step_tokens
        self.run_history.append(
            {
                "ticket_id": ticket.get("id"),
                "customer": ticket.get("customer", {}).get("name"),
                "steps": deepcopy(result.get("steps", [])),
                "result": deepcopy(result),
            }
        )
        return result

    def process_tickets(self, tickets: List[Dict[str, Any]]) -> List[TicketState]:
        """Process a list of tickets sequentially."""
        results = []
        for ticket in tickets:
            results.append(self.process_ticket(ticket))
        return results

    def classify_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Expose classifier output for reuse."""
        classification = self._run_classifier(ticket)
        return {
            "category": classification["category"],
            "confidence": classification["category_confidence"],
            "rationale": classification["steps"][-1]["details"],
        }

    def calculate_cost(self) -> Dict[str, Any]:
        """Return aggregated token usage and estimated cost."""
        return self.usage.to_dict(self.model_name)

    # -------------------------------------------------------------------------
    # LangGraph assembly
    # -------------------------------------------------------------------------
    def _build_graph(self):
        """Create and compile the LangGraph workflow."""
        graph = StateGraph(TicketState)

        graph.add_node("classifier", self._classifier_node)
        graph.add_node("prioritizer", self._prioritizer_node)
        graph.add_node("router", self._router_node)
        graph.add_node("responder", self._response_node)

        graph.set_entry_point("classifier")
        graph.add_edge("classifier", "prioritizer")
        graph.add_edge("prioritizer", "router")
        graph.add_edge("router", "responder")
        graph.set_finish_point("responder")

        return graph.compile()

    # -------------------------------------------------------------------------
    # Graph nodes
    # -------------------------------------------------------------------------
    def _classifier_node(self, state: TicketState) -> TicketState:
        return self._run_classifier(state["ticket"], state.get("steps"))

    def _prioritizer_node(self, state: TicketState) -> TicketState:
        return self._run_prioritizer(state["ticket"], state["category"], state.get("steps"))

    def _router_node(self, state: TicketState) -> TicketState:
        return self._run_router(state["category"], state.get("steps"))

    def _response_node(self, state: TicketState) -> TicketState:
        return self._run_responder(state, state.get("steps"))

    # -------------------------------------------------------------------------
    # Agent logic
    # -------------------------------------------------------------------------
    def _run_classifier(
        self,
        ticket: Dict[str, Any],
        existing_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> TicketState:
        prompt = f"""You are a senior customer support classifier.
Analyze the ticket message and choose ONE category from this list:
- SHIPPING_INQUIRY
- REFUND_REQUEST
- ADDRESS_CHANGE
- BULK_SALES
- TECHNICAL_ISSUE
- PRODUCT_QUESTION
- ACCOUNT_ISSUE
If none apply directly, use GENERAL_INQUIRY.

Return a JSON object with:
- category (uppercase string)
- confidence (0-100 integer)
- rationale (short sentence explaining the decision)

Ticket message:
\"\"\"{ticket.get('message', '').strip()}\"\"\""""

        response_text, token_info = self._invoke_llm(prompt, "Classifier Agent")
        parsed = self._parse_json(response_text)

        category = self._normalize_category(parsed.get("category"))
        confidence = float(parsed.get("confidence", 70))
        rationale = parsed.get("rationale", "").strip()

        steps = self._append_step(
            existing_steps,
            agent="Classifier Agent",
            summary=f"Category → {category}",
            details=rationale or "Assigned default category",
            token_info=token_info,
        )

        return {
            "category": category,
            "category_confidence": confidence,
            "steps": steps,
        }

    def _run_prioritizer(
        self,
        ticket: Dict[str, Any],
        category: str,
        existing_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> TicketState:
        customer = ticket.get("customer", {})
        prompt = f"""You are a customer support lead prioritizing tickets.
Assess urgency based on the category, message tone, customer tier, and business impact.

Choose one priority: CRITICAL, HIGH, MEDIUM, LOW.
Return JSON with:
- priority (uppercase string)
- score (0-100 integer representing urgency)
- reasoning (concise explanation)

Ticket message:
\"\"\"{ticket.get('message', '').strip()}\"\"\"

Category: {category}
Customer tier: {customer.get('tier', 'standard')}
"""

        response_text, token_info = self._invoke_llm(prompt, "Prioritizer Agent")
        parsed = self._parse_json(response_text)

        priority = self._normalize_priority(parsed.get("priority"))
        score = float(parsed.get("score", 60))
        reasoning = parsed.get("reasoning", "").strip()

        steps = self._append_step(
            existing_steps,
            agent="Prioritizer Agent",
            summary=f"Priority → {priority}",
            details=reasoning or "Assigned default priority",
            token_info=token_info,
        )

        return {
            "priority": priority,
            "priority_score": score,
            "priority_reasoning": reasoning,
            "steps": steps,
        }

    def _run_router(
        self,
        category: str,
        existing_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> TicketState:
        department = self._department_map.get(category, "Customer Success")
        steps = self._append_step(
            existing_steps,
            agent="Router Agent",
            summary=f"Department → {department}",
            details=f"Mapped from category {category}",
            token_info=None,
        )
        return {
            "department": department,
            "steps": steps,
        }

    def _run_responder(
        self,
        state: TicketState,
        existing_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> TicketState:
        ticket = state["ticket"]
        prompt = f"""You are an experienced customer support specialist.
Draft a concise, empathetic reply (3-4 sentences) referencing the customer's issue.
Match the urgency of the priority level.
Provide clear next steps the company will take and what the customer should expect.
Avoid markdown, lists, or bullet points.

Ticket Details:
- Customer name: {ticket.get('customer', {}).get('name', 'Customer')}
- Category: {state.get('category')}
- Priority: {state.get('priority')}
- Department: {state.get('department')}
- Message: \"\"\"{ticket.get('message', '').strip()}\"\"\"
"""

        response_text, token_info = self._invoke_llm(prompt, "Response Agent")
        cleaned_response = self._clean_response(response_text)

        steps = self._append_step(
            existing_steps,
            agent="Response Agent",
            summary="Drafted customer reply",
            details=cleaned_response,
            token_info=token_info,
        )

        return {
            "response": cleaned_response,
            "steps": steps,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _invoke_llm(self, prompt: str, agent_name: str) -> Tuple[str, Dict[str, int]]:
        """Call Gemini model and keep usage metrics."""
        prompt_tokens = estimate_tokens(prompt)
        message = self.llm.invoke(prompt)
        response_text = self._extract_text(message)
        completion_tokens = estimate_tokens(response_text)

        token_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens

        return response_text, token_info

    def _extract_text(self, message: Any) -> str:
        """Extract plain text from LangChain AIMessage."""
        if message is None:
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()

        text = getattr(message, "text", "")
        return str(text).strip()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Attempt to parse JSON from model response."""
        if not text:
            return {}

        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                snippet = text[start : end + 1]
                return json.loads(snippet)
        except json.JSONDecodeError:
            pass

        return {}

    def _append_step(
        self,
        existing_steps: Optional[List[Dict[str, Any]]],
        agent: str,
        summary: str,
        details: str,
        token_info: Optional[Dict[str, int]],
    ) -> List[Dict[str, Any]]:
        """Append a step log entry."""
        steps = list(existing_steps) if existing_steps else []
        entry = {
            "agent": agent,
            "summary": summary,
            "details": details.strip(),
        }
        if token_info:
            entry["token_usage"] = token_info
        steps.append(entry)
        return steps

    def _normalize_category(self, value: Optional[str]) -> str:
        if not value:
            return "GENERAL_INQUIRY"
        candidate = value.strip().upper()
        if candidate not in self._allowed_categories:
            return "GENERAL_INQUIRY"
        return candidate

    def _normalize_priority(self, value: Optional[str]) -> str:
        if not value:
            return "MEDIUM"
        candidate = value.strip().upper()
        if candidate not in self._priority_levels:
            return "MEDIUM"
        return candidate

    def _clean_response(self, text: str) -> str:
        """Normalize whitespace and remove markdown artifacts."""
        if not text:
            return ""
        cleaned = text.replace("*", "")
        cleaned = cleaned.replace("\u2028", " ").replace("\u2029", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned.strip()

    # -------------------------------------------------------------------------
    # API compatibility method
    # -------------------------------------------------------------------------
    def run(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """
        API-compatible wrapper for process_ticket

        Args:
            ticket: Ticket dict with required fields (ticket_id, subject, description, customer_email, etc.)

        Returns:
            Result dict with all processing information
        """
        return self.process_ticket(ticket)
