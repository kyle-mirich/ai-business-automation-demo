"""
Inventory optimization agent powered by LangChain + Gemini.

This agent:
- Loads pre-generated inventory data
- Spots low stock, overstock, and fast-moving items
- Builds lightweight Prophet forecasts for each SKU
- Calls Gemini (via LangChain) to recommend reorder actions
- Tracks estimated token usage and cost for transparency
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.cost_calculator import calculate_gemini_cost, estimate_tokens


REQUIRED_COLUMNS = {
    "sku",
    "product_name",
    "current_stock",
    "reorder_point",
    "lead_time_days",
    "cost_per_unit",
    "last_30_days_sales",
    "category",
}


@dataclass
class TokenUsage:
    """Simple token usage tracker for Gemini calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "gemini-2.0-flash"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost(self) -> float:
        # Use new accurate cost calculator
        cost_info = calculate_gemini_cost(
            input_tokens=self.prompt_tokens,
            output_tokens=self.completion_tokens,
            model=self.model
        )
        return cost_info['total_cost']

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "model": self.model,
        }


class InventoryAgent:
    """Inventory analytics + recommendation engine."""

    def __init__(
        self,
        api_key: str,
        data_path: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.4,
    ):
        if not api_key:
            raise ValueError("Google Gemini API key is required for InventoryAgent.")

        self.data_path = data_path
        self.model_name = model
        self.temperature = temperature
        self.df: Optional[pd.DataFrame] = None
        self.analysis_result: Optional[Dict[str, Any]] = None
        self.forecast_result: Optional[pd.DataFrame] = None
        self.forecast_timelines: Dict[str, pd.DataFrame] = {}
        self.usage = TokenUsage(model=model)
        self.last_recommendations: Optional[Dict[str, Any]] = None
        self.last_briefing: Optional[Dict[str, Any]] = None

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            convert_system_message_to_human=True,
        )

        self._recommendation_chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are a senior inventory planner. Review inventory health data, "
                            "demand forecasts, and lead times. Recommend reorder actions that "
                            "balance stockouts and overstock risk. Respond in strict JSON format."
                        ),
                    ),
                    (
                        "human",
                        (
                            "Inventory snapshot:\n"
                            "{inventory_overview}\n\n"
                            "Items needing attention:\n"
                            "{items_of_interest}\n\n"
                            "Demand forecasts (30-day total units):\n"
                            "{forecast_summaries}\n\n"
                            "Return your response as valid JSON with this schema:\n"
                            "{{\n"
                            '  "recommendations": [\n'
                            "    {{\n"
                            '      "sku": string,\n'
                            '      "product_name": string,\n'
                            '      "priority": "HIGH" | "MEDIUM" | "LOW",\n'
                            '      "recommended_order_qty": integer,\n'
                            '      "estimated_cost": float,\n'
                            '      "reason": string\n'
                            "    }}\n"
                            "  ],\n"
                            '  "overall_notes": string\n'
                            "}}\n"
                            "Only include SKUs that truly need action. Cap recommendations to 12 items."
                        ),
                    ),
                ]
            )
            | self.llm
            | StrOutputParser()
        )

        self._briefing_chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are an AI chief operations strategist. Craft an executive-ready inventory briefing "
                            "grounded in the supplied facts. Always respond with well-structured JSON that can be parsed. "
                            "If a data point is missing, acknowledge it instead of inventing numbers."
                        ),
                    ),
                    (
                        "human",
                        (
                            "Context data (JSON):\n"
                            "{context}\n\n"
                            "Return JSON with exactly these keys:\n"
                            "{{\n"
                            '  "executive_summary": string (markdown with 2-3 bullet points),\n'
                            '  "priority_actions": [\n'
                            "    {{\n"
                            '      "sku": string,\n'
                            '      "headline": string,\n'
                            '      "action": string,\n'
                            '      "impact": string,\n'
                            '      "confidence": "High" | "Medium" | "Low"\n'
                            "    }}\n"
                            "  ],\n"
                            '  "finance_callouts": string (<=80 words),\n'
                            '  "risk_watchlist": [\n'
                            "    {{\n"
                            '      "sku": string,\n'
                            '      "risk": string,\n'
                            '      "trigger": string\n'
                            "    }}\n"
                            "  ],\n"
                            '  "team_broadcast": {{\n'
                            '    "channel": "Slack",\n'
                            '    "message": string (<=80 words)\n'
                            "  }}\n"
                            "}}\n"
                            "Ensure each array includes at least one item. Base everything on the context provided."
                        ),
                    ),
                ]
            )
            | self.llm
            | StrOutputParser()
        )

        self._qa_chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are an inventory strategist who answers questions strictly using the provided context. "
                            "Flag any uncertainties, suggest next data to pull when needed, and keep responses actionable."
                        ),
                    ),
                    (
                        "human",
                        (
                            "Context (JSON):\n{context}\n\n"
                            "Recent dialogue:\n{conversation}\n\n"
                            "Question:\n{question}\n\n"
                            "Respond in markdown with clear, concise guidance."
                        ),
                    ),
                ]
            )
            | self.llm
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # Data loading & validation
    # ------------------------------------------------------------------
    def load_inventory(self) -> Dict[str, Any]:
        """Load inventory CSV into a DataFrame and compute quick stats."""
        df = pd.read_csv(self.data_path)

        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"Inventory file missing columns: {', '.join(sorted(missing))}")

        numeric_cols = [
            "current_stock",
            "reorder_point",
            "lead_time_days",
            "cost_per_unit",
            "last_30_days_sales",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[numeric_cols].isnull().any().any():
            raise ValueError("Inventory data contains invalid numeric values (NaN).")

        self.df = df.copy()
        df["inventory_value"] = df["current_stock"] * df["cost_per_unit"]

        summary = {
            "total_skus": int(len(df)),
            "total_inventory_value": float(df["inventory_value"].sum()),
            "average_lead_time": float(df["lead_time_days"].mean()),
            "categories": df["category"].value_counts().to_dict(),
        }

        return summary

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def analyze_inventory(self) -> Dict[str, Any]:
        """Identify low stock, overstock, and fast movers."""
        if self.df is None:
            raise RuntimeError("Inventory data not loaded. Call load_inventory() first.")

        df = self.df.copy()
        df["inventory_value"] = df["current_stock"] * df["cost_per_unit"]
        df["stock_gap"] = df["reorder_point"] - df["current_stock"]
        df["turnover_ratio"] = np.where(
            df["current_stock"] > 0,
            df["last_30_days_sales"] / df["current_stock"],
            np.nan,
        )

        low_stock = df[df["current_stock"] < df["reorder_point"]].copy()
        low_stock["urgency_score"] = (
            (low_stock["reorder_point"] - low_stock["current_stock"]).clip(lower=0)
            + low_stock["last_30_days_sales"] * 0.5
            + low_stock["lead_time_days"] * 0.3
        )
        low_stock = low_stock.sort_values("urgency_score", ascending=False)

        overstock = df[
            (df["current_stock"] > df["reorder_point"] * 1.6)
            & (df["last_30_days_sales"] < 10)
        ].copy()
        overstock["excess_units"] = overstock["current_stock"] - overstock["reorder_point"]

        fast_movers = df[df["last_30_days_sales"] > 50].copy()
        fast_movers["days_of_cover"] = np.where(
            fast_movers["last_30_days_sales"] > 0,
            (fast_movers["current_stock"] / fast_movers["last_30_days_sales"]) * 30,
            np.nan,
        )

        summary_metrics = {
            "low_stock_count": int(len(low_stock)),
            "overstock_count": int(len(overstock)),
            "fast_mover_count": int(len(fast_movers)),
            "stockout_risk_pct": round((len(low_stock) / len(df)) * 100, 1),
            "inventory_turnover": round(float(df["turnover_ratio"].dropna().mean()), 2)
            if df["turnover_ratio"].notna().any()
            else 0.0,
            "total_inventory_value": float(df["inventory_value"].sum()),
            "average_lead_time": float(df["lead_time_days"].mean()),
        }

        self.analysis_result = {
            "low_stock": low_stock.to_dict("records"),
            "overstock": overstock.to_dict("records"),
            "fast_movers": fast_movers.to_dict("records"),
            "summary": summary_metrics,
        }
        return self.analysis_result

    # ------------------------------------------------------------------
    # Forecasting with Prophet
    # ------------------------------------------------------------------
    def forecast_demand(self, horizon_days: int = 30, history_days: int = 60) -> pd.DataFrame:
        """Train lightweight Prophet models per SKU and return 30-day demand forecasts."""
        if self.df is None:
            raise RuntimeError("Inventory data not loaded. Call load_inventory() first.")

        self.forecast_timelines.clear()
        forecasts: List[Dict[str, Any]] = []
        for _, row in self.df.iterrows():
            history = self._build_history(row, history_days)

            try:
                model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.3,
                )
                model.fit(history)
                future = model.make_future_dataframe(periods=horizon_days, freq="D")
                forecast = model.predict(future)
                future_window = forecast.tail(horizon_days)
                total_demand = float(future_window["yhat"].clip(lower=0).sum())
                avg_daily = float(future_window["yhat"].clip(lower=0).mean())
                lower = float(future_window["yhat_lower"].clip(lower=0).sum())
                upper = float(future_window["yhat_upper"].clip(lower=0).sum())
                timeline = future_window[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                timeline["sku"] = row["sku"]
                timeline["product_name"] = row["product_name"]
            except Exception:
                # Fallback to naive forecast using recent sales velocity
                velocity = max(row["last_30_days_sales"] / 30, 0.1)
                total_demand = velocity * horizon_days * 1.05
                avg_daily = velocity * 1.05
                lower = velocity * horizon_days * 0.9
                upper = velocity * horizon_days * 1.3
                future_dates = pd.date_range(
                    start=pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
                    periods=horizon_days,
                    freq="D",
                )
                yhat = np.full(horizon_days, avg_daily)
                timeline = pd.DataFrame(
                    {
                        "ds": future_dates,
                        "yhat": yhat,
                        "yhat_lower": yhat * 0.8,
                        "yhat_upper": yhat * 1.2,
                        "sku": row["sku"],
                        "product_name": row["product_name"],
                    }
                )

            self.forecast_timelines[row["sku"]] = timeline

            forecasts.append(
                {
                    "sku": row["sku"],
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "forecast_units": round(total_demand, 1),
                    "avg_daily_demand": round(avg_daily, 2),
                    "forecast_lower": round(lower, 1),
                    "forecast_upper": round(upper, 1),
                    "current_stock": int(row["current_stock"]),
                    "reorder_point": int(row["reorder_point"]),
                    "lead_time_days": int(row["lead_time_days"]),
                    "cost_per_unit": float(row["cost_per_unit"]),
                    "last_30_days_sales": int(row["last_30_days_sales"]),
                }
            )

        forecast_df = pd.DataFrame(forecasts)
        forecast_df["coverage_days"] = np.where(
            forecast_df["avg_daily_demand"] > 0,
            (forecast_df["current_stock"] / forecast_df["avg_daily_demand"]),
            np.nan,
        )
        forecast_df["coverage_days"] = forecast_df["coverage_days"].replace([np.inf, -np.inf], np.nan)
        forecast_df["projected_gap"] = (
            forecast_df["forecast_units"] + forecast_df["reorder_point"] - forecast_df["current_stock"]
        )

        self.forecast_result = forecast_df
        return forecast_df

    def _build_history(self, row: pd.Series, history_days: int) -> pd.DataFrame:
        """Generate synthetic daily demand history for Prophet."""
        seed = int(hashlib.md5(row["sku"].encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        avg_daily_sales = max(row["last_30_days_sales"] / 30, 0.4)

        if row["last_30_days_sales"] > row["reorder_point"] * 1.2:
            growth_factor = 1.012
        elif row["last_30_days_sales"] < row["reorder_point"] * 0.75:
            growth_factor = 0.988
        else:
            growth_factor = 1.0

        values = []
        base = avg_daily_sales
        for i in range(history_days):
            trend_multiplier = growth_factor ** (i - history_days / 2)
            lam = max(0.15, base * trend_multiplier)
            noise = rng.normal(0, lam * 0.25)
            value = max(0.0, lam + noise)
            values.append(round(value, 3))

        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=history_days, freq="D")
        return pd.DataFrame({"ds": dates, "y": values})

    # ------------------------------------------------------------------
    # Gemini-powered recommendations
    # ------------------------------------------------------------------
    def generate_recommendations(
        self,
        max_items: int = 12,
    ) -> Dict[str, Any]:
        """Call Gemini (via LangChain) to produce reorder guidance."""
        if self.df is None or self.analysis_result is None or self.forecast_result is None:
            raise RuntimeError(
                "Missing inputs. Call load_inventory(), analyze_inventory(), and forecast_demand() first."
            )

        focus_df = self._prepare_focus_dataframe(max_items)

        overview = self._format_inventory_overview()
        items_text = focus_df.to_json(orient="records")
        forecast_slice = (
            self.forecast_result[self.forecast_result["sku"].isin(focus_df["sku"])]
            .loc[:, ["sku", "product_name", "forecast_units", "avg_daily_demand", "coverage_days"]]
        )
        forecast_text = forecast_slice.to_json(orient="records")

        prompt_preview = (
            f"Inventory overview: {overview}\nItems: {items_text}\nForecasts: {forecast_text}"
        )

        llm_response = self._recommendation_chain.invoke(
            {
                "inventory_overview": overview,
                "items_of_interest": items_text,
                "forecast_summaries": forecast_text,
            }
        )

        parsed = self._parse_recommendations(llm_response)

        self.usage.prompt_tokens += estimate_tokens(prompt_preview)
        self.usage.completion_tokens += estimate_tokens(llm_response)

        result = {
            "recommendations": parsed.get("recommendations", []),
            "overall_notes": parsed.get("overall_notes", ""),
            "raw_response": llm_response,
            "token_usage": self.usage.to_dict(),
            "prompt_preview": prompt_preview,
        }
        self.last_recommendations = result
        return result

    def _build_context_snapshot(
        self,
        recommendations: Optional[Dict[str, Any]],
        watchlist_items: int,
    ) -> Dict[str, Any]:
        """Assemble structured data for downstream GenAI prompts."""
        rec_payload = recommendations or self.last_recommendations or {}
        rec_list = rec_payload.get("recommendations", []) or []
        top_recommendations = rec_list[: min(5, len(rec_list))]

        focus_df = self._prepare_focus_dataframe(max(3, watchlist_items))
        watchlist_records: List[Dict[str, Any]] = []
        if not focus_df.empty:
            watchlist_records = json.loads(focus_df.to_json(orient="records"))

        df = self.df.copy()
        df["inventory_value"] = df["current_stock"] * df["cost_per_unit"]
        category_series = (
            df.groupby("category")["inventory_value"]
            .sum()
            .sort_values(ascending=False)
            .head(3)
        )
        category_snapshot = [
            {"category": str(cat), "inventory_value": round(float(value), 2)}
            for cat, value in category_series.items()
        ]

        low_stock_sample: List[Dict[str, Any]] = []
        if self.analysis_result:
            low_stock_raw = self.analysis_result.get("low_stock", [])[:4]
            if low_stock_raw:
                low_stock_df = pd.DataFrame(low_stock_raw)
                low_stock_sample = json.loads(
                    low_stock_df.where(pd.notnull(low_stock_df), None).to_json(orient="records")
                )

        fast_movers_sample: List[Dict[str, Any]] = []
        if self.analysis_result:
            fast_movers_raw = self.analysis_result.get("fast_movers", [])[:3]
            if fast_movers_raw:
                fast_df = pd.DataFrame(fast_movers_raw)
                fast_df = fast_df.replace([np.inf, -np.inf], np.nan)
                fast_movers_sample = json.loads(
                    fast_df.where(pd.notnull(fast_df), None).to_json(orient="records")
                )

        forecast_snapshot: List[Dict[str, Any]] = []
        if self.forecast_result is not None:
            forecast_slice = (
                self.forecast_result.replace([np.inf, -np.inf], np.nan)
                .sort_values("coverage_days", na_position="last")
                .head(5)
            )
            if not forecast_slice.empty:
                forecast_subset = forecast_slice[
                    [
                        "sku",
                        "product_name",
                        "avg_daily_demand",
                        "coverage_days",
                        "forecast_units",
                        "projected_gap",
                    ]
                ]
                forecast_snapshot = json.loads(
                    forecast_subset.where(pd.notnull(forecast_subset), None).to_json(orient="records")
                )

        summary_snapshot = self.analysis_result["summary"] if self.analysis_result else {}

        return {
            "summary": summary_snapshot,
            "overall_notes": rec_payload.get("overall_notes", ""),
            "top_recommendations": top_recommendations,
            "watchlist": watchlist_records,
            "top_categories_by_value": category_snapshot,
            "low_stock_sample": low_stock_sample,
            "fast_movers_sample": fast_movers_sample,
            "forecast_snapshot": forecast_snapshot,
        }

    def generate_briefing(
        self,
        recommendations: Optional[Dict[str, Any]] = None,
        watchlist_items: int = 6,
    ) -> Dict[str, Any]:
        """Produce an executive-ready GenAI briefing that summarizes the run."""
        if self.df is None or self.analysis_result is None or self.forecast_result is None:
            raise RuntimeError(
                "Missing inputs. Call load_inventory(), analyze_inventory(), and forecast_demand() first."
            )

        context_payload = self._build_context_snapshot(recommendations, watchlist_items)
        prompt_context = json.dumps(context_payload)

        llm_response = self._briefing_chain.invoke({"context": prompt_context})

        fallback = {
            "executive_summary": "Unable to generate briefing. Review underlying data manually.",
            "priority_actions": [],
            "finance_callouts": "N/A",
            "risk_watchlist": [],
            "team_broadcast": {"channel": "Slack", "message": "No update available."},
        }
        parsed = self._extract_json_object(llm_response) or fallback

        self.usage.prompt_tokens += estimate_tokens(prompt_context)
        self.usage.completion_tokens += estimate_tokens(llm_response)

        result = {
            "structured_brief": parsed,
            "raw_response": llm_response,
            "prompt_preview": prompt_context,
            "token_usage": self.usage.to_dict(),
        }
        self.last_briefing = result
        return result

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        recommendations: Optional[Dict[str, Any]] = None,
        watchlist_items: int = 6,
    ) -> Dict[str, Any]:
        """Provide conversational Q&A grounded in the latest inventory context."""
        if self.df is None or self.analysis_result is None or self.forecast_result is None:
            raise RuntimeError(
                "Missing inputs. Call load_inventory(), analyze_inventory(), and forecast_demand() first."
            )
        if not question.strip():
            raise ValueError("Question must be a non-empty string.")

        context_payload = self._build_context_snapshot(recommendations, watchlist_items)
        context_json = json.dumps(context_payload)
        conversation_text = ""
        if chat_history:
            recent_turns = chat_history[-4:]
            conversation_text = "\n".join(
                f"{turn.get('role', 'user').capitalize()}: {turn.get('content', '')}"
                for turn in recent_turns
                if turn.get("content")
            )

        prompt_preview = (
            f"Context: {context_json}\n"
            f"Conversation: {conversation_text or 'None'}\n"
            f"Question: {question}"
        )
        llm_response = self._qa_chain.invoke(
            {
                "context": context_json,
                "conversation": conversation_text or "None",
                "question": question,
            }
        )

        self.usage.prompt_tokens += estimate_tokens(prompt_preview)
        self.usage.completion_tokens += estimate_tokens(llm_response)

        return {
            "answer": llm_response,
            "raw_response": llm_response,
            "prompt_preview": prompt_preview,
            "token_usage": self.usage.to_dict(),
        }

    def _prepare_focus_dataframe(self, max_items: int) -> pd.DataFrame:
        """Combine risk signals to surface SKUs for the LLM."""
        df = self.df.copy()
        df = df.merge(
            self.forecast_result[["sku", "forecast_units", "coverage_days", "projected_gap"]],
            on="sku",
            how="left",
        )

        df["risk_score"] = 0.0
        df.loc[df["current_stock"] < df["reorder_point"], "risk_score"] += 3
        df.loc[df["coverage_days"] < df["lead_time_days"], "risk_score"] += 2
        df.loc[df["last_30_days_sales"] > 50, "risk_score"] += 1.5
        df.loc[df["current_stock"] > df["reorder_point"] * 1.6, "risk_score"] += 1.2
        df.loc[df["projected_gap"] > 0, "risk_score"] += 1

        focus_df = (
            df[df["risk_score"] > 0]
            .sort_values(["risk_score", "projected_gap"], ascending=False)
            .head(max_items)
            .copy()
        )

        focus_df["current_stock"] = focus_df["current_stock"].astype(int)
        focus_df["reorder_point"] = focus_df["reorder_point"].astype(int)
        focus_df["lead_time_days"] = focus_df["lead_time_days"].astype(int)
        focus_df["projected_gap"] = focus_df["projected_gap"].round(1)
        focus_df["forecast_units"] = focus_df["forecast_units"].round(1)
        focus_df["coverage_days"] = focus_df["coverage_days"].round(1)

        return focus_df

    def _format_inventory_overview(self) -> str:
        """Build terse overview string for the prompt."""
        summary = self.analysis_result.get("summary", {}) if self.analysis_result else {}
        total_value = summary.get("total_inventory_value") or float(
            (self.df["current_stock"] * self.df["cost_per_unit"]).sum()
        )
        avg_lead = summary.get("average_lead_time") or float(self.df["lead_time_days"].mean())
        return json.dumps(
            {
                "total_skus": int(len(self.df)),
                "inventory_value": round(total_value, 2),
                "avg_lead_time": round(avg_lead, 1),
                "low_stock": summary.get("low_stock_count"),
                "overstock": summary.get("overstock_count"),
                "fast_movers": summary.get("fast_mover_count"),
            }
        )

    def _parse_recommendations(self, response: str) -> Dict[str, Any]:
        """Extract JSON payload from Gemini response."""
        payload = self._extract_json_object(response)
        if payload is not None:
            return payload

        return {
            "recommendations": [],
            "overall_notes": "Unable to parse Gemini response. Please review raw output.",
        }

    @staticmethod
    def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of JSON from an LLM response string."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            json_start = raw.find("{")
            json_end = raw.rfind("}")
            if json_start != -1 and json_end != -1:
                snippet = raw[json_start : json_end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
        return None
