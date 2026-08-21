"""Phase 6.2 & 6.3: Narrative generators and LLM validation.

Provides template-based narrative generation, LLM refinement wrapper,
and factual validation rules to prevent hallucinations of customer data.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

import structlog

from ledger.explain.models import Evidence

logger = structlog.get_logger(__name__)


def generate_deterministic_narrative(evidence: Evidence) -> str:
    """System of record template narrative, citing exact facts in the evidence."""
    parts = []
    parts.append(
        f"Alert {evidence.alert_id} generated for account {evidence.account_id} "
        f"with confidence score {evidence.confidence_score:.4f}."
    )

    if evidence.time_window:
        parts.append(
            f"The suspicious activity occurred within the window {evidence.time_window[0]} "
            f"to {evidence.time_window[1]}."
        )

    if evidence.contributing_accounts:
        parts.append(f"Contributing accounts involved: {', '.join(evidence.contributing_accounts)}.")

    if evidence.contributing_transactions:
        parts.append(
            f"Contributing transactions involved: {', '.join(evidence.contributing_transactions)}."
        )

    if evidence.motif_type_detected:
        parts.append(
            f"Graph analysis detected a money laundering {evidence.motif_type_detected} motif "
            f"connecting these entities."
        )

    if evidence.feature_attributions:
        sorted_feats = sorted(
            evidence.feature_attributions.items(), key=lambda x: x[1], reverse=True
        )
        feat_str = ", ".join(f"{k} ({v:.2%})" for k, v in sorted_feats)
        parts.append(f"The primary model feature attributions driving this alert are: {feat_str}.")

    return " ".join(parts)


def validate_narrative(narrative: str, evidence: Evidence) -> bool:
    """Factually validate an LLM narrative against the evidence object.

    Rejects the narrative if it contains account IDs, transaction IDs,
    or dates not present in the structured evidence.
    """
    allowed_ids = set(evidence.contributing_accounts) | set(evidence.contributing_transactions)
    allowed_ids.add(evidence.account_id)
    allowed_ids.add(evidence.alert_id)
    if evidence.motif_type_detected:
        allowed_ids.add(evidence.motif_type_detected)

    # 1. Validate identifiers (Account patterns like 1_A1 or A1, transaction patterns)
    words = re.findall(r"\b[a-zA-Z0-9_]+_[a-zA-Z0-9_]+\b|\b[A-Z][0-9]+\b", narrative)
    for word in words:
        if word not in allowed_ids:
            # If it's a code with an underscore or an account letter-digit pattern, reject
            if "_" in word or re.match(r"^[A-Z][0-9]+$", word):
                logger.warning(
                    "Factual validation failed: found unauthorized identifier in narrative",
                    identifier=word,
                )
                return False

    # 2. Validate dates
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", narrative)
    if dates and evidence.time_window:
        allowed_dates = {d[:10] for d in evidence.time_window if d}
        for d in dates:
            if d not in allowed_dates:
                logger.warning(
                    "Factual validation failed: found unauthorized date in narrative", date=d
                )
                return False

    return True


class LLMNarrativeProvider:
    """Base class for LLM-refined narrative generators."""

    def generate(self, evidence: Evidence) -> str:
        raise NotImplementedError


class NullLLMNarrativeProvider(LLMNarrativeProvider):
    """Fallback provider returning the deterministic narrative directly."""

    def generate(self, evidence: Evidence) -> str:
        return generate_deterministic_narrative(evidence)


class GeminiLLMNarrativeProvider(LLMNarrativeProvider):
    """Gemini API provider with standard library HTTP implementation."""

    def __init__(
        self, api_key: str, model_name: str, temperature: float, prompt_template: str
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_template = prompt_template

    def generate(self, evidence: Evidence) -> str:
        evidence_json = evidence.to_json(indent=2)
        prompt = self.prompt_template.format(evidence_json=evidence_json)

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}"
            f":generateContent?key={self.api_key}"
        )
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }

        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), headers=headers
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                refined = text.strip()

                # Factually validate before returning
                if validate_narrative(refined, evidence):
                    return refined
                else:
                    logger.warning(
                        "LLM narrative failed validation, falling back to deterministic"
                    )
                    return generate_deterministic_narrative(evidence)
        except Exception as exc:
            logger.error("Gemini API call failed, falling back to deterministic", error=str(exc))
            return generate_deterministic_narrative(evidence)


def get_narrative(evidence: Evidence, config: Any) -> str:
    """Generate narrative using configured provider and validate it."""
    if not getattr(config, "llm_enabled", False) or config.provider == "null":
        return generate_deterministic_narrative(evidence)

    # Resolve API Key
    api_key = (
        config.api_key
        if hasattr(config, "api_key") and config.api_key
        else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    )

    if not api_key:
        logger.warning("LLM enabled but no API key found, falling back to deterministic")
        return generate_deterministic_narrative(evidence)

    if config.provider == "gemini":
        provider = GeminiLLMNarrativeProvider(
            api_key=api_key,
            model_name=config.model_name,
            temperature=config.temperature,
            prompt_template=config.prompt_template,
        )
        return provider.generate(evidence)

    return generate_deterministic_narrative(evidence)
