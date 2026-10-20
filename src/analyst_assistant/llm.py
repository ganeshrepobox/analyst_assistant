"""Gemini wrapper: structured output, rate limiting and budget accounting.

Model ids come from config rather than source. That is partly requirement 8
(changing a model should not be a deploy) and partly because it is necessary:
gemini-2.5-flash is still listed by the API but returns 404 on use, so anything
hardcoded from the brief would already be broken.

Calls are spaced by a rate limiter because the free tier allows about 10 requests
per minute. Learning that by collecting 429s wastes the quota it is protecting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from .data.bigquery import RateLimiter
from .obs.events import Recorder
from .resilience.breaker import CircuitBreaker, call_with_resilience
from .resilience.budget import Budget

T = TypeVar("T", bound=BaseModel)

# Which model handles which job. Overridable from the environment.
DEFAULT_MODELS = {
    "triage": "gemini-3.5-flash-lite",
    "sql": "gemini-3.5-flash-lite",
    "synthesis": "gemini-3.5-flash-lite",
    "embedding": "gemini-embedding-001",
}

# Measured on a free-tier key: flash-lite answers in 11-39s, while
# gemini-3.8-flash took 154s for a one-word reply (170s with thinking lowered, so
# it is queuing rather than reasoning). Larger models are a config change away if
# the key has better quota.
REQUEST_TIMEOUT_MS = 90_000

# Temperature per job. Classification and SQL should be reproducible: two runs of
# the same revenue question earlier produced two different metric definitions, and
# stable SQL also means a recorded fixture is reusable. Prose is left warmer.
TEMPERATURES = {"triage": 0.0, "sql": 0.0, "synthesis": 0.4}

ENV_KEYS = {
    "triage": "MODEL_TRIAGE",
    "sql": "MODEL_SQL",
    "synthesis": "MODEL_SYNTHESIS",
    "embedding": "MODEL_EMBEDDING",
}


class LLMBudgetExceeded(RuntimeError):
    """The turn has used its allowance of model calls."""


@dataclass
class Usage:
    model: str
    tokens_in: int
    tokens_out: int


class Gemini:
    """Thin wrapper over google-genai.

    Deliberately thin. The graph needs three things from a model - a typed object,
    a block of text, and an embedding - so that is the whole surface.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        models: dict[str, str] | None = None,
        limiter: RateLimiter | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set; see docs/SETUP.md")

        self.models = dict(DEFAULT_MODELS)
        for job, env_key in ENV_KEYS.items():
            if os.environ.get(env_key):
                self.models[job] = os.environ[env_key]
        if models:
            self.models.update(models)

        # Per model, because the quota is per model. A single shared gap made a
        # two-call turn wait longer than the models themselves took.
        self._limiter_template = limiter if limiter is not None else RateLimiter()
        self._limiters: dict[str, RateLimiter] = {}
        self.breaker = breaker if breaker is not None else CircuitBreaker("Gemini")
        self._client: Any | None = None

    def _limiter_for(self, model: str) -> RateLimiter:
        if model not in self._limiters:
            self._limiters[model] = RateLimiter(
                min_interval_seconds=self._limiter_template.min_interval_seconds
            )
        return self._limiters[model]

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai

            # Without a timeout a queued request can hang for minutes. Failing at
            # 90s lets the breaker and the degradation path do their job.
            self._client = genai.Client(
                api_key=self.api_key,
                http_options={"timeout": REQUEST_TIMEOUT_MS},
            )
        return self._client

    def _check_budget(self, budget: Budget | None) -> None:
        if budget is not None and not budget.can_call_llm():
            raise LLMBudgetExceeded(budget.exhausted_reason() or "model call budget used up")

    def _record(
        self,
        job: str,
        usage: Usage,
        budget: Budget | None,
        recorder: Recorder | None,
        span: dict[str, Any] | None,
    ) -> None:
        if budget is not None:
            budget.charge_llm(tokens_in=usage.tokens_in, tokens_out=usage.tokens_out)
        if span is not None:
            span["model"] = usage.model
            span["tokens_in"] = usage.tokens_in
            span["tokens_out"] = usage.tokens_out

    @staticmethod
    def _usage_of(response: Any, model: str) -> Usage:
        meta = getattr(response, "usage_metadata", None)
        return Usage(
            model=model,
            tokens_in=getattr(meta, "prompt_token_count", 0) or 0,
            tokens_out=getattr(meta, "candidates_token_count", 0) or 0,
        )

    def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        job: str = "triage",
        budget: Budget | None = None,
        recorder: Recorder | None = None,
        span: dict[str, Any] | None = None,
    ) -> T:
        """Ask for a typed object back.

        Gemini validates against the schema server-side, so a malformed reply is the
        provider's problem rather than ours to parse. Raises if the model returns
        something unusable, which the caller treats like any other node failure.
        """
        self._check_budget(budget)
        model = self.models[job]
        self._limiter_for(model).wait()

        def call() -> Any:
            return self.client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "temperature": TEMPERATURES.get(job, 0.0),
                },
            )

        response = call_with_resilience(call, breaker=self.breaker)
        self._record(job, self._usage_of(response, model), budget, recorder, span)

        parsed = getattr(response, "parsed", None)
        if not isinstance(parsed, schema):
            raise RuntimeError(f"{model} did not return a usable {schema.__name__}")
        return parsed

    def text(
        self,
        prompt: str,
        *,
        job: str = "synthesis",
        budget: Budget | None = None,
        recorder: Recorder | None = None,
        span: dict[str, Any] | None = None,
    ) -> str:
        """Ask for prose back. Used for the written answer."""
        self._check_budget(budget)
        model = self.models[job]
        self._limiter_for(model).wait()

        def call() -> Any:
            return self.client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": TEMPERATURES.get(job, 0.4)},
            )

        response = call_with_resilience(call, breaker=self.breaker)
        self._record(job, self._usage_of(response, model), budget, recorder, span)
        return (getattr(response, "text", "") or "").strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed one or more strings.

        Used at build time for the golden trios and once per question at query time,
        so the corpus is never re-embedded on the hot path.
        """
        if not texts:
            return []
        model = self.models["embedding"]
        self._limiter_for(model).wait()

        def call() -> Any:
            return self.client.models.embed_content(model=model, contents=texts)

        response = call_with_resilience(call, breaker=self.breaker)
        return [list(e.values) for e in response.embeddings]

    def available_models(self) -> list[str]:
        """Models this key can actually call. Used by doctor.

        Worth checking rather than trusting config: a listed model can still 404.
        """
        return sorted(
            m.name.replace("models/", "")
            for m in self.client.models.list()
            if "generateContent" in (getattr(m, "supported_actions", None) or [])
        )
