"""Loading personas and entitlements from config, with hot reload. Requirement 8.

The CEO wants the tone changed weekly, so tone lives in YAML and is re-read when
the file's mtime changes. No restart, no deploy.

What a persona cannot do is the point. It controls tone, length and layout only.
The PII rules, the SQL whitelist and entitlement scoping are in code, so someone
editing a YAML file to change how reports sound cannot also switch the guardrails
off. "Non-developers can edit the prompt" and "the guardrails are in the prompt"
together mean non-developers can delete the guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..safety.entitlements import Entitlement

CONFIG_DIR = Path("config")


@dataclass(frozen=True)
class Persona:
    """How answers should read. Loaded from config/personas/<name>.yaml."""

    name: str
    label: str
    tone: str
    verbosity: str = "medium"
    format: str = "prose_then_table"
    include_action_items: bool = True

    def style_instructions(self) -> str:
        """The block appended to the report prompt."""
        layout = {
            "prose_then_table": (
                "A short written answer first, then a compact table of the figures."
            ),
            "table_only": "A table only, with a single sentence above it.",
            "bullets": "Bullet points only, no paragraphs.",
        }.get(self.format, "A short written answer, then a table.")

        length = {
            "short": "At most 120 words.",
            "medium": "At most 250 words.",
            "long": "Up to 450 words.",
        }.get(self.verbosity, "At most 250 words.")

        parts = [f"Tone: {self.tone.strip()}", f"Layout: {layout}", f"Length: {length}"]
        if self.include_action_items:
            parts.append("End with 2-3 concrete action items, each tied to a figure above.")
        return "\n".join(parts)


DEFAULT_PERSONA = Persona(
    name="default",
    label="Balanced analyst",
    tone="Direct and factual. Lead with the number, then the reason.",
)


class ConfigStore:
    """Reads personas and entitlements, reloading when a file changes on disk."""

    def __init__(self, directory: str | Path = CONFIG_DIR) -> None:
        self.directory = Path(directory)
        self._cache: dict[Path, tuple[float, object]] = {}

    def _load_yaml(self, path: Path) -> dict:
        """Read a YAML file, using the cached parse unless the mtime moved.

        mtime rather than a timer, so an edit takes effect on the next question
        instead of whenever a TTL happens to expire.
        """
        if not path.exists():
            return {}
        stamp = path.stat().st_mtime
        cached = self._cache.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]  # type: ignore[return-value]
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self._cache[path] = (stamp, data)
        return data

    def personas(self) -> list[str]:
        folder = self.directory / "personas"
        return sorted(p.stem for p in folder.glob("*.yaml")) if folder.exists() else []

    def persona(self, name: str) -> Persona:
        """Load one persona, falling back to the default if it is missing or broken.

        A bad persona file degrades the tone, never the run. The alternative is a
        config typo taking the agent down, which is a poor trade for a file the
        whole point of which is that non-developers edit it.
        """
        data = self._load_yaml(self.directory / "personas" / f"{name}.yaml")
        if not data:
            return DEFAULT_PERSONA
        try:
            return Persona(
                name=data.get("name", name),
                label=data.get("label", name),
                tone=data.get("tone", DEFAULT_PERSONA.tone),
                verbosity=data.get("verbosity", "medium"),
                format=data.get("format", "prose_then_table"),
                include_action_items=bool(data.get("include_action_items", True)),
            )
        except (TypeError, ValueError):
            return DEFAULT_PERSONA

    def global_note(self) -> str:
        """Extra guidance a non-developer can add without touching code."""
        data = self._load_yaml(self.directory / "prompt_overrides.yaml")
        return str(data.get("global_note") or "").strip()

    def users(self) -> dict[str, Entitlement]:
        """Everyone the prototype can act as.

        In production this comes from an SSO token claim. Entitlements must never be
        supplied by the client, which is why this is config and not a CLI flag.
        """
        data = self._load_yaml(self.directory / "entitlements.yaml")
        out: dict[str, Entitlement] = {}
        for user_id, spec in (data.get("users") or {}).items():
            categories = spec.get("categories")
            out[user_id] = Entitlement(
                user_id=user_id,
                display_name=spec.get("display_name", user_id),
                categories=frozenset(categories) if categories else None,
                persona=spec.get("persona", "default"),
            )
        return out

    def user(self, user_id: str) -> Entitlement | None:
        return self.users().get(user_id)
