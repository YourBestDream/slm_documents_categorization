"""Document label taxonomy used by the training and inference scripts."""

from __future__ import annotations

from dataclasses import dataclass


RVL_CDIP_LABELS: tuple[str, ...] = (
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
)


LABEL_ALIASES: dict[str, str] = {
    "report": "scientific report",
    "publication": "scientific publication",
    "banking": "budget",
    "bank statement": "budget",
    "statement": "budget",
}


@dataclass(frozen=True)
class LabelSpace:
    labels: tuple[str, ...] = RVL_CDIP_LABELS

    def normalize(self, value: str) -> str:
        normalized = " ".join(value.strip().lower().replace("_", " ").split())
        normalized = LABEL_ALIASES.get(normalized, normalized)
        if normalized not in self.labels:
            raise ValueError(
                f"Unknown label '{value}'. Expected one of: {', '.join(self.labels)}"
            )
        return normalized

    def choices_text(self) -> str:
        return ", ".join(self.labels)

    def closest_from_text(self, value: str) -> str:
        text = value.strip().lower()
        if not text:
            return "unknown"

        for label in self.labels:
            if label == text or label in text:
                return label

        for alias, label in LABEL_ALIASES.items():
            if alias in text:
                return label

        first_line = text.splitlines()[0]
        first_line = first_line.strip(" .,:;`'\"")
        return LABEL_ALIASES.get(first_line, first_line)


DEFAULT_LABEL_SPACE = LabelSpace()

