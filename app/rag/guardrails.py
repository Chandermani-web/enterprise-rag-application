"""Lightweight, dependency-free guardrails (input, retrieved-context and output).

Deliberately deterministic: no extra LLM call, no added latency, fully unit-testable.
    input   : length limit, prompt-injection / jailbreak detection, secret + card/SSN redaction (+ optional PII)
    context : neutralise instructions hidden inside retrieved documents (indirect prompt injection)
    output  : system-prompt canary leak check, secret scrubbing
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.logging_setup import scrub

CANARY = "RAGCANARY-7f3a91"

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier|your) (instructions|rules|prompts?|guidelines)",
    r"disregard (all |any |the )?(previous|prior|above|your) (instructions|rules|prompts?)",
    r"forget (everything|all|your) (you|instructions|rules)",
    r"you are now (dan|an? unrestricted|in developer mode|jailbroken)",
    r"\bdan mode\b|\bdeveloper mode\b|\bjailbreak(ed)?\b",
    r"(reveal|show|print|repeat|output|leak) (me )?(your|the) (system|hidden|initial|original) (prompt|instructions|message)",
    r"act as (if you (have|had) )?no (restrictions|rules|limits|filters)",
    r"pretend (that )?you (have|are) no (restrictions|rules|guidelines)",
    r"override (your|the) (safety|system|guardrails?)",
]
_INJ = re.compile("|".join(INJECTION_PATTERNS), re.I)

_SECRET_RE = re.compile(
    r"(sk-or-[A-Za-z0-9_\-]{10,}|gsk_[A-Za-z0-9]{10,}|hf_[A-Za-z0-9]{10,}|tvly-[A-Za-z0-9_\-]{10,}|"
    r"AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}(?!\w)")

REFUSAL = (
    "I can't help with that request. I answer questions using the knowledge base and keep my instructions "
    "and configuration private. Feel free to ask me something about the documentation instead."
)


def _luhn(num: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", num)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


@dataclass
class InputVerdict:
    allowed: bool
    text: str
    reason: str = ""
    redactions: int = 0


class Guardrails:
    def __init__(self, enabled: bool = True, max_chars: int = 4000, redact_pii: bool = False):
        self.enabled, self.max_chars, self.redact_pii = enabled, max_chars, redact_pii

    def check_input(self, text: str) -> InputVerdict:
        text = (text or "").replace("\x00", "").strip()
        if not text:
            return InputVerdict(False, text, "empty message")
        if len(text) > self.max_chars:
            return InputVerdict(False, text[:200], f"message longer than {self.max_chars} characters")
        if not self.enabled:
            return InputVerdict(True, text)
        if _INJ.search(text):
            return InputVerdict(False, text, "prompt-injection attempt")
        redactions = 0

        def sub(pattern: re.Pattern, label: str, s: str, check=None) -> str:
            nonlocal redactions

            def repl(m: re.Match) -> str:
                nonlocal redactions
                if check and not check(m.group(0)):
                    return m.group(0)
                redactions += 1
                return f"[{label}]"

            return pattern.sub(repl, s)

        text = sub(_SECRET_RE, "REDACTED_SECRET", text)
        text = sub(_CARD_RE, "REDACTED_CARD", text, _luhn)
        text = sub(_SSN_RE, "REDACTED_ID", text)
        if self.redact_pii:
            text = sub(_EMAIL_RE, "EMAIL", text)
            text = sub(_PHONE_RE, "PHONE", text)
        return InputVerdict(True, text, redactions=redactions)

    def sanitize_context(self, text: str) -> str:
        """Retrieved text is *data*, never instructions: drop lines that try to command the model."""
        if not self.enabled:
            return text
        kept = [("[removed: suspicious instruction]" if _INJ.search(line) else line) for line in text.splitlines()]
        return "\n".join(kept)

    def check_output(self, text: str) -> tuple[str, bool]:
        """Returns (safe_text, blocked)."""
        if not self.enabled:
            return text, False
        if CANARY in text:
            return REFUSAL, True
        return scrub(text) if _SECRET_RE.search(text) else text, False
