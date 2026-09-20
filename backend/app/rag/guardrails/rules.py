import re
from dataclasses import dataclass, field

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_RE = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |your |previous |the )?(instructions|prompts|rules)", re.I),
    re.compile(r"reveal (your |the )?(system prompt|instructions)", re.I),
    re.compile(r"you are now\s*(DAN|jailbroken|an unrestricted)", re.I),
    re.compile(r"act as (if )?you (are|were) .*(no|without|bypass).*(rules|restrictions)", re.I),
    re.compile(r"forget (all |your |the )?previous", re.I),
    re.compile(r"bypass (all )?restrictions", re.I),
]
_SECRET_PATTERNS = [
    re.compile(r"password\s+is\s+\S+", re.I),
    re.compile(r"api[_\s]?key\s*[:=]\s*\S+", re.I),
]
_HARMFUL_PATTERNS = [
    re.compile(r"here('s| is) (how|the way) to (hack|steal|attack)", re.I),
]
_PROFANITY: list[str] = []  # intentionally empty; extend via config in production

PII_PATTERNS = {"email": _EMAIL_RE, "phone": _PHONE_RE, "ssn": _SSN_RE, "credit_card": _CARD_RE}
MASK_MAP = {
    "email": "[EMAIL REDACTED]",
    "phone": "[PHONE REDACTED]",
    "ssn": "[SSN REDACTED]",
    "credit_card": "[CARD REDACTED]",
}


@dataclass
class GuardrailResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    cleaned_text: str = ""
    masked: bool = False


def clean_input(text: str) -> str:
    out = re.sub(r"[-]{3,}", "", text)
    out = re.sub(r"[=]{3,}", "", out)
    out = out.replace("{{", "{ {").replace("}}", "} }")
    return out.strip()


def mask_pii(text: str) -> str:
    masked = text
    for name, pattern in PII_PATTERNS.items():
        masked = pattern.sub(MASK_MAP[name], masked)
    return masked


def check_prompt_injection(text: str) -> GuardrailResult:
    reasons = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
    return GuardrailResult(passed=not reasons, reasons=reasons)


def check_pii(text: str) -> GuardrailResult:
    reasons = [name for name, pat in PII_PATTERNS.items() if pat.search(text)]
    return GuardrailResult(passed=not reasons, reasons=reasons)


def check_profanity(text: str) -> GuardrailResult:
    reasons = [w for w in _PROFANITY if w in text.lower()]
    return GuardrailResult(passed=not reasons, reasons=reasons)


def run_guardrails(text: str) -> GuardrailResult:
    cleaned = clean_input(text)
    reasons: list[str] = []
    blocked = False
    for check in (check_prompt_injection, check_profanity):
        res = check(cleaned)
        blocked = blocked or not res.passed
        reasons.extend(res.reasons)
    pii_masked = mask_pii(cleaned)
    masked = pii_masked != cleaned
    return GuardrailResult(
        passed=not blocked,
        reasons=reasons,
        cleaned_text=pii_masked,
        masked=masked,
    )


def validate_output(text: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    masked = mask_pii(text)
    if masked != text:
        warnings.append("PII masked in output")
    for pat in _SECRET_PATTERNS:
        if pat.search(masked):
            masked = pat.sub("[REDACTED]", masked)
            warnings.append("secret pattern masked")
    for pat in _HARMFUL_PATTERNS:
        if pat.search(masked):
            masked = "[Response blocked: potentially harmful content]"
            warnings.append("harmful content blocked")
            break
    return masked, warnings
