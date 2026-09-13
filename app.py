"""
app.py

CLI inference tool for the Phishing Email Classifier.

Usage:
    # JSON input
    python app.py --json '{"subject":"Verify now","body":"Click http://evil.com","url_count":5,"has_attachments":1,"spf_fail":1}'

    # Interactive mode
    python app.py --interactive
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR:   Path = Path(__file__).resolve().parent
MODEL_PATH: Path = ROOT_DIR / "models" / "phishing_pipeline.joblib"

# ── Feature schema ────────────────────────────────────────────────────────────
REQUIRED_FIELDS: list[str] = [
    "subject", "body", "url_count", "has_attachments", "spf_fail"
]

# ── Explanation thresholds (tunable without retraining) ───────────────────────
_EXPLAIN_RULES: list[tuple[str, Any, str]] = [
    # (field, threshold_or_value, human-readable reason)
    ("spf_fail",        1,   "SPF authentication failure detected"),
    ("url_count",       5,   "High URL count detected"),
    ("url_count",       3,   "Elevated URL count detected"),
    ("has_attachments", 1,   "Unexpected attachment present"),
]

_PHISHING_KEYWORDS: list[str] = [
    "urgent", "verify", "suspended", "click here", "confirm",
    "password", "account", "limited", "unusual activity", "prize",
    "winner", "claim", "bank", "paypal", "update billing",
]


def load_pipeline(path: Path) -> Any:
    """
    Load the serialised Scikit-learn pipeline from disk.

    Parameters
    ----------
    path : Path  Location of the .joblib file.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    if not path.exists():
        print(json.dumps({
            "error": f"Model not found at {path}. Run: python -m src.train"
        }, indent=2))
        sys.exit(1)
    return joblib.load(path)


def build_dataframe(payload: dict) -> pd.DataFrame:
    """
    Validate input payload and construct a single-row DataFrame.

    Parameters
    ----------
    payload : dict  Must contain all REQUIRED_FIELDS.

    Returns
    -------
    pd.DataFrame with one row.

    Raises
    ------
    ValueError if required fields are missing.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return pd.DataFrame([{
        "subject":         str(payload["subject"]),
        "body":            str(payload["body"]),
        "url_count":       int(payload["url_count"]),
        "has_attachments": int(payload["has_attachments"]),
        "spf_fail":        int(payload["spf_fail"]),
    }])


def generate_explanations(payload: dict, phishing_proba: float) -> list[str]:
    """
    Produce human-readable reasons for the classification decision.

    Strategy (no external dependencies):
    1. Apply threshold rules on numeric heuristic features.
    2. Scan lowercased subject + body for known phishing keywords.
    3. Return empty list when email is confidently classified as legitimate.

    Parameters
    ----------
    payload        : Raw input dict.
    phishing_proba : Predicted probability of phishing (0.0 – 1.0).

    Returns
    -------
    List of explanation strings (may be empty for clean legitimate emails).
    """
    reasons: list[str] = []

    # Heuristic rule checks
    for field, threshold, message in _EXPLAIN_RULES:
        val = payload.get(field, 0)
        if int(val) >= threshold:
            reasons.append(message)
            break  # one url_count reason is enough

    # Re-apply spf separately (independent of url_count break)
    if int(payload.get("spf_fail", 0)) == 1 and "SPF" not in " ".join(reasons):
        reasons.insert(0, "SPF authentication failure detected")

    if int(payload.get("has_attachments", 0)) == 1:
        reasons.append("Unexpected attachment present")

    # Keyword scan on subject + body
    combined_text = (
        str(payload.get("subject", "")) + " " + str(payload.get("body", ""))
    ).lower()

    found_keywords = [kw for kw in _PHISHING_KEYWORDS if kw in combined_text]
    if found_keywords:
        reasons.append(f"Suspicious keywords detected: {', '.join(found_keywords[:5])}")

    # Probability signal
    if phishing_proba >= 0.80:
        reasons.append(f"Model confidence: {phishing_proba:.0%} phishing probability")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_reasons: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique_reasons.append(r)

    return unique_reasons


def classify(pipeline: Any, payload: dict) -> dict:
    """
    Run the full classification pipeline and return a structured result.

    Parameters
    ----------
    pipeline : Fitted sklearn Pipeline.
    payload  : Raw input dict with email fields.

    Returns
    -------
    dict with keys:
        phishing_probability : float  (rounded to 4 dp)
        classification_label : str   "PHISHING" | "LEGITIMATE"
        confidence           : str   "HIGH" | "MEDIUM" | "LOW"
        explanations         : list[str]
        input_summary        : dict  (echo of numeric features)
    """
    df = build_dataframe(payload)

    proba: float = float(pipeline.predict_proba(df)[0][1])
    label: str   = "PHISHING" if pipeline.predict(df)[0] == 1 else "LEGITIMATE"

    if proba >= 0.80 or proba <= 0.20:
        confidence = "HIGH"
    elif proba >= 0.65 or proba <= 0.35:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    explanations = generate_explanations(payload, proba) if label == "PHISHING" else []

    return {
        "phishing_probability": round(proba, 4),
        "classification_label": label,
        "confidence":           confidence,
        "explanations":         explanations,
        "input_summary": {
            "url_count":       int(payload["url_count"]),
            "has_attachments": int(payload["has_attachments"]),
            "spf_fail":        int(payload["spf_fail"]),
        },
    }


def interactive_mode(pipeline: Any) -> None:
    """Prompt the user to enter email fields one by one."""
    print("\n[Phishing Email Classifier — Interactive Mode]")
    print("Enter email details below (Ctrl+C to quit).\n")

    try:
        payload = {
            "subject":         input("Subject     : ").strip(),
            "body":            input("Body        : ").strip(),
            "url_count":       input("URL count   : ").strip() or "0",
            "has_attachments": input("Attachments (0/1): ").strip() or "0",
            "spf_fail":        input("SPF fail (0/1)   : ").strip() or "0",
        }
        result = classify(pipeline, payload)
        print("\n" + json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("\n[Aborted]")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phishing Email Classifier — CLI inference tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--json", metavar="JSON_STRING",
        help=(
            'JSON string with keys: subject, body, url_count,\n'
            'has_attachments, spf_fail.\n'
            'Example:\n'
            '  --json \'{"subject":"Urgent","body":"Click here",'
            '"url_count":5,"has_attachments":1,"spf_fail":1}\''
        ),
    )
    group.add_argument(
        "--interactive", action="store_true",
        help="Prompt for each field interactively.",
    )

    args = parser.parse_args()
    pipeline = load_pipeline(MODEL_PATH)

    if args.interactive:
        interactive_mode(pipeline)
        return

    try:
        payload = json.loads(args.json)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON: {exc}"}, indent=2))
        sys.exit(1)

    try:
        result = classify(pipeline, payload)
        print(json.dumps(result, indent=2))
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
