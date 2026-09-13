import pandas as pd
import pytest

from app import build_dataframe, classify, generate_explanations
from src.train import build_pipeline

VALID_PAYLOAD = {
    "subject": "Urgent verify",
    "body": "Click here to confirm your account",
    "url_count": 5,
    "has_attachments": 1,
    "spf_fail": 1,
}


def test_build_dataframe_requires_all_input_fields():
    with pytest.raises(ValueError, match="Missing required fields"):
        build_dataframe({"subject": "hello"})


def test_generate_explanations_is_empty_for_non_suspicious_input():
    assert generate_explanations(
        {
            "subject": "Team update",
            "body": "The meeting is tomorrow.",
            "url_count": 0,
            "has_attachments": 0,
            "spf_fail": 0,
        },
        0.2,
    ) == []


def test_classify_returns_stable_result_schema():
    pipeline = build_pipeline()
    pipeline.fit(
        pd.DataFrame(
            [
                {
                    "subject": "urgent verify",
                    "body": "click here",
                    "url_count": 5,
                    "has_attachments": 1,
                    "spf_fail": 1,
                },
                {
                    "subject": "team update",
                    "body": "meeting tomorrow",
                    "url_count": 0,
                    "has_attachments": 0,
                    "spf_fail": 0,
                },
            ]
        ),
        [1, 0],
    )

    result = classify(pipeline, VALID_PAYLOAD)

    assert set(result) == {
        "phishing_probability",
        "classification_label",
        "confidence",
        "explanations",
        "input_summary",
    }
    assert 0.0 <= result["phishing_probability"] <= 1.0
    assert result["classification_label"] in {"PHISHING", "LEGITIMATE"}
