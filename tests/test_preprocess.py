import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    HeuristicsExtractorTransformer,
    TextCleanerTransformer,
)


def test_text_cleaner_removes_html_and_normalizes_text():
    frame = pd.DataFrame(
        {
            "subject": ["URGENT: Verify now"],
            "body": ["<p>Click <a href='http://evil.test'>here</a></p>"],
        }
    )

    result = TextCleanerTransformer().fit_transform(frame)

    np.testing.assert_array_equal(
        result,
        ["urgent verify now click here"],
    )


def test_text_cleaner_reports_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        TextCleanerTransformer().fit_transform(pd.DataFrame({"subject": ["hello"]}))


def test_heuristics_extractor_coerces_invalid_values_to_zero():
    frame = pd.DataFrame(
        {
            "url_count": ["5", "not-a-number"],
            "has_attachments": [1, None],
            "spf_fail": ["0", "1"],
        }
    )

    result = HeuristicsExtractorTransformer().fit_transform(frame)

    np.testing.assert_array_equal(result, [[5.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
