import random

import numpy as np
import pandas as pd

from data.generate_mock_data import generate_dataset


def test_generate_dataset_is_reproducible_and_balanced():
    random.seed(42)
    np.random.seed(42)
    first = generate_dataset(20, 0.5)

    random.seed(42)
    np.random.seed(42)
    second = generate_dataset(20, 0.5)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [
        "subject",
        "body",
        "has_attachments",
        "url_count",
        "spf_fail",
        "label",
    ]
    assert first["label"].value_counts().to_dict() == {0: 10, 1: 10}
