"""
src/preprocess.py

Custom Scikit-learn transformers for the Phishing Email Classifier.

Exported classes:
    TextCleanerTransformer          – Strips HTML, normalises text, combines subject + body.
    HeuristicsExtractorTransformer  – Validates and returns numeric heuristic features.

Design constraints (enforced):
    ✗  No model training or data splitting (belongs in train.py).
    ✓  Both transformers implement the full Scikit-learn estimator API
       (BaseEstimator + TransformerMixin) → compatible with Pipeline, GridSearchCV,
       cross_val_score, clone(), and joblib serialisation.
    ✓  Stateless fit() → safe to call multiple times with no side-effects.
    ✓  Defensive validation → ready for future NoSQL / relational DB payloads
       where column presence or dtypes cannot be guaranteed.
"""

import logging
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────
_DEFAULT_SUBJECT_COL: str       = "subject"
_DEFAULT_BODY_COL: str          = "body"
_DEFAULT_HEURISTIC_COLS: List[str] = ["url_count", "has_attachments", "spf_fail"]

# Pre-compiled regex patterns (compiled once at import time for performance)
_RE_MULTI_SPACE: re.Pattern = re.compile(r"\s+")
_RE_NON_ALPHA:   re.Pattern = re.compile(r"[^a-z0-9\s]")


# ── TextCleanerTransformer ────────────────────────────────────────────────────
class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Strip HTML from email fields, normalise text, and concatenate subject + body.

    This transformer is *stateless*; fit() is a no-op and returns self.
    It is designed to feed directly into TfidfVectorizer inside a Pipeline.

    Parameters
    ----------
    subject_col : str, default "subject"
        DataFrame column that holds the email subject line.
    body_col : str, default "body"
        DataFrame column that holds the email body (may contain raw HTML).
    html_parser : str, default "lxml"
        BeautifulSoup parser backend.
        "lxml" is fast (C extension) and handles malformed markup gracefully.
        Falls back to "html.parser" (stdlib) if lxml is unavailable.

    Input
    -----
    X : pd.DataFrame
        Must contain at least [subject_col, body_col] columns.
        Extra columns are silently ignored.

    Output
    ------
    np.ndarray of shape (n_samples,), dtype=object
        One cleaned, combined text string per row.
        Ready to be consumed by sklearn.feature_extraction.text.TfidfVectorizer.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "subject": ["URGENT: Verify now"],
    ...     "body":    ["<p>Click <a href='http://evil.com'>here</a></p>"],
    ... })
    >>> cleaner = TextCleanerTransformer()
    >>> cleaner.fit_transform(df)
    array(['urgent verify now click here'], dtype=object)
    """

    def __init__(
        self,
        subject_col: str = _DEFAULT_SUBJECT_COL,
        body_col:    str = _DEFAULT_BODY_COL,
        html_parser: str = "lxml",
    ) -> None:
        self.subject_col = subject_col
        self.body_col    = body_col
        self.html_parser = html_parser

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "TextCleanerTransformer":
        """
        No-op fit method required by the Scikit-learn API.

        Parameters
        ----------
        X : pd.DataFrame  (unused)
        y : ignored

        Returns
        -------
        self
        """
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Clean and combine subject + body columns.

        Per-sample pipeline:
        1. Cast both columns to str  → guards against NaN / non-string DB values.
        2. Strip HTML via BeautifulSoup (separator=" " preserves word boundaries).
        3. Lowercase the extracted text.
        4. Remove all characters outside [a-z 0-9].
        5. Collapse consecutive whitespace into a single space and strip edges.
        6. Concatenate: "<cleaned_subject> <cleaned_body>".

        Parameters
        ----------
        X : pd.DataFrame
            Must contain [subject_col, body_col].

        Returns
        -------
        np.ndarray of shape (n_samples,), dtype=object
        """
        self._validate_input(X, required_cols=[self.subject_col, self.body_col])

        subjects: pd.Series = X[self.subject_col].astype(str)
        bodies:   pd.Series = X[self.body_col].astype(str)

        cleaned: np.ndarray = np.fromiter(
            (
                self._clean_field(subj) + " " + self._clean_field(body)
                for subj, body in zip(subjects, bodies)
            ),
            dtype=object,
            count=len(X),
        )

        logger.debug(
            "TextCleanerTransformer: transformed %d samples.", len(cleaned)
        )
        return cleaned

    # ── Private helpers ───────────────────────────────────────────────────────

    def _clean_field(self, raw: str) -> str:
        """
        Sanitise a single raw text field.

        1. BeautifulSoup strips HTML / XML tags; get_text(separator=" ") avoids
           words from adjacent tags being merged (e.g. "<b>click</b><i>here</i>"
           → "click here" not "clickhere").
        2. Lowercase.
        3. Remove punctuation and special characters.
        4. Collapse whitespace.
        """
        # Step 1 – HTML stripping
        text: str = BeautifulSoup(raw, self.html_parser).get_text(separator=" ")

        # Step 2 – Lowercase
        text = text.lower()

        # Step 3 – Keep only [a-z0-9 ]
        text = _RE_NON_ALPHA.sub(" ", text)

        # Step 4 – Collapse whitespace
        text = _RE_MULTI_SPACE.sub(" ", text).strip()

        return text

    def _validate_input(self, X: pd.DataFrame, required_cols: List[str]) -> None:
        """Raise informative errors for malformed input DataFrames."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"TextCleanerTransformer expects a pd.DataFrame, "
                f"received {type(X).__name__}."
            )
        missing: List[str] = [c for c in required_cols if c not in X.columns]
        if missing:
            raise ValueError(
                f"TextCleanerTransformer: missing required columns {missing}. "
                f"Available columns: {sorted(X.columns.tolist())}"
            )


# ── HeuristicsExtractorTransformer ───────────────────────────────────────────
class HeuristicsExtractorTransformer(BaseEstimator, TransformerMixin):
    """
    Extract and validate numeric heuristic features for downstream scaling.

    This transformer is *stateless*; fit() is a no-op and returns self.
    It is designed to precede StandardScaler (or a passthrough) inside a Pipeline.

    Parameters
    ----------
    feature_cols : list of str or None, default None
        Ordered list of numeric columns to extract.
        When None, defaults to ["url_count", "has_attachments", "spf_fail"].
        Passing an explicit list allows future DB schemas to extend the feature set
        without modifying this class.

    Input
    -----
    X : pd.DataFrame
        Must contain at least the columns in feature_cols.
        Non-numeric values in those columns are coerced to NaN, then filled with 0.

    Output
    ------
    np.ndarray of shape (n_samples, len(feature_cols)), dtype=float64
        Clean numeric matrix ready for StandardScaler or direct use in a model.

    Notes
    -----
    NaN filling with 0 is intentional: for binary flags like has_attachments and
    spf_fail, 0 is the conservative (non-suspicious) default when data is absent.
    For url_count, 0 is semantically correct for missing values in a DB row.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "url_count":       [5, 0],
    ...     "has_attachments": [1, 0],
    ...     "spf_fail":        [1, 0],
    ... })
    >>> extractor = HeuristicsExtractorTransformer()
    >>> extractor.fit_transform(df)
    array([[5., 1., 1.],
           [0., 0., 0.]])
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        # Store exactly what was passed — required for BaseEstimator.get_params()
        # and clone() to work correctly. Do NOT resolve None here.
        self.feature_cols = feature_cols

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "HeuristicsExtractorTransformer":
        """
        No-op fit method required by the Scikit-learn API.

        Parameters
        ----------
        X : pd.DataFrame  (unused)
        y : ignored

        Returns
        -------
        self
        """
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Extract, coerce, and return numeric heuristic columns.

        Steps:
        1. Resolve which columns to use (parameter or module default).
        2. Validate DataFrame input and column presence.
        3. Coerce each column to numeric (non-parseable → NaN).
        4. Fill NaN with 0 (safe default for missing heuristics).
        5. Return as float64 array.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples, n_features), dtype=float64
        """
        cols: List[str] = (
            self.feature_cols
            if self.feature_cols is not None
            else _DEFAULT_HEURISTIC_COLS
        )

        self._validate_input(X, required_cols=cols)

        array: np.ndarray = (
            X[cols]
            .apply(pd.to_numeric, errors="coerce")   # bad strings → NaN
            .fillna(0.0)                              # NaN → 0 (conservative default)
            .to_numpy(dtype=np.float64)
        )

        logger.debug(
            "HeuristicsExtractorTransformer: extracted %d samples, %d features.",
            array.shape[0],
            array.shape[1],
        )
        return array

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate_input(self, X: pd.DataFrame, required_cols: List[str]) -> None:
        """Raise informative errors for malformed input DataFrames."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"HeuristicsExtractorTransformer expects a pd.DataFrame, "
                f"received {type(X).__name__}."
            )
        missing: List[str] = [c for c in required_cols if c not in X.columns]
        if missing:
            raise ValueError(
                f"HeuristicsExtractorTransformer: missing required columns {missing}. "
                f"Available columns: {sorted(X.columns.tolist())}"
            )
