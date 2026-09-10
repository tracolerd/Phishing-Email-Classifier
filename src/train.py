"""
src/train.py

Loads mock_emails.csv, builds a full Scikit-learn Pipeline, trains a
RandomForestClassifier tuned for High Precision, prints a classification
report, and exports the pipeline to models/phishing_pipeline.joblib.

Usage:
    python -m src.train
"""

import sys
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocess import TextCleanerTransformer, HeuristicsExtractorTransformer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR: Path     = Path(__file__).resolve().parents[1]
DATA_PATH: Path    = ROOT_DIR / "data" / "mock_emails.csv"
MODEL_DIR: Path    = ROOT_DIR / "models"
MODEL_PATH: Path   = MODEL_DIR / "phishing_pipeline.joblib"

# ── Feature column groups ─────────────────────────────────────────────────────
TEXT_COLS:      list[str] = ["subject", "body"]
HEURISTIC_COLS: list[str] = ["url_count", "has_attachments", "spf_fail"]
TARGET_COL:     str       = "label"

# ── Hyperparameters ───────────────────────────────────────────────────────────
TEST_SIZE:    float = 0.20
RANDOM_STATE: int   = 42

# RandomForest tuned for High Precision (reduce False Positives)
# class_weight balances the loss; high min_samples_leaf avoids noisy splits
RF_PARAMS: dict = {
    "n_estimators":     300,
    "max_depth":        20,
    "min_samples_leaf": 4,
    "class_weight":     "balanced",
    "n_jobs":           -1,
    "random_state":     RANDOM_STATE,
}

TFIDF_PARAMS: dict = {
    "max_features": 5_000,
    "ngram_range":  (1, 2),
    "sublinear_tf": True,       # dampens high-frequency term dominance
}


def build_pipeline() -> Pipeline:
    """
    Construct the end-to-end preprocessing + classifier Pipeline.

    ColumnTransformer branches:
        text_branch       → TextCleaner → TF-IDF
        heuristics_branch → HeuristicsExtractor → StandardScaler

    Returns
    -------
    sklearn.pipeline.Pipeline (unfitted)
    """
    text_branch = Pipeline([
        ("cleaner", TextCleanerTransformer()),
        ("tfidf",   TfidfVectorizer(**TFIDF_PARAMS)),
    ])

    heuristics_branch = Pipeline([
        ("extractor", HeuristicsExtractorTransformer(feature_cols=HEURISTIC_COLS)),
        ("scaler",    StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text",       text_branch,       TEXT_COLS),
            ("heuristics", heuristics_branch, HEURISTIC_COLS),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(**RF_PARAMS)),
    ])

    return pipeline


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and validate the CSV dataset.

    Parameters
    ----------
    path : Path  Path to mock_emails.csv

    Returns
    -------
    X : pd.DataFrame   Feature columns
    y : pd.Series      Target column (label)
    """
    if not path.exists():
        logger.error("Dataset not found at %s. Run data/generate_mock_data.py first.", path)
        sys.exit(1)

    df = pd.read_csv(path)
    required = TEXT_COLS + HEURISTIC_COLS + [TARGET_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[TEXT_COLS + HEURISTIC_COLS]
    y = df[TARGET_COL]

    logger.info("Loaded %d records. Phishing: %d | Legitimate: %d",
                len(df), y.sum(), (y == 0).sum())
    return X, y


def main() -> None:
    """Full training run: load → split → fit → evaluate → export."""

    # 1. Load data
    X, y = load_data(DATA_PATH)

    # 2. Train / test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info("Split → train: %d | test: %d", len(X_train), len(X_test))

    # 3. Build and fit pipeline
    pipeline = build_pipeline()
    logger.info("Training pipeline ...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    # 4. Evaluate
    y_pred = pipeline.predict(X_test)
    print("\n" + "=" * 56)
    print("  Classification Report")
    print("=" * 56)
    print(classification_report(
        y_test, y_pred,
        target_names=["Legitimate (0)", "Phishing (1)"],
        digits=4,
    ))

    # 5. Export full pipeline (preprocessor + model as a single artifact)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Pipeline exported → %s", MODEL_PATH)
    print(f"[+] Model saved: {MODEL_PATH}\n")


if __name__ == "__main__":
    main()
