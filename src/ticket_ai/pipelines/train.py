import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    """Cria um pipeline (não treinado) para classificação de tickets."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                    lowercase=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b\w\w+\b",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    penalty="l2",
                    C=2.0,
                    max_iter=2000,
                    n_jobs=-1,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train(df: pd.DataFrame) -> Pipeline:
    """
    Treina pipeline de classificação de tickets.
    Retorna pipeline treinado.
    """
    X = df["texto"].fillna("").astype(str)
    y = df["categoria"].fillna("").astype(str).str.strip().str.lower()

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline