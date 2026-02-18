import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from ticket_ai.pipelines.train import build_pipeline

def train_with_tracking(
    df: pd.DataFrame,
    experiment_name: str = "ticket-classification",
    test_size: float = 0.2,
    random_state: int = 42,
    enable_cross_validation: bool = True,
) -> object:
    """
    Treina o pipeline e registra parâmetros/métricas/artefatos no MLflow.
    Retorna o modelo treinado (pipeline sklearn).
    """
    if df.empty:
        raise ValueError("DataFrame vazio. Nada para treinar.")

    df = df.copy()
    df["texto"] = df["texto"].fillna("").astype(str)
    df["categoria"] = df["categoria"].fillna(
        "").astype(str).str.strip().str.lower()

    mlflow.set_experiment(experiment_name)

    # Split estratificado: evita que classes fiquem desbalanceadas no holdout
    X_train, X_val, y_train, y_val = train_test_split(
        df["texto"],
        df["categoria"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["categoria"],
    )

    with mlflow.start_run():
        # --
        # Parâmetros principais
        # --
        mlflow.log_param("model_type", "LogisticRegression + TFIDF")
        mlflow.log_param("tfidf_max_features", 50000)
        mlflow.log_param("tfidf_ngram_range", "(1,2)")
        mlflow.log_param("lr_solver", "saga")
        mlflow.log_param("lr_C", 2.0)

        mlflow.log_param("split_test_size", test_size)
        mlflow.log_param("split_random_state", random_state)
        mlflow.log_param("train_size", int(len(X_train)))
        mlflow.log_param("val_size", int(len(X_val)))

        # --
        # Cross-validation (opcional)
        # --
        if enable_cross_validation:
            cv_model = build_pipeline()
            cv_results = cross_validate(
                cv_model,
                X_train,
                y_train,
                cv=5,
                scoring=["accuracy", "f1_weighted"],
                return_train_score=True,
                n_jobs=-1,
            )
            mlflow.log_metric("cv_val_f1_weighted_mean", float(
                cv_results["test_f1_weighted"].mean()))
            mlflow.log_metric(
                "cv_f1_weighted_gap",
                float(cv_results["train_f1_weighted"].mean() -
                      cv_results["test_f1_weighted"].mean()),
            )

        # --
        # Treino final (somente no treino)
        # --
        model = build_pipeline()
        model.fit(X_train, y_train)

        # --
        # Avaliação no holdout
        # --
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        _, _, f1_w, _ = precision_recall_fscore_support(
            y_val, y_pred, average="weighted", zero_division=0
        )
        _, _, f1_m, _ = precision_recall_fscore_support(
            y_val, y_pred, average="macro", zero_division=0
        )

        mlflow.log_metric("val_accuracy", float(accuracy))
        mlflow.log_metric("val_f1_weighted", float(f1_w))
        mlflow.log_metric("val_f1_macro", float(f1_m))

        # --
        # Métricas por classe (como métricas no MLflow)
        # --
        report_dict = classification_report(
            y_val, y_pred, output_dict=True, zero_division=0
        )
        for label, metrics in report_dict.items():
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            safe_label = str(label).strip().lower().replace(" ", "_")
            if isinstance(metrics, dict):
                if "precision" in metrics:
                    mlflow.log_metric(
                        f"val_precision_{safe_label}", float(metrics["precision"]))
                if "recall" in metrics:
                    mlflow.log_metric(
                        f"val_recall_{safe_label}", float(metrics["recall"]))
                if "f1-score" in metrics:
                    mlflow.log_metric(
                        f"val_f1_{safe_label}", float(metrics["f1-score"]))
                if "support" in metrics:
                    mlflow.log_metric(
                        f"val_support_{safe_label}", float(metrics["support"]))

        # --
        # Artefatos de diagnóstico
        # --
        mlflow.log_text(
            classification_report(y_val, y_pred, zero_division=0),
            "classification_report.txt",
        )
        mlflow.log_text(
            str(confusion_matrix(y_val, y_pred)),
            "confusion_matrix.txt",
        )

        # --
        # Signature / contrato do modelo
        # --
        signature = infer_signature(X_train, model.predict(X_train))

        # Log do modelo como artefato MLflow
        mlflow.sklearn.log_model(
            model, artifact_path="model", signature=signature)

    return model
