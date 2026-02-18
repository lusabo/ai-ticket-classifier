from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from joblib import dump
from ticket_ai.data.loader import TicketDataLoader
from ticket_ai.data.quality import DataQualityChecker
from ticket_ai.pipelines.train_with_mlflow import train_with_tracking


EXPECTED_CATEGORIES = [
    "financeiro",
    "assinatura",
    "informacoes",
    "suporte",
    "feedback",
    "logistica",
]


def prepare_and_train_with_mlflow():
    print("🔄 Iniciando pipeline de treinamento (com MLflow)...")
    print("=" * 60)

    # 1) Carregar RAW (antes de limpeza)
    loader = TicketDataLoader(db_path="data/tickets.db")
    df_raw = loader.load_raw_data()

    # 2) Gate de qualidade no RAW
    checker = DataQualityChecker(
        df_raw,
        expected_categories=EXPECTED_CATEGORIES,
        min_text_len=10,
    )
    checker.print_report()
    results = checker.run_all_checks()
    if not results["is_valid"]:
        raise ValueError("Dataset inválido. Corrija antes de treinar.")

    # 3) Preparar dataset (limpeza/filtros)
    df = loader.prepare_training_data(df_raw, min_samples_per_category=10)

    # 4) Treinar + tracking no MLflow
    model = train_with_tracking(
        df,
        experiment_name="ticket-classification",
        test_size=0.2,
        random_state=42,
        enable_cross_validation=True,
    )

    # 5) Salvar artefatos locais (consumo pela API)
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "ticket_clf.joblib"
    dump(model, model_path)
    print(f"\n💾 Modelo salvo em: {model_path}")

    reference_path = model_dir / "reference_data.parquet"
    df.to_parquet(reference_path, index=False)
    print(f"📋 Dados de referência salvos em: {reference_path}")

    print("=" * 60)
    print("✅ Treinamento concluído com sucesso!")
    print("=" * 60)
    return model

if __name__ == "__main__":
    prepare_and_train_with_mlflow()