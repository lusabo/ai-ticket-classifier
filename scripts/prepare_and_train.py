from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from joblib import dump
from ticket_ai.data.loader import TicketDataLoader
from ticket_ai.data.quality import DataQualityChecker
from ticket_ai.pipelines.train import train


def prepare_and_train():
    print("🔄 Iniciando pipeline de treinamento...")
    print("=" * 60)

    # 1) Carregar dados RAW (sem limpeza)
    loader = TicketDataLoader(db_path="data/tickets.db")
    df_raw = loader.load_raw_data()

    # 2) Checar qualidade no RAW (antes de qualquer transformação)
    checker = DataQualityChecker(df_raw)
    checker.print_report()
    results = checker.run_all_checks()
    if not results["is_valid"]:
        raise ValueError("Dataset inválido. Corrija antes de treinar.")

    # 3) Preparar dataset (limpeza + filtros) e treinar
    df = loader.prepare_training_data(df_raw, min_samples_per_category=10)
    model = train(df)

    # 4) Salvar modelo
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "ticket_clf.joblib"
    dump(model, model_path)
    print(f"\n💾 Modelo salvo em: {model_path}")

    # 5) Salvar dados de referência (para drift)
    reference_path = model_dir / "reference_data.parquet"
    df.to_parquet(reference_path, index=False)
    print(f"📋 Dados de referência salvos em: {reference_path}")

    print("=" * 60)
    print("✅ Treinamento concluído com sucesso!")
    print("=" * 60)
    return model


if __name__ == "__main__":
    prepare_and_train()