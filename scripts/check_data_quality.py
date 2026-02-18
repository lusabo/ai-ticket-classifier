from ticket_ai.data.quality import DataQualityChecker
from ticket_ai.data.loader import TicketDataLoader
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


EXPECTED_CATEGORIES = [
    "financeiro",
    "assinatura",
    "informacoes",
    "suporte",
    "feedback",
    "logistica",
]


def main():
    print("🔎 Rodando quality check (dataset de tickets)...")
    print("=" * 60)

    loader = TicketDataLoader(db_path="data/tickets.db")

    # 1) RAW: gate antes de transformar
    df_raw = loader.load_raw_data()
    checker = DataQualityChecker(
        df_raw,
        expected_categories=EXPECTED_CATEGORIES,
        min_text_len=10,
    )
    checker.print_report()

    results = checker.run_all_checks()
    if not results["is_valid"]:
        raise SystemExit(
            "❌ Quality gate falhou. Corrija os issues antes de continuar.")

    # 2) Preparação (limpeza/filtros) para seguir
    loader.prepare_training_data(df_raw, min_samples_per_category=1)
    print("✅ Quality gate passou. Dataset pronto para seguir.")


if __name__ == "__main__":
    main()
