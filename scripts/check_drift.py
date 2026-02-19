from pathlib import Path
import sys
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ticket_ai.data.loader import TicketDataLoader
from ticket_ai.monitoring.drift_monitoring import compare_baseline_vs_current


def main():
    ref_path = Path("models/reference_data.parquet")
    if not ref_path.exists():
        raise SystemExit("❌ reference_data.parquet não encontrado. Rode o treino primeiro.")

    df_ref = pd.read_parquet(ref_path)

    loader = TicketDataLoader(db_path="data/tickets.db")
    date_from = datetime(2026, 1, 1)
    date_to = datetime(2026, 1, 31, 23, 59, 59)

    # RAW: direto do banco (recorte Jan/2026)
    df_cur_raw = loader.load_raw_data(date_from=date_from, date_to=date_to)

    # PREPARED: mesmo contrato do baseline (limpeza/normalização do loader)
    # min_samples_per_category=1 para observar distribuição (não filtrar demais no drift)
    df_cur = loader.prepare_training_data(df_cur_raw, min_samples_per_category=1)

    raw_n = len(df_cur_raw)
    prep_n = len(df_cur)
    retention = (prep_n / raw_n * 100) if raw_n else 0.0

    result = compare_baseline_vs_current(df_ref, df_cur)

    print("\n" + "=" * 60)
    print("📉 DRIFT CHECK (baseline 2025 vs Jan/2026)")
    print("=" * 60)
    print(f"Linhas baseline:   {result['ref_rows']}")
    print(f"Linhas Jan/2026:   {result['cur_rows']}")
    print(f"Avg len baseline:  {result['ref_avg_text_len']:.1f}")
    print(f"Avg len Jan/2026:  {result['cur_avg_text_len']:.1f}")
    print(f"Delta len (%):     {result['avg_text_len_delta_pct']:.1f}%")
    print(f"PSI (categorias):  {result['psi_category']:.4f}")
    print("=" * 60)

    if result["psi_category"] > 0.2:
        print("⚠️  PSI alto: possível drift relevante. Considere investigar/retreinar.")
    else:
        print("✅ PSI baixo: sem sinal forte de drift na distribuição (por enquanto).")


if __name__ == "__main__":
    main()
