import sqlite3
import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/tickets_jan_2026.csv")
DB_PATH = Path("data/tickets.db")

REQUIRED_COLS = [
    "texto",
    "categoria",
    "origem",
    "data_criacao",
    "status",
    "prioridade",
    "cliente_id",
]

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Banco não encontrado em {DB_PATH}.")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV não encontrado em {CSV_PATH}")

    # 1) Lê CSV
    df = pd.read_csv(CSV_PATH, delimiter=";", encoding="utf-8")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {missing}")

    print(f"📥 CSV carregado: {len(df)} linhas")

    # 2) Normalização mínima
    df["texto"] = df["texto"].fillna("").astype(str).str.strip()
    df["categoria"] = df["categoria"].fillna("").astype(str).str.strip().str.lower()

    # remove registros “vazios”
    before_nonempty = len(df)
    df = df[(df["texto"].str.len() > 0) & (df["categoria"].str.len() > 0)].copy()
    removed_empty = before_nonempty - len(df)
    if removed_empty:
        print(f"🧹 Removidos vazios (texto/categoria): {removed_empty}")

    # 3) Deduplicação interna do CSV (texto+categoria)
    before_dedup_csv = len(df)
    df = df.drop_duplicates(subset=["texto", "categoria"]).copy()
    removed_dup_csv = before_dedup_csv - len(df)
    print(f"🧹 Dedup no CSV (texto+categoria): {before_dedup_csv} -> {len(df)} (removidos {removed_dup_csv})")

    # 4) Deduplicação contra o banco (texto+categoria)
    with sqlite3.connect(DB_PATH) as conn:
        existing = pd.read_sql_query(
            "SELECT texto, categoria FROM tickets",
            conn,
        )
        existing["texto"] = existing["texto"].fillna("").astype(str).str.strip()
        existing["categoria"] = existing["categoria"].fillna("").astype(str).str.strip().str.lower()

        merged = df.merge(
            existing,
            on=["texto", "categoria"],
            how="left",
            indicator=True,
        )

        to_insert = merged[merged["_merge"] == "left_only"][REQUIRED_COLS]
        already_in_db = int((merged["_merge"] != "left_only").sum())

        if already_in_db:
            print(f"🧯 Já existiam no DB (texto+categoria): {already_in_db}")

        if to_insert.empty:
            print("ℹ️ Nada novo para inserir (tudo já existia).")
            return

        # 5) Insere o que é novo
        to_insert.to_sql("tickets", conn, if_exists="append", index=False)
        conn.commit()

    print(f"✅ Inseridos {len(to_insert)} novos tickets no banco.")

if __name__ == "__main__":
    main()
