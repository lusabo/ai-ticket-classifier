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

    # 2) Normalização mínima (evita variações bobas virarem “novos registros”)
    df["texto"] = df["texto"].fillna("").astype(str).str.strip()
    df["categoria"] = df["categoria"].fillna(
        "").astype(str).str.strip().str.lower()

    # remove registros “vazios”
    df = df[(df["texto"].str.len() > 0) & (
        df["categoria"].str.len() > 0)].copy()

    # 3) Deduplicação por chave simples
    # Observação: aqui usamos (texto, categoria, data_criacao) como “chave natural” didática.
    # Em produção, pode existir um id do ticket/origem, hash do texto + timestamp, etc.
    with sqlite3.connect(DB_PATH) as conn:
        existing = pd.read_sql_query(
            "SELECT texto, categoria, data_criacao FROM tickets",
            conn,
        )

        merged = df.merge(
            existing,
            on=["texto", "categoria", "data_criacao"],
            how="left",
            indicator=True,
        )

        to_insert = merged[merged["_merge"] == "left_only"][REQUIRED_COLS]

        if to_insert.empty:
            print("ℹ️ Nada novo para inserir (tudo já existia).")
            return

        # 4) Insere o que é novo
        to_insert.to_sql("tickets", conn, if_exists="append", index=False)
        conn.commit()

    print(f"✅ Inseridos {len(to_insert)} novos tickets no banco.")


if __name__ == "__main__":
    main()
