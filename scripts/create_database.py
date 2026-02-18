import sqlite3
import pandas as pd
from pathlib import Path
import os

CSV_PATH = Path("data/tickets_sinteticos.csv")
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

def criar_banco_a_partir_csv() -> None:
    """Cria (ou recria) o banco SQLite e importa dados do arquivo CSV."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado em {CSV_PATH}")

    os.makedirs("data", exist_ok=True)

    print("📥 Lendo arquivo CSV...")
    df = pd.read_csv(CSV_PATH, delimiter=";", encoding="utf-8")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {missing}")

    print("🗄 (Re)criando tabela tickets...")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS tickets")
        cursor.execute(
            """
            CREATE TABLE tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                texto TEXT NOT NULL,
                categoria TEXT NOT NULL,
                origem TEXT NOT NULL,
                data_criacao TEXT,
                status TEXT,
                prioridade TEXT,
                cliente_id INTEGER
            )
            """
        )

        print("📤 Inserindo dados no banco...")
        df.to_sql("tickets", conn, if_exists="append", index=False)

    print(f"✅ Banco criado com {len(df)} registros.")
    print(f"📍 Localização: {DB_PATH}")


if __name__ == "__main__":
    criar_banco_a_partir_csv()