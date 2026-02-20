import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


class TicketDataLoader:
    """Carrega e prepara dados de tickets a partir de um banco SQLite."""

    def __init__(self, db_path: str = "data/tickets.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Banco de dados não encontrado em '{self.db_path}'. "
                "Execute: uv run python scripts/create_database.py"
            )

    def load_raw_data(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Carrega dados do SQLite SEM limpeza/normalização (útil para gates)."""
        return self._load_from_db(date_from=date_from, date_to=date_to)

    def prepare_training_data(
        self,
        df_raw: pd.DataFrame,
        min_samples_per_category: int = 10,
        return_full: bool = False,
    ) -> pd.DataFrame:
        """Aplica limpeza/normalização + filtros e retorna DF pronto para treino."""
        df = self._clean_data(df_raw)
        df = self._filter_by_category_count(df, min_samples_per_category)
        self._print_summary(df)

        if return_full:
            return df # baseline operacional (com metadados)

        return df[["texto", "categoria"]]

    def load_training_data(
        self,
        min_samples_per_category: int = 10,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Atalho: carrega RAW e prepara para treino."""
        df_raw = self.load_raw_data(date_from=date_from, date_to=date_to)
        return self.prepare_training_data(df_raw, min_samples_per_category=min_samples_per_category)

    def _load_from_db(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Executa query no SQLite com filtros opcionais por data."""

        def _format_dt(value: datetime) -> str:
            if value.tzinfo is not None:
                value = value.astimezone(timezone.utc).replace(tzinfo=None)
            return value.strftime("%Y-%m-%d %H:%M:%S")

        query = """
        SELECT texto, categoria, data_criacao, status, prioridade, cliente_id
        FROM tickets
        WHERE 1=1
        """
        params: list[str] = []

        if date_from:
            query += " AND data_criacao >= ?"
            params.append(_format_dt(date_from))
        if date_to:
            query += " AND data_criacao <= ?"
            params.append(_format_dt(date_to))

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e normaliza os dados."""
        if df.empty:
            return df

        required = {"texto", "categoria"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Colunas obrigatórias ausentes no banco: {missing}")

        df = df.dropna(subset=["texto", "categoria"])
        df["texto"] = df["texto"].astype(str).str.strip()
        df = df[df["texto"].str.len() >= 10]
        df["categoria"] = df["categoria"].astype(str).str.strip().str.lower()

        if "data_criacao" in df.columns:
            df["data_criacao"] = pd.to_datetime(
                df["data_criacao"], errors="coerce")

        df = df.drop_duplicates(subset=["texto", "categoria"])
        return df.reset_index(drop=True)

    def _filter_by_category_count(self, df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
        """Remove categorias com poucos exemplos."""
        if df.empty:
            return df

        counts = df["categoria"].value_counts()
        valid = counts[counts >= min_samples].index
        filtered = df[df["categoria"].isin(valid)].copy()

        removed = len(df) - len(filtered)
        if removed > 0:
            print(
                f"⚠️ Removidos {removed} tickets de categorias com menos de {min_samples} exemplos.")

        return filtered.reset_index(drop=True)

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Resumo rápido do dataset."""
        print("\n📊 Dataset carregado para treinamento")
        print("-" * 50)
        print(f"Total: {len(df)}")
        if df.empty:
            print("⚠️ Dataset vazio após filtros/limpeza.")
            return

        print("\nDistribuição por categoria:")
        print(df["categoria"].value_counts().to_string())

        avg_len = df["texto"].str.len().mean()
        print(f"\nTamanho médio do texto: {avg_len:.1f} caracteres")

        if "data_criacao" in df.columns and df["data_criacao"].notna().any():
            print(
                f"Período: {df['data_criacao'].min().date()} → {df['data_criacao'].max().date()}")

        print("-" * 50)

    def get_category_stats(self) -> pd.DataFrame:
        """Estatísticas por categoria (debug/EDA rápido)."""
        query = """
        SELECT
            categoria,
            COUNT(*) as total,
            AVG(LENGTH(texto)) as tamanho_medio_texto,
            MIN(data_criacao) as primeiro_ticket,
            MAX(data_criacao) as ultimo_ticket
        FROM tickets
        GROUP BY categoria
        ORDER BY total DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            stats = pd.read_sql_query(query, conn)
        return stats
