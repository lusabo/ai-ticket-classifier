import pandas as pd
from typing import Dict, Optional, Sequence


class DataQualityChecker:
    """Verificações básicas de qualidade para datasets de tickets."""

    def __init__(
        self,
        df: pd.DataFrame,
        expected_categories: Optional[Sequence[str]] = None,
        min_text_len: int = 10,
    ):
        self.df = df.copy()
        self.expected_categories = set(
            expected_categories) if expected_categories else None
        self.min_text_len = min_text_len

    def run_all_checks(self) -> Dict:
        results = {
            "is_valid": True,
            "total_rows": int(len(self.df)),
            "issues": [],
            "warnings": [],
        }

        # 0) Schema mínimo
        required_cols = {"texto", "categoria"}
        missing = required_cols - set(self.df.columns)
        if missing:
            results["issues"].append(
                f"Colunas obrigatórias ausentes: {sorted(missing)}")
            results["is_valid"] = False
            return results

        # 1) Dataset vazio
        if self.df.empty:
            results["issues"].append("DataFrame vazio.")
            results["is_valid"] = False
            return results

        texto = self.df["texto"].fillna("").astype(str)
        categoria = self.df["categoria"].fillna("").astype(str).str.strip()

        # 2) Nulos reais (antes de normalizar)
        null_counts = self.df[["texto", "categoria"]].isnull().sum()
        if null_counts.any():
            results["warnings"].append(
                f"Nulos detectados (antes de normalizar): {null_counts[null_counts > 0].to_dict()}"
            )

        # 3) Categorias vazias (bloqueante)
        empty_cat = (categoria.str.len() == 0).sum()
        if empty_cat > 0:
            results["issues"].append(
                f"{empty_cat} linhas com categoria vazia.")
            results["is_valid"] = False

        # 4) Textos muito curtos (warning)
        short_texts = (texto.str.len() < self.min_text_len).sum()
        if short_texts > 0:
            results["warnings"].append(
                f"{short_texts} textos com menos de {self.min_text_len} caracteres.")

        # 5) Duplicatas exatas (warning)
        dup = pd.DataFrame({"texto": texto.str.strip(
        ), "categoria": categoria.str.lower()}).duplicated().sum()
        if dup > 0:
            results["warnings"].append(
                f"{dup} duplicatas exatas (texto+categoria).")

        # 6) Desbalanceamento alto (warning)
        counts = categoria.str.lower().value_counts()
        if len(counts) > 1 and counts.min() > 0:
            ratio = counts.max() / counts.min()
            if ratio > 10:
                results["warnings"].append(
                    f"Desbalanceamento alto (max/min = {ratio:.1f}x).")

        # 7) Categorias fora do esperado (bloqueante, se informado)
        if self.expected_categories is not None:
            unknown = sorted(set(counts.index) - self.expected_categories)
            if unknown:
                results["issues"].append(
                    f"Categorias fora do conjunto esperado: {unknown}")
                results["is_valid"] = False

        # 8) Ruído por pontuação (warnings)
        punct_count = texto.str.count(r"[^\w\s]")
        avg_punct = float(punct_count.mean())
        pct_high_punct = float((punct_count > 20).mean() * 100)
        if avg_punct > 8:
            results["warnings"].append(
                f"Média alta de caracteres especiais por texto: {avg_punct:.1f}")
        if pct_high_punct > 2:
            results["warnings"].append(
                f"{pct_high_punct:.1f}% dos textos têm >20 caracteres especiais.")

        # 9) Possível vazamento (heurístico)
        leakage_hits = (texto.str.lower().str.contains(
            r"\bfinanceiro\b|\bassinatura\b|\blogistica\b")).mean() * 100
        if leakage_hits > 15:
            results["warnings"].append(
                f"Possível vazamento: {leakage_hits:.1f}% dos textos contêm nomes de categorias (heurístico)."
            )

        return results

    def print_report(self) -> None:
        r = self.run_all_checks()
        print("\n" + "=" * 60)
        print("📋 RELATÓRIO DE QUALIDADE DOS DADOS")
        print("=" * 60)
        print(f"\nTotal de registros: {r['total_rows']}")
        print(
            f"Status: {'✅ APROVADO' if r['is_valid'] else '❌ PROBLEMAS DETECTADOS'}")

        if r["issues"]:
            print("\n❌ Issues (bloqueantes):")
            for i, issue in enumerate(r["issues"], start=1):
                print(f"{i}. {issue}")

        if r["warnings"]:
            print("\n⚠️ Warnings (atenção):")
            for i, w in enumerate(r["warnings"], start=1):
                print(f"{i}. {w}")

        if not r["issues"] and not r["warnings"]:
            print("\n✅ Nenhum problema identificado.")

        print("=" * 60)
