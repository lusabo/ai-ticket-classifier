import pandas as pd
import numpy as np
from typing import Dict


def _psi(expected: pd.Series, actual: pd.Series, eps: float = 1e-8) -> float:
    """
    PSI (Population Stability Index) simples para comparar distribuições categóricas.
    Retorna um escalar; >0.2 costuma indicar mudança relevante (regra de bolso).
    """
    expected = expected.astype(float)
    actual = actual.astype(float)

    # normaliza para proporções
    expected = expected / (expected.sum() + eps)
    actual = actual / (actual.sum() + eps)

    # alinha índices (categorias presentes em qualquer um dos lados)
    idx = sorted(set(expected.index) | set(actual.index))
    expected = expected.reindex(idx, fill_value=0.0) + eps
    actual = actual.reindex(idx, fill_value=0.0) + eps

    # PSI = Σ (actual - expected) * ln(actual/expected)
    return float(((actual - expected) * np.log(actual / expected)).sum())


def compare_baseline_vs_current(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    category_col: str = "categoria",
    text_col: str = "texto",
) -> Dict:
    """
    Compara baseline vs atual em:
    - tamanho médio do texto
    - distribuição de categorias (com PSI)
    """
    ref = df_ref.copy()
    cur = df_cur.copy()

    # normalização mínima
    ref[text_col] = ref[text_col].fillna("").astype(str)
    cur[text_col] = cur[text_col].fillna("").astype(str)

    ref[category_col] = ref[category_col].fillna(
        "").astype(str).str.strip().str.lower()
    cur[category_col] = cur[category_col].fillna(
        "").astype(str).str.strip().str.lower()

    # métricas simples
    ref_len = ref[text_col].str.len()
    cur_len = cur[text_col].str.len()

    ref_dist = ref[category_col].value_counts()
    cur_dist = cur[category_col].value_counts()

    psi = _psi(ref_dist, cur_dist)

    return {
        "ref_rows": int(len(ref)),
        "cur_rows": int(len(cur)),
        "ref_avg_text_len": float(ref_len.mean()),
        "cur_avg_text_len": float(cur_len.mean()),
        "avg_text_len_delta_pct": float(
            ((cur_len.mean() - ref_len.mean()) / (ref_len.mean() + 1e-8)) * 100
        ),
        "ref_category_dist": ref_dist.to_dict(),
        "cur_category_dist": cur_dist.to_dict(),
        "psi_category": psi,
    }
