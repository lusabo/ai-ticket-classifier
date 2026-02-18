import pandas as pd
from ticket_ai.data.quality import DataQualityChecker

def test_quality_checker_detects_empty_dataframe():
    df = pd.DataFrame(columns=["texto", "categoria"])
    checker = DataQualityChecker(df)
    result = checker.run_all_checks()
    assert result["is_valid"] is False
    assert "DataFrame vazio." in result["issues"]

def test_quality_checker_valid_dataset():
    df = pd.DataFrame({
        "texto": [
            "Quero cancelar minha assinatura e entender a cobrança recente.",
            "Produto chegou com defeito e preciso trocar."
        ],
        "categoria": [
            "assinatura",
            "logistica"
        ]
    })
    checker = DataQualityChecker(df)
    result = checker.run_all_checks()
    assert result["is_valid"] is True
    assert result["issues"] == []