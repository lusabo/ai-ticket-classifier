from pydantic import BaseModel, field_validator

class PredictRequest(BaseModel):
    """Schema de entrada do endpoint /predict."""
    texto: str

    @field_validator("texto")
    @classmethod
    def texto_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Texto deve ter pelo menos 3 caracteres.")
        return v

class PredictResponse(BaseModel):
    """Schema de saída do endpoint /predict."""
    categoria: str
    resposta: str