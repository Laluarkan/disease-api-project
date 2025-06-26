from pydantic import BaseModel
from typing import Dict

class GejalaInput(BaseModel):
    jawaban: Dict[str, int]  # {"fever": 1, "cough": 0, ...}

class PenyakitInput(BaseModel):
    nama_penyakit: str
