from __future__ import annotations

from pathlib import Path

# Caminho padrão para o dataset de evasão
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "evasao_curso_semestre_treino_MSDOS.csv"

# Colunas de atributos utilizadas pelo modelo
FEATURE_COLUMNS = [
    "ID_CURSO",
    "SEMESTRE",
    "CH_CURSADA_TOTAL",
    "CH_TOTAL_CURSO",
    "CH_TOTAL_SEMESTRE",
    "CH_APROV_SEM",
    "MEDIA_DISC_SEMESTRE",
    "TOT_REPROV",
    "QTD_DISC_SEM",
    "PERC_REPROV",
    "BOM_PAGADOR",
]

# Coluna alvo
TARGET_COLUMN = "CLASSE"

# Hiperparâmetros de treinamento
TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
RANDOM_STATE = 42
