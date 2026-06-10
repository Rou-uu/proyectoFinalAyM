"""Configuración central: semilla, rutas y listas de variables.

Las rutas se resuelven a partir de la ubicación de este archivo
(`BASE = raíz del proyecto`), de modo que funcionan sin importar el directorio
de trabajo desde el que se ejecute un notebook o un script.
"""

from pathlib import Path

# Semilla única de reproducibilidad (usada en todo split, modelo y muestreo).
SEMILLA = 42

# Raíz del proyecto = carpeta que contiene a src/.
BASE = Path(__file__).resolve().parents[1]
DIR_DATA = BASE / "data"
DIR_MODELS = BASE / "models"
DIR_FIGURES = BASE / "figures"
DIR_REPORTS = BASE / "reports"

# --- Variables del modelo supervisado ---
# Información disponible AL RECIBIR el reporte (sin fuga).
NUMERICAS = ["HORA", "MES", "ANIO", "latitud", "longitud"]
CATEGORICAS = [
    "tipo_incidente_c4",
    "incidente_c4",
    "tipo_entrada",
    "alcaldia_catalogo",
    "dia_semana",
    "FRANJA",
]

# Variables que solo se conocen AL CERRAR el reporte: fuga de información.
# Nunca entran al modelo (ver el experimento de fuga en el notebook 02).
LEAKAGE = ["clas_con_f_alarma", "TIEMPO_CIERRE_MIN"]

# Variable objetivo.
TARGET = "REAL"
