"""Funciones de graficado y estadística descriptiva reutilizables.

Separan la presentación del análisis: los notebooks deciden QUÉ graficar; el
CÓMO vive aquí. Estilo heredado de la Tarea 2.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from . import config


def barras(serie, titulo, xlabel, ax=None, color="steelblue", rot=0):
    """Gráfica de barras simple a partir de una serie ya agregada."""
    ax = ax or plt.gca()
    serie.plot(kind="bar", ax=ax, color=color, edgecolor="black", linewidth=.4)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("conteo")
    ax.tick_params(axis="x", rotation=rot)
    return ax


def guardar(fig, nombre, dir_figuras=None):
    """Guarda una figura en figures/ (crea la carpeta si no existe)."""
    ruta = (dir_figuras or config.DIR_FIGURES)
    ruta.mkdir(parents=True, exist_ok=True)
    destino = ruta / nombre
    fig.savefig(destino, bbox_inches="tight", dpi=120)
    print("Figura guardada:", destino)
    return destino


def resumen_numerico(serie):
    """Resumen descriptivo de una serie numérica (incluye asimetría y curtosis)."""
    s = serie.dropna().astype("float64")
    return pd.Series({
        "n": len(s),
        "media": s.mean(),
        "mediana": s.median(),
        "std": s.std(),
        "min": s.min(),
        "Q1": s.quantile(.25),
        "Q3": s.quantile(.75),
        "max": s.max(),
        "asimetria": stats.skew(s),
        "curtosis": stats.kurtosis(s),  # exceso de curtosis (normal = 0)
    })
