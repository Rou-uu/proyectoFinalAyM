"""Preparación de datos mediante el patrón de diseño Pipeline.

Cada paso de limpieza es una subclase de `PasoLimpieza` (HERENCIA) con una
única responsabilidad. `PipelinePreparacion` los encadena (COMPOSICIÓN) y
registra una bitácora auditable de filas antes/después de cada paso.

La lógica es idéntica a la del notebook 01: quita los duplicados (`D`), quita
el año-ruido 2021 y construye el target binario `REAL`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PasoLimpieza(ABC):
    """Paso atómico de limpieza: una función df -> df con nombre."""

    nombre: str = "paso"

    @abstractmethod
    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class QuitarDuplicados(PasoLimpieza):
    """Elimina los cierres 'D' (duplicado): un evento real ya reportado, cuya
    deduplicación corresponde al sistema de despacho, no al modelo."""

    nombre = "quitar_duplicados"

    def aplicar(self, df):
        return df[df["codigo_cierre"] != "D"].copy()


class QuitarAnioRuido(PasoLimpieza):
    """Elimina los reportes de 2021 (decenas de registros de frontera)."""

    nombre = "quitar_anio_ruido"

    def aplicar(self, df):
        return df[df["ANIO"] >= 2022].copy()


class ConstruirTarget(PasoLimpieza):
    """Construye el target REAL = 1 si el cierre es afirmativo ('A'), 0 si no."""

    nombre = "construir_target"

    def aplicar(self, df):
        df = df.copy()
        df["REAL"] = (df["codigo_cierre"] == "A").astype("int8")
        return df


class PipelinePreparacion:
    """Encadena pasos de limpieza en orden y guarda una bitácora auditable."""

    def __init__(self, pasos: list[PasoLimpieza]):
        self.pasos = pasos
        self.bitacora: list[tuple[str, int, int]] = []

    def aplicar(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        for paso in self.pasos:
            antes = len(df)
            df = paso.aplicar(df)
            self.bitacora.append((paso.nombre, antes, len(df)))
            if verbose:
                print(f"  {paso.nombre:28s} {antes:>8,} -> {len(df):>8,}")
        return df


def pipeline_estandar() -> PipelinePreparacion:
    """Pipeline del proyecto: quitar duplicados -> quitar año-ruido -> target."""
    return PipelinePreparacion(
        [QuitarDuplicados(), QuitarAnioRuido(), ConstruirTarget()]
    )
