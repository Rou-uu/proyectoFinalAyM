"""Clasificadores del proyecto mediante el patrón Pipeline de scikit-learn.

`ModeloClasificador` es la clase base abstracta (HERENCIA) que encapsula el
preprocesamiento común y el ensamblado del `Pipeline`. `ModeloRandomForest` y
`ModeloMLP` solo difieren en el tratamiento de las numéricas y en el estimador,
de modo que la herencia es natural: comparten el flujo, cambian las piezas.

Cada `construir(**params)` devuelve EXACTAMENTE el mismo `Pipeline` que el
código inline del notebook 02 (mismo `ColumnTransformer`, mismos
hiperparámetros), por lo que los resultados son idénticos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class ModeloClasificador(ABC):
    """Clase base: define el flujo prep -> clf y deja que las subclases
    decidan el tratamiento numérico y el estimador."""

    def __init__(self, numericas=None, categoricas=None):
        self.numericas = list(numericas) if numericas is not None else list(config.NUMERICAS)
        self.categoricas = list(categoricas) if categoricas is not None else list(config.CATEGORICAS)

    # --- piezas que cambian entre subclases ---
    @abstractmethod
    def _ramas(self) -> list:
        """Lista de transformers (nombre, transformador, columnas) para el
        ColumnTransformer, en el orden de la subclase."""

    @abstractmethod
    def _crear_estimador(self, **params):
        """Estimador final (RandomForest o MLP) con los params dados."""

    # --- flujo común ---
    def _preprocesador(self) -> ColumnTransformer:
        return ColumnTransformer(transformers=self._ramas())

    def construir(self, **params) -> Pipeline:
        """Patrón Pipeline: prep -> clf en un solo objeto."""
        return Pipeline([("prep", self._preprocesador()), ("clf", self._crear_estimador(**params))])

    @staticmethod
    def _rama_categorica_imputada():
        """One-hot precedido de imputación por moda (para árboles, que no
        aceptan NaN nativo)."""
        return Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])


class ModeloRandomForest(ModeloClasificador):
    """RandomForest balanceado. Las numéricas pasan directas (los árboles no
    necesitan escalado); opción `imputar_num` para el experimento de fuga."""

    def __init__(self, numericas=None, categoricas=None, imputar_num=False):
        super().__init__(numericas, categoricas)
        self.imputar_num = imputar_num

    def _ramas(self):
        rama_num = SimpleImputer(strategy="median") if self.imputar_num else "passthrough"
        return [
            ("cat", self._rama_categorica_imputada(), self.categoricas),
            ("num", rama_num, self.numericas),
        ]

    def _crear_estimador(self, **params):
        base = dict(n_estimators=300, class_weight="balanced", random_state=config.SEMILLA)
        base.update(params)
        return RandomForestClassifier(**base)


class ModeloMLP(ModeloClasificador):
    """Red neuronal multicapa. Las numéricas se estandarizan (el descenso de
    gradiente es sensible a la escala)."""

    def _ramas(self):
        return [
            ("num", StandardScaler(), self.numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categoricas),
        ]

    def _crear_estimador(self, **params):
        base = dict(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            early_stopping=True,
            max_iter=60,
            random_state=config.SEMILLA,
        )
        base.update(params)
        return MLPClassifier(**base)
