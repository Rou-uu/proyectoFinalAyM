"""Agrupamiento espacial de incidentes con DBSCAN.

`ClusteringEspacial` encapsula la proyección de coordenadas a metros, la curva
k-distance para elegir `eps` y el ajuste de DBSCAN. La lógica es idéntica a la
del notebook 03.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class ClusteringEspacial:
    """DBSCAN sobre coordenadas proyectadas a metros (equirectangular local)."""

    def __init__(self, lat0=19.36, lon0=-99.13, eps=150.0, min_samples=200):
        self.lat0 = lat0
        self.lon0 = lon0
        self.eps = eps
        self.min_samples = min_samples
        self.modelo = None

    # --- proyección equirectangular local centrada en la CDMX ---
    def proyectar(self, lat, lon):
        """(lat, lon) en grados -> (x, y) en metros. Acepta escalares o arrays."""
        x = (lon - self.lon0) * 111320.0 * math.cos(math.radians(self.lat0))
        y = (lat - self.lat0) * 110540.0
        return x, y

    def desproyectar(self, x, y):
        """Inversa de `proyectar`: metros -> (lat, lon)."""
        lon = x / (111320.0 * math.cos(math.radians(self.lat0))) + self.lon0
        lat = y / 110540.0 + self.lat0
        return lat, lon

    def matriz_coords(self, df):
        """DataFrame con latitud/longitud -> matriz (n, 2) en metros."""
        lat = df["latitud"].astype("float64").to_numpy()
        lon = df["longitud"].astype("float64").to_numpy()
        xm, ym = self.proyectar(lat, lon)
        return np.column_stack([xm, ym])

    # --- selección de eps y ajuste ---
    def curva_kdistance(self, X, n_jobs=2):
        """Distancias ordenadas al `min_samples`-ésimo vecino (para el codo)."""
        nn = NearestNeighbors(n_neighbors=self.min_samples, n_jobs=n_jobs).fit(X)
        dist, _ = nn.kneighbors(X)
        return np.sort(dist[:, -1])

    def ajustar(self, X, n_jobs=2):
        """Ajusta DBSCAN y devuelve las etiquetas (-1 = ruido)."""
        self.modelo = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=n_jobs).fit(X)
        return self.modelo.labels_
