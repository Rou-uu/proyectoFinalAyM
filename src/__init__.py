"""Código fuente del proyecto: triage de reportes de incidentes viales del C5.

Módulos (separación de responsabilidades):
  - config:      semilla, rutas y listas de variables.
  - preparacion: patrón Pipeline para limpiar los datos (datos).
  - modelado:    clasificadores RandomForest y MLP (modelado).
  - clustering:  agrupamiento espacial con DBSCAN (modelado no supervisado).
  - viz:         funciones de graficado reutilizables (reporte/presentación).
"""
