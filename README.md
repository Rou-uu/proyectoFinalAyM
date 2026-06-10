# Triage de reportes de incidentes viales del C5 (CDMX, 2022–2024)

**Proyecto Final — Almacenes y Minería de Datos**
Facultad de Ciencias, UNAM

Profesora: Jessica Santizo Galicia
Ayudante de Teoría: Diego Antonio Villalba González · Ayudante de Laboratorio: Ares Gael Castro Romero
Fecha: junio de 2026

---

## 1. Descripción

¿Es real o falso un reporte de incidente vial que llega al C5? Este proyecto
predice, con la información disponible **al momento de recibir un reporte**, si
corresponde a un incidente confirmado (*afirmativo*) o a un reporte
falso/informativo, para apoyar la priorización del despacho de unidades. De
forma complementaria, descubre mediante agrupamiento espacial las **vialidades**
donde se concentran los incidentes y de qué tipo.

Cubre el ciclo completo de minería de datos: diccionario y preparación de datos
(patrón **Pipeline**), análisis exploratorio, modelo supervisado, agrupamiento
no supervisado, persistencia y reutilización del modelo.

- **Dataset:** Incidentes viales reportados por el C5, Portal de Datos Abiertos
  de la CDMX (CC-BY-4.0). 504 261 reportes crudos → **289 885** tras quitar
  duplicados.
- **Variable objetivo:** `REAL` (1 = afirmativo, 0 = falso/informativo);
  balance 63.5 % / 36.5 %.
- **Modelo final:** RandomForest. Comparado contra una línea base trivial y una
  red neuronal (MLP).
- **Agrupamiento:** DBSCAN sobre coordenadas, con mapa interactivo de corredores.

---

## 2. Estructura del repositorio

```
finalProyectoAyM/
├── README.md                     Este archivo
├── requirements.txt              Dependencias con versiones exactas
├── .gitignore
├── _quarto.yml                   Configuración del sitio Quarto (output-dir: docs)
├── index.qmd · mapa.qmd · instalacion.qmd   Páginas del sitio
├── docs/                         Sitio HTML compilado (GitHub Pages sirve de aquí)
│
├── data/
│   ├── raw/                      Crudo (NO versionado; ver paso 4.2)
│   │   └── c5_inViales_2022_2024.csv
│   ├── c5_modelado.parquet       504 261 reportes (con código de cierre)
│   ├── c5_listo.parquet          289 885 reportes depurados (dataset de trabajo)
│   ├── alcaldias.geojson         Límites de las 16 alcaldías (para el mapa)
│   └── corredores_geocodificados.json  Nombres de vialidades de los corredores
│
├── src/                          Código fuente (POO; lo importan los notebooks)
│   ├── config.py                 Semilla, rutas y listas de variables
│   ├── preparacion.py            PasoLimpieza + subclases + PipelinePreparacion
│   ├── modelado.py               ModeloClasificador → ModeloRandomForest / ModeloMLP
│   ├── clustering.py             ClusteringEspacial (proyección + DBSCAN)
│   └── viz.py                    Funciones de graficado y descriptivas
│
├── notebooks/
│   ├── 01_eda.ipynb              Diccionario, patrón Pipeline, EDA y correlaciones
│   ├── 02_clasificacion.ipynb    Baseline + RandomForest + MLP + recarga
│   └── 03_clustering.ipynb       DBSCAN + mapa interactivo de corredores
│
├── diagrams/
│   ├── uml_clases.png            Diagrama UML de clases de src/
│   ├── uml_clases.mmd            Fuente Mermaid del UML
│   └── generar_uml.py            Script que dibuja el UML
│
├── models/
│   ├── modelo_rf.joblib          Modelo final (RandomForest)
│   └── modelo_mlp.joblib         Red neuronal (comparación)
│
├── figures/                      Figuras de los notebooks + mapa_corredores.html
│
└── reports/
    ├── diccionario_datos.md      Diccionario de datos
    ├── reporte.tex               Reporte final (LaTeX)
    └── reporte.pdf               Reporte compilado
```

---

## 3. Requisitos

- **Python 3.12**
- Dependencias en [`requirements.txt`](requirements.txt) (scikit-learn, pandas,
  plotly, etc.). No se usan `torch`, `xgboost` ni `lightgbm`.

El **código fuente** (clases y módulos) vive en [`src/`](src/), organizado por
responsabilidad (datos / modelado / clustering / visualización). Los notebooks
no redefinen lógica: la **importan** desde `src/`. El diseño aplica herencia
(pasos de limpieza y clasificadores) y composición (el pipeline de preparación),
y el patrón de diseño es **Pipeline** en dos niveles (ver
[`diagrams/uml_clases.png`](diagrams/uml_clases.png) y el reporte).

---

## 4. Instalación y ejecución

### 4.1 Entorno

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

### 4.2 Datos

Los datasets procesados (`data/*.parquet`) **ya están incluidos**, por lo que
los notebooks corren sin descargar nada. Solo si quieres regenerar desde el
crudo, descarga el archivo original desde el Portal de Datos Abiertos de la CDMX
(dataset «Incidentes viales reportados por el C5»,
<https://datos.cdmx.gob.mx>) y colócalo en `data/raw/c5_inViales_2022_2024.csv`.

> Nota: el servidor de archivo de datos de la CDMX puede presentar un
> certificado SSL vencido; en ese caso usa `curl -k` para descargar. La fuente
> es oficial y citable a través del registro del portal (licencia CC-BY-4.0).

### 4.3 Ejecutar los notebooks

```bash
python3 -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_eda.ipynb --ExecutePreprocessor.timeout=3600
python3 -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/02_clasificacion.ipynb --ExecutePreprocessor.timeout=3600
python3 -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/03_clustering.ipynb --ExecutePreprocessor.timeout=3600
```

El **mapa interactivo** de corredores queda embebido en `03_clustering.ipynb` y
también como archivo independiente `figures/mapa_corredores.html` (se abre en el
navegador con zoom y desplazamiento).

### 4.4 Reutilización del modelo (sin reentrenar)

La última sección de `02_clasificacion.ipynb` carga el modelo desde disco con
`joblib.load("models/modelo_rf.joblib")` y predice reportes nuevos. Ejemplo de
demostración en vivo (predicción de un reporte de cámara en zona central vs. una
llamada al 911 en la periferia).

### 4.5 Compilar el reporte

```bash
cd reports && pdflatex reporte.tex && pdflatex reporte.tex
```

### 4.6 Sitio Quarto y GitHub Pages

El sitio (portada, los 3 notebooks, mapa interactivo e instalación) ya está
compilado en `docs/`. Para reconstruirlo tras editar los notebooks:

```bash
quarto render
```

Para publicarlo: sube el repositorio a GitHub y activa **Settings → Pages →
Branch: `main` / carpeta `/docs`**. La URL quedará como
`https://<usuario>.github.io/<repo>/` (pégala en la sección 8).

---

## 5. Reproducibilidad

- **Semilla única `42`** en particiones, modelos y muestreos.
- Partición estratificada **80/20**; los umbrales/ajustes se eligen en
  validación, nunca en el conjunto de prueba.
- Preparación de datos mediante el **patrón Pipeline** (clase
  `PipelinePreparacion` en `01_eda.ipynb`) con bitácora auditable
  (504 261 → 289 885 filas).
- Versiones exactas en `requirements.txt`.

---

## 6. Resultados principales (conjunto de prueba)

| Modelo | Accuracy | F1 macro | AUC |
|---|---|---|---|
| Línea base (mayoritario) | 0.635 | 0.388 | 0.500 |
| **RandomForest (final)** | 0.670 | **0.658** | **0.729** |
| Red neuronal (MLP) | 0.686 | 0.638 | 0.726 |

- El RandomForest supera ampliamente a la línea base (F1 macro +0.27).
- El **canal de reporte** es el predictor más fuerte de la veracidad (radio y
  cámara ~93 % confirmados vs. llamada al 911 ~60 %).
- DBSCAN identifica corredores/cruceros viales con nombre propio (Circuito
  Interior, Calz. Ignacio Zaragoza, Av. Tláhuac, …).

Detalle completo en [`reports/reporte.pdf`](reports/reporte.pdf) y
[`reports/diccionario_datos.md`](reports/diccionario_datos.md).

---

## 7. Integrantes

| Integrante | Número de cuenta |
|---|---|
| José Rubén Alfaro González | 320516436 |

## 8. Entregables en línea

- **Repositorio:** <https://github.com/Rou-uu/proyectoFinalAyM>
- **Sitio (GitHub Pages):** <https://rou-uu.github.io/proyectoFinalAyM/>
  (activar en *Settings → Pages → Branch: `main` / carpeta `/docs`*)
- **Video de presentación:** TODO
