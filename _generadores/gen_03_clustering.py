"""Generador del notebook notebooks/03_clustering.ipynb (clustering espacial,
SOLO DBSCAN).

Esto es solo TOOLING: arma las celdas con nbformat. El código DENTRO de las
celdas es SIMPLE e INLINE (estilo tarea del curso): se lee el parquet con
pandas, se proyecta lat/lon a metros, se corre DBSCAN en una celda y se grafica;
a lo sumo se definen funciones auxiliares pequeñas inline (proyectar, desproyectar,
anillos_geojson). Nada de paquetes src/, clases, factories ni repositorios.

Para regenerar:  python3 _generadores/gen_03_clustering.py
Para ejecutar:   python3 -m jupyter nbconvert --to notebook --execute --inplace \
                     notebooks/03_clustering.ipynb --ExecutePreprocessor.timeout=3600
"""

from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "notebooks" / "03_clustering.ipynb"

nb = nbf.v4.new_notebook()
celdas = []


def md(texto: str) -> None:
    celdas.append(nbf.v4.new_markdown_cell(texto.strip("\n")))


def code(texto: str) -> None:
    celdas.append(nbf.v4.new_code_cell(texto.strip("\n")))


# --------------------------------------------------------------------------- #
# 0. Encabezado
# --------------------------------------------------------------------------- #
md(
    r"""
# Triage de reportes de incidentes viales del C5 (CDMX 2022–2024)
## Notebook 3 — Clustering espacial: corredores viales con DBSCAN

**Proyecto Final — Almacenes y Minería de Datos**

Profesora: Jessica Santizo Galicia
Ayudantes: Diego Antonio Villalba González y Ares Gael Castro Romero
Integrante: José Rubén Alfaro González — No. de cuenta: 320516436
Fecha: 2026-06

---

En los notebooks anteriores hicimos EDA (`01_eda.ipynb`) y un clasificador
supervisado (`02_clasificacion.ipynb`). Aquí cambiamos de pregunta: en lugar de
*predecir* si un reporte se confirma, buscamos **estructura geográfica** en los
incidentes. Concretamente: **¿en qué vialidades de la CDMX se concentran los
incidentes viales, y de qué tipo son?**

Usamos **DBSCAN** sobre las coordenadas (`latitud`, `longitud`). El target `REAL`
**no** participa en el clustering: lo usamos solo *después* para caracterizar
cada corredor (validación externa).
"""
)

md(
    r"""
### ¿Por qué DBSCAN y no K-Means?

DBSCAN agrupa por **densidad**: junta puntos que tienen muchos vecinos cerca y
deja como **ruido** los que están aislados. Esto encaja con nuestra pregunta por
tres razones:

1. **Encuentra zonas densas de forma libre.** Un corredor vial es una nube
   *alargada* que sigue el trazo de una avenida. K-Means solo arma grupos
   ~esféricos (fronteras convexas) y nunca capturaría un corredor lineal.
2. **No hay que fijar `k`.** El número de corredores **emerge** de la densidad de
   los datos; no lo imponemos a mano como el `k` de K-Means.
3. **Separa el ruido.** La mayoría de los incidentes de la ciudad están dispersos,
   fuera de cualquier corredor denso. DBSCAN los etiqueta como ruido en vez de
   forzarlos a un grupo; ese ruido **es señal** (incidentes difusos por toda la
   traza urbana), no un error.

Por eso aquí usamos **solo DBSCAN**: es la herramienta correcta para descubrir
corredores y cruceros viales con nombre propio.
"""
)

# --------------------------------------------------------------------------- #
# 1. Imports y carga
# --------------------------------------------------------------------------- #
md(
    r"""
## 1. Carga de datos

Leemos el dataset de trabajo `data/c5_listo.parquet` (289,885 incidentes): el mismo
que usan los notebooks 01 y 02, **ya depurado** (sin duplicados `D`, sin el año-ruido
2021) y con el target nuevo `REAL`. Usamos rutas **relativas a la ubicación del
notebook** con `pathlib`: el notebook vive en `notebooks/`, así que los datos están en
`../data/`.

`REAL` no participa en el clustering; lo cargamos solo para **caracterizar** los
corredores *después* (validación externa). En este dataset el balance global es
**63.5 % afirmativos** (`REAL = 1`) frente a 36.5 % falsos/informativos (`REAL = 0`).
"""
)

code(
    r"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
SEMILLA = 42  # semilla única para todo split/modelo/muestreo

# Rutas relativas a la ubicación del notebook (notebooks/ -> ../data, ../figures)
DIR_DATA = Path("..") / "data"
DIR_FIG = Path("..") / "figures"
DIR_FIG.mkdir(exist_ok=True)

df = pd.read_parquet(DIR_DATA / "c5_listo.parquet").reset_index(drop=True)
print(f"Incidentes cargados: {len(df):,} filas x {df.shape[1]} columnas")
print(f"% afirmativos global (target REAL=1): {df['REAL'].mean() * 100:.1f}%")
df[["latitud", "longitud", "incidente_c4", "alcaldia_catalogo", "FRANJA", "REAL"]].head()
"""
)

# --------------------------------------------------------------------------- #
# 2. Proyección a metros
# --------------------------------------------------------------------------- #
md(
    r"""
## 2. Proyección de lat/lon a metros

DBSCAN necesita un radio de vecindad `eps`. Sobre `latitud`/`longitud` en
**grados**, `eps` no tiene un significado físico claro: a la latitud de la CDMX un
grado de longitud (~105 km) y uno de latitud (~110.5 km) miden distinto, así que
una distancia euclídea en grados deforma el mapa.

Proyectamos a **metros** con una equirectangular local centrada en la CDMX:

$$x = (\text{lon}-\text{lon}_0)\cdot 111320 \cdot \cos(\text{lat}_0), \qquad
  y = (\text{lat}-\text{lat}_0)\cdot 110540$$

con $\text{lat}_0 = 19.36$, $\text{lon}_0 = -99.13$. Tras proyectar, **`eps` está en
metros sobre el asfalto**: un `eps` de 150 m se lee directamente como *"incidentes a
menos de 150 m unos de otros pertenecen al mismo corredor"*. Así el parámetro es
interpretable y reproducible.
"""
)

code(
    r"""
LAT0, LON0 = 19.36, -99.13  # centro de proyección (centro de la CDMX)


def proyectar(lat, lon):
    '''Equirectangular local (lat, lon en grados) -> (x, y) en metros.'''
    x = (lon - LON0) * 111320.0 * math.cos(math.radians(LAT0))
    y = (lat - LAT0) * 110540.0
    return x, y


def desproyectar(x, y):
    '''Inversa de proyectar: metros -> (lat, lon). Para nombrar centroides.'''
    lon = x / (111320.0 * math.cos(math.radians(LAT0))) + LON0
    lat = y / 110540.0 + LAT0
    return lat, lon


# Las coordenadas son Float64 (nullable de PyArrow); para sklearn las casteamos a
# float64 de numpy antes de proyectar (un par de líneas, como en una tarea).
lat = df["latitud"].astype("float64").to_numpy()
lon = df["longitud"].astype("float64").to_numpy()
xm, ym = proyectar(lat, lon)
X = np.column_stack([xm, ym])  # (289885, 2) en metros

print(f"Proyectados {X.shape[0]:,} puntos a metros.")
print(f"  x (oeste-este): [{X[:, 0].min():.0f}, {X[:, 0].max():.0f}] m")
print(f"  y (sur-norte):  [{X[:, 1].min():.0f}, {X[:, 1].max():.0f}] m")
"""
)

# --------------------------------------------------------------------------- #
# 3. Selección de eps: curva k-distance
# --------------------------------------------------------------------------- #
md(
    r"""
## 3. Selección de `eps`: la curva k-distance

El método estándar para elegir `eps` es la **curva k-distance**: para cada punto se
calcula la distancia a su $k$-ésimo vecino (con $k = $ `min_samples`), se ordenan
ascendentemente y se busca el **codo** —donde la curva pasa de plana (zona densa) a
empinada (zona dispersa)—. Ese codo separa *núcleo denso* de *ruido*.

Calculamos la curva sobre las **~290k filas completas**, la misma densidad sobre la
que correrá el DBSCAN final. (Una submuestra al ~10% bajaría la densidad local ~10×
y desplazaría el codo a un `eps` engañoso.)
"""
)

code(
    r"""
from sklearn.neighbors import NearestNeighbors

MIN_SAMPLES = 200  # núcleo denso de un corredor (~200 incidentes en 2022-2024)
EPS_M = 150.0      # metros; ~ancho de un crucero / vialidad densa

# Distancia al MIN_SAMPLES-ésimo vecino, para cada punto, sobre el conjunto completo.
nn = NearestNeighbors(n_neighbors=MIN_SAMPLES, n_jobs=2).fit(X)
dist_vecinos, _ = nn.kneighbors(X)
kdist = np.sort(dist_vecinos[:, -1])          # distancia al k-ésimo vecino, ordenada
pct = np.linspace(0, 100, len(kdist))         # percentil de cada punto
cruce = float((kdist <= EPS_M).mean() * 100)  # % de puntos con su k-vecino a <= eps

fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(pct, kdist, color="#1f77b4", lw=1.6,
        label=f"distancia al {MIN_SAMPLES}º vecino")
ax.axhline(EPS_M, color="#d62728", ls="--", label=f"eps elegido = {EPS_M:.0f} m")
ax.axvline(cruce, color="0.4", ls=":", label=f"codo ~ percentil {cruce:.0f}")
ax.set_ylim(0, 1200)
ax.set_xlabel("Percentil de puntos (ordenados por densidad)")
ax.set_ylabel(f"Distancia al {MIN_SAMPLES}º vecino (m)")
ax.set_title("Curva k-distance para elegir eps (sobre las ~290k filas)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

print(f"eps = {EPS_M:.0f} m cruza la curva en el percentil ~{cruce:.0f}.")
"""
)

md(
    r"""
**Interpretación.** La curva es **plana y baja** solo en su primer tramo (el ~12 % de
los puntos más densos, con su 200º vecino a ≤ 150 m) y enseguida **se dispara**: ese
despegue temprano es el codo. Por debajo del codo, los puntos tienen sus 200 vecinos muy
cerca (cruceros y corredores reales); por encima, la distancia al 200º vecino crece
rápido (incidentes dispersos). Elegimos **`eps` = 150 m** —el ancho típico de un crucero
o una vialidad densa— y **`min_samples` = 200**, deliberadamente **alto**: no buscamos
micro-cúmulos de cuatro choques, sino **corredores con masa crítica** (≥200 incidentes en
2022–2024). *Implicación:* con estos valores la **mayor parte** de los puntos quedará como
ruido, y eso es lo esperado —queremos quedarnos solo con las vialidades densas e
identificables, no con todo el mapa.
"""
)

# --------------------------------------------------------------------------- #
# 4. DBSCAN
# --------------------------------------------------------------------------- #
md(
    r"""
## 4. DBSCAN sobre las ~290k coordenadas

Corremos DBSCAN con los parámetros justificados arriba sobre las coordenadas
**proyectadas a metros**. La etiqueta `-1` es ruido (puntos que no pertenecen a
ningún núcleo denso).
"""
)

code(
    r"""
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=EPS_M, min_samples=MIN_SAMPLES, n_jobs=2).fit(X)
df["cluster"] = db.labels_

n_ruido = int((df["cluster"] == -1).sum())
pct_ruido = n_ruido / len(df) * 100
n_clusters = df.loc[df["cluster"] != -1, "cluster"].nunique()

print(f"Clusters densos encontrados: {n_clusters}")
print(f"Ruido: {n_ruido:,} puntos ({pct_ruido:.1f}%)")
print(f"Puntos en algún corredor: {len(df) - n_ruido:,} ({100 - pct_ruido:.1f}%)")
"""
)

md(
    r"""
**Interpretación.** DBSCAN descubre **~170 cúmulos densos** y deja alrededor de
**tres cuartas partes de los incidentes como ruido** (~77 %). Ese ruido alto **no es un
fallo del método**: tras quitar los duplicados, la mayoría de los incidentes viales de
la CDMX ocurren *fuera* de corredores densos, repartidos por toda la traza urbana. Solo
~1 de cada 5 incidentes cae en una vialidad con masa crítica (≥200 reportes en 2022–2024).
*Implicación operativa:* hay una minoría "concentrada" que se puede atacar por corredores
concretos (lo que sigue) y una mayoría "difusa" que exige cobertura general; distinguirlas
ya es información útil.
"""
)

# --------------------------------------------------------------------------- #
# 5. Caracterización de los clusters top
# --------------------------------------------------------------------------- #
md(
    r"""
## 5. Caracterización de los corredores más grandes

Nos quedamos con los **12 clusters más grandes** (los corredores con más masa) y, para
cada uno, calculamos con un `groupby` simple:

- **n**: número de incidentes.
- **% REAL**: tasa de reportes afirmativos (`REAL = 1`; caracterización *a posteriori*,
  el target no entró en DBSCAN).
- **tipo de incidente dominante** (`incidente_c4`) con su porcentaje.
- **alcaldía dominante** y **franja pico**.

Además medimos la **morfología** con PCA 2D sobre los puntos de cada cluster:

- **elongación** $= \sqrt{\text{var}_1/\text{var}_2}$ (qué tan estirada está la nube),
- **longitud** = extensión a lo largo del eje principal (PC1), en km.

Clasificamos como **CORREDOR** si elongación $\ge 3$ **y** longitud $\ge 1$ km; en otro
caso, **CRUCERO/ZONA** (mancha compacta).
"""
)

code(
    r"""
from sklearn.decomposition import PCA

GLOB_REAL = df["REAL"].mean() * 100  # 63.5% afirmativos global

tam = df.loc[df["cluster"] != -1, "cluster"].value_counts()
top = list(tam.head(12).index)  # los 12 clusters más grandes

filas = []
centroides = {}  # cluster -> (lat, lon) del centroide, para nombrar y mapear
for cl in top:
    g = df[df["cluster"] == cl]
    pts = X[g.index.to_numpy()]  # puntos en metros de este cluster

    # Centroide (en metros -> lat/lon)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    clat, clon = desproyectar(cx, cy)
    centroides[cl] = (clat, clon)

    # Morfología por PCA
    p = PCA(n_components=2, random_state=SEMILLA).fit(pts)
    var = p.explained_variance_
    elong = math.sqrt(var[0] / var[1]) if var[1] > 0 else float("inf")
    proj1 = p.transform(pts)[:, 0]
    long_km = (proj1.max() - proj1.min()) / 1000.0
    clase = "CORREDOR" if (elong >= 3 and long_km >= 1) else "CRUCERO/ZONA"

    inc = g["incidente_c4"].value_counts()
    filas.append({
        "cluster": int(cl),
        "n": len(g),
        "pct_real": round(g["REAL"].mean() * 100, 1),
        "incidente_dom": inc.index[0],
        "pct_inc_dom": round(inc.iloc[0] / len(g) * 100, 1),
        "alcaldia_dom": g["alcaldia_catalogo"].value_counts().index[0],
        "franja_pico": g["FRANJA"].value_counts().index[0],
        "elongacion": round(elong, 2),
        "longitud_km": round(long_km, 2),
        "clase": clase,
    })

caracter = pd.DataFrame(filas)
n_corr = (caracter["clase"] == "CORREDOR").sum()
print(f"De los {len(top)} clusters top: {n_corr} corredores lineales y "
      f"{len(top) - n_corr} cruceros/zonas compactas.")
caracter
"""
)

md(
    r"""
**Interpretación.** Tras quitar los duplicados, los clusters más grandes son
mayoritariamente **cruceros / zonas compactas** (intersecciones y nodos donde se acumulan
choques) y solo unos pocos alcanzan la forma de **corredor lineal** (elongación ≥ 3 y
≥ 1 km de longitud): la deduplicación adelgaza las líneas y deja sobre todo manchas
densas alrededor de los nodos. El **incidente dominante** es casi siempre *Choque sin
lesionados* —coherente con vialidades de alto flujo— y la **franja pico** cae en
tarde/noche (hora pico del tránsito). *Implicación:* el tipo de despacho difiere —un
crucero se cubre con una grúa o patrulla fija; un corredor exige patrullaje *a lo largo*
del tramo—, así que la morfología no es un dato cosmético sino operativo.
"""
)

# --------------------------------------------------------------------------- #
# 6. Nombres de avenida
# --------------------------------------------------------------------------- #
md(
    r"""
## 6. Nombres reales de las vialidades

Un cluster es solo un centroide; para que sea **accionable** necesita un nombre de
avenida. En `data/corredores_geocodificados.json` ya tenemos una lista de corredores
**geocodificados** (clave `cluster`, más `lat`, `lon`, `alcaldia`, `road`, `vialidad`,
etc.), preparada de antemano para no llamar a ninguna API desde el notebook.

**Método de cruce.** Asignamos a cada centroide del top el nombre del registro
geocodificado **más cercano** (distancia en metros sobre las coordenadas proyectadas).
Este criterio por proximidad es robusto aunque el etiquetado numérico de DBSCAN difiera
del usado al geocodificar (el JSON se geocodificó sobre otra corrida): siempre toma la
vialidad del corredor que cae sobre el mismo punto del mapa. Para no **mal-etiquetar**,
imponemos un **umbral de 1,000 m**: si el registro geocodificado más cercano queda más
lejos, no hay un nombre confiable y usamos una etiqueta genérica con la alcaldía
dominante. La columna `dist_geo_m` deja auditable la calidad de cada cruce.
"""
)

code(
    r"""
geo = json.loads((DIR_DATA / "corredores_geocodificados.json").read_text(encoding="utf-8"))

# Centroides geocodificados, proyectados a metros para medir distancia.
gx, gy = proyectar(np.array([g["lat"] for g in geo]),
                   np.array([g["lon"] for g in geo]))


def vialidad_mas_cercana(clat, clon):
    '''Nombre del corredor geocodificado mas cercano al centroide (clat, clon).'''
    cx, cy = proyectar(clat, clon)
    d2 = (gx - cx) ** 2 + (gy - cy) ** 2
    j = int(np.argmin(d2))
    return geo[j]["vialidad"], math.sqrt(d2[j])


UMBRAL_NOMBRE_M = 1000.0  # más lejos que esto -> no hay nombre confiable

alcaldia_dom = dict(zip(caracter["cluster"], caracter["alcaldia_dom"]))
nombre = {}        # cluster -> vialidad (con colonia)
nombre_corto = {}  # cluster -> solo la avenida (antes del " · ")
dist_geo = {}      # cluster -> distancia (m) al registro geocodificado
for cl in top:
    clat, clon = centroides[cl]
    vial, dist_m = vialidad_mas_cercana(clat, clon)
    dist_geo[cl] = dist_m
    if dist_m <= UMBRAL_NOMBRE_M:
        nombre[cl] = vial
        nombre_corto[cl] = vial.split(" · ")[0]
        marca = ""
    else:  # cruce poco fiable -> etiqueta genérica por alcaldía
        nombre[cl] = f"Zona densa ({alcaldia_dom[cl]})"
        nombre_corto[cl] = f"Zona densa ({alcaldia_dom[cl]})"
        marca = "  [lejano -> etiqueta genérica]"
    print(f"cluster {cl:>4} (n={tam[cl]:5d}) -> {nombre_corto[cl][:42]:42s} | "
          f"centroide a {dist_m:6.0f} m del registro geocodificado{marca}")

caracter["vialidad"] = caracter["cluster"].map(nombre_corto)
caracter["dist_geo_m"] = caracter["cluster"].map(lambda c: round(dist_geo[c], 0))
caracter[["vialidad", "n", "pct_real", "incidente_dom", "alcaldia_dom",
          "franja_pico", "clase", "longitud_km", "dist_geo_m"]]
"""
)

md(
    r"""
**Interpretación.** Buena parte de los nodos densos cae sobre la red primaria real de la
CDMX —**Circuito Interior (Melchor Ocampo)**, **Avenida Río Churubusco**, **Calzada
Ignacio Zaragoza**, **Distribuidor Vial San Antonio**— más cruceros densos en Venustiano
Carranza, Gustavo A. Madero, Benito Juárez e Iztapalapa. Donde el centroide cae a pocos
metros del registro geocodificado (`dist_geo_m` pequeño) el nombre es confiable; donde el
registro más cercano queda lejos (recordemos que el JSON se geocodificó sobre **otra**
corrida de DBSCAN), preferimos la **etiqueta genérica por alcaldía** antes que arriesgar
un nombre equivocado. *Implicación:* los nodos con nombre confiable quedan directamente
accionables para patrullaje; el resto se localiza por alcaldía.
"""
)

# --------------------------------------------------------------------------- #
# 7. Mapas
# --------------------------------------------------------------------------- #
md(
    r"""
## 7. Los mapas: corredores sobre la ciudad

La petición central del equipo: **ver el resultado sobre la ciudad**. Primero un mapa
estático (matplotlib) con el contorno oficial de las 16 alcaldías; después el mapa
**interactivo** con zoom (Plotly + OpenStreetMap).

### 7.1 Mapa estático

Dibujamos el contorno de las alcaldías parseando `data/alcaldias.geojson` a mano
(con `json` + matplotlib, ya que **no** hay geopandas en el entorno), el **ruido en
gris claro** (la mitad dispersa), los clusters coloreados y los **8 corredores
principales** etiquetados con su nombre de vialidad. Para que el dibujo sea ligero,
tomamos una muestra de ≤40k puntos.
"""
)

code(
    r"""
def anillos_geojson(ruta):
    '''Parseo manual del GeoJSON de alcaldias -> lista de anillos (arrays lon/lat).'''
    gj = json.loads(Path(ruta).read_text(encoding="utf-8"))
    anillos = []
    for f in gj["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            anillos.append(np.asarray(poly[0], dtype=float))  # anillo exterior
    return anillos


anillos = anillos_geojson(DIR_DATA / "alcaldias.geojson")
top8 = top[:8]  # los 8 corredores a etiquetar

# Muestra <=40k para dibujar (semilla 42).
muestra = df.sample(n=min(40000, len(df)), random_state=SEMILLA)
es_top = muestra["cluster"].isin(top)
es_ruido = muestra["cluster"] == -1

fig, ax = plt.subplots(figsize=(9.5, 9.5))
for an in anillos:                                   # contorno de alcaldias
    ax.plot(an[:, 0], an[:, 1], color="0.35", lw=0.8, zorder=1)

ruido = muestra[es_ruido]                            # ruido gris de fondo
ax.scatter(ruido["longitud"], ruido["latitud"], s=2, c="0.82", alpha=0.5,
           zorder=2, label="Ruido (disperso, ~77%)")

otros = muestra[~es_ruido & ~es_top]                 # otros clusters densos
ax.scatter(otros["longitud"], otros["latitud"], s=3, c="#9fb8d6", alpha=0.45,
           zorder=3, label="Otros clusters densos")

paleta = sns.color_palette("tab20", len(top8))       # top-8 coloreados + etiqueta
for i, cl in enumerate(top8):
    sub = muestra[muestra["cluster"] == cl]
    ax.scatter(sub["longitud"], sub["latitud"], s=8, color=paleta[i], zorder=4)
    clat, clon = centroides[cl]
    ax.annotate(nombre_corto[cl], (clon, clat), fontsize=8, fontweight="bold",
                color="black", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=paleta[i], alpha=0.85))

ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.set_title("Corredores de incidentes viales (DBSCAN) sobre las 16 alcaldias de la CDMX")
ax.legend(loc="lower left", fontsize=8)
ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()
plt.savefig(DIR_FIG / "figura_dbscan_vialidades.png", dpi=130, bbox_inches="tight")
plt.show()
print("Figura guardada en figures/figura_dbscan_vialidades.png")
"""
)

md(
    r"""
**Interpretación.** El mapa cuenta la historia de un vistazo: sobre la silueta real de
la ciudad, los nodos coloreados resaltan **cruceros y ejes viales reconocibles** (centro,
oriente y sur), mientras el gris de fondo —cerca del 77 % de los incidentes— recuerda que
la mayor parte del problema está **dispersa**. Esta es la ventaja de DBSCAN: no particiona
la ciudad, **resalta las estructuras densas** y deja el resto como fondo. *Implicación:*
la imagen es directamente presentable a quien asigna recursos.
"""
)

md(
    r"""
### 7.2 Mapa interactivo (Plotly + OpenStreetMap)

Aquí está la petición estrella: el **mapa interactivo incrustado en el notebook**. Con
`plotly.express.scatter_map` sobre teselas de OpenStreetMap, cada punto va coloreado por
corredor y el *hover* muestra **vialidad, tipo de incidente, alcaldía, franja y % real**.
La celda queda como un **objeto Plotly interactivo** (zoom / pan / hover). Limitamos la
muestra a ≤40k puntos para un output fluido.
"""
)

code(
    r"""
import plotly.express as px
import plotly.io as pio

# plotly_mimetype -> el output del notebook se guarda como JSON Plotly nativo
# (application/vnd.plotly.v1+json), que se renderiza con zoom/pan/hover sin inflar
# el .ipynb con una copia de plotly.js.
pio.renderers.default = "plotly_mimetype"

top_set = set(top)
pct_real_corr = dict(zip(caracter["cluster"], caracter["pct_real"]))


def grupo(cl):
    cl = int(cl)
    if cl in top_set:
        return nombre_corto[cl]
    return "Ruido (disperso)" if cl == -1 else "Otros clusters densos"


dvis = df.sample(n=min(40000, len(df)), random_state=SEMILLA).copy()
dvis["lat"] = dvis["latitud"].astype("float64")
dvis["lon"] = dvis["longitud"].astype("float64")
dvis["corredor"] = dvis["cluster"].map(grupo)
dvis["pct_real_corredor"] = dvis["cluster"].map(lambda c: pct_real_corr.get(int(c), np.nan))

fig = px.scatter_map(
    dvis, lat="lat", lon="lon", color="corredor",
    hover_data={"incidente_c4": True, "alcaldia_catalogo": True, "FRANJA": True,
                "pct_real_corredor": ":.1f", "lat": False, "lon": False,
                "corredor": False},
    map_style="open-street-map", center={"lat": 19.36, "lon": -99.13},
    zoom=10, height=720, opacity=0.6,
    title="Corredores de incidentes viales — CDMX (DBSCAN, interactivo)")
fig.update_layout(legend=dict(font=dict(size=9)), margin=dict(l=0, r=0, t=40, b=0))
fig.show()
"""
)

md(
    r"""
**Interpretación.** Sobre las teselas reales de OpenStreetMap se confirma que los
corredores siguen avenidas concretas. El mapa es **interactivo**: se puede hacer zoom y
*pan*, y al pasar el cursor cada punto muestra su vialidad, tipo de incidente, alcaldía,
franja y % confirmado del corredor. *Implicación:* es la vista ideal para la demo en
vivo, porque permite "acercarse" a cada corredor durante la presentación.
"""
)

md(
    r"""
También guardamos el mapa como **HTML standalone** en `figures/mapa_corredores.html`.
Ese archivo se abre directamente en cualquier **navegador** (no requiere Jupyter), con
el mismo zoom libre y hover, y es muy útil para la demo o para compartir el resultado.
"""
)

code(
    r"""
salida_html = DIR_FIG / "mapa_corredores.html"
fig.write_html(salida_html, include_plotlyjs="cdn")
print(f"Mapa interactivo guardado en {salida_html} "
      f"({salida_html.stat().st_size / 1e6:.1f} MB).")
print("Abrelo en el navegador para la demo (zoom y pan libres, sin Jupyter).")
"""
)

# --------------------------------------------------------------------------- #
# 8. Evaluación del clustering
# --------------------------------------------------------------------------- #
md(
    r"""
## 8. Evaluación del clustering

### 8.1 Coeficiente de silueta (cohesión / separación)

La **silueta** mide, para cada punto, qué tan cerca está de su propio cluster frente al
cluster vecino más próximo (de −1 a 1; más alto = clusters más cohesionados y
separados). La calculamos sobre los puntos **no-ruido** (DBSCAN no asigna silueta al
ruido) con una muestra de ≤20,000 puntos y semilla 42.
"""
)

code(
    r"""
from sklearn.metrics import silhouette_score

no_ruido = df[df["cluster"] != -1]
muestra_sil = no_ruido.sample(n=min(20000, len(no_ruido)), random_state=SEMILLA)
Xs = X[muestra_sil.index.to_numpy()]
etiq = muestra_sil["cluster"].to_numpy()

sil = silhouette_score(Xs, etiq, random_state=SEMILLA)
print(f"Coeficiente de silueta (no-ruido, muestra {len(muestra_sil):,}): {sil:.3f}")
"""
)

md(
    r"""
**Interpretación.** La silueta es **alta y positiva** (~0.75), lo que indica que los
puntos densos están, en promedio, **mucho más cerca de su propio cluster que del vecino**:
los grupos de DBSCAN tienen cohesión y separación reales, no son cortes arbitrarios. El
valor alto es coherente con lo observado —la deduplicación deja **nodos compactos y bien
separados** entre sí, rodeados de un mar de ruido—; respalda que la estructura encontrada
es genuina. *Implicación:* podemos confiar en los cúmulos como unidades de análisis.
"""
)

md(
    r"""
### 8.2 Validación externa: % REAL por corredor

El target `REAL` **no** entró en DBSCAN. Comparamos el % afirmativo de cada corredor del
top contra el **63.5% global**: si la vialidad concreta lleva señal sobre la veracidad
del reporte, veremos una **banda amplia** de tasas, no todas pegadas al global.
"""
)

code(
    r"""
car = caracter.sort_values("pct_real").reset_index(drop=True)
# Etiqueta única por barra (vialidad + cluster) para no colapsar nombres repetidos.
car["etiqueta"] = car["vialidad"] + "  [cl " + car["cluster"].astype(str) + "]"
colores = ["#1f77b4" if v >= GLOB_REAL else "#d62728" for v in car["pct_real"]]

fig, ax = plt.subplots(figsize=(9.5, 6))
ax.barh(car["etiqueta"], car["pct_real"], color=colores, edgecolor="white")
ax.axvline(GLOB_REAL, color="black", ls="--", label=f"global = {GLOB_REAL:.1f}%")
for i, (v, n) in enumerate(zip(car["pct_real"], car["n"])):
    ax.text(v + 0.3, i, f"{v:.1f}% (n={n:,})", va="center", fontsize=7.5)
ax.set_xlabel("% reportes afirmativos (REAL = 1)")
ax.set_title("Validacion externa: % afirmativo por corredor (DBSCAN no uso el target)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Banda sobre TODOS los clusters con masa (n>=200), no solo el top-12.
g_all = (df[df["cluster"] != -1].groupby("cluster")["REAL"]
         .agg(media="mean", n="count"))
g_all["media"] *= 100
band = g_all[g_all["n"] >= 200]
pct_cluster = df.loc[df["cluster"] != -1, "REAL"].mean() * 100
pct_ruido = df.loc[df["cluster"] == -1, "REAL"].mean() * 100
print(f"% afirmativo  EN clusters densos = {pct_cluster:.1f}%  vs ruido = {pct_ruido:.1f}%"
      f"  vs global = {GLOB_REAL:.1f}%")
print(f"Banda entre los {len(band)} clusters con n>=200: "
      f"{band['media'].min():.1f}% a {band['media'].max():.1f}% (mediana {band['media'].median():.1f}%).")
print(f"  por encima del global: {(band['media'] >= GLOB_REAL).sum()} | "
      f"por debajo: {(band['media'] < GLOB_REAL).sum()}")
"""
)

md(
    r"""
**Interpretación.** Aparecen **dos hallazgos**. Primero, el % afirmativo se abre en una
**banda amplia** entre los clusters con masa —de ~41 % a ~90 %—: la **vialidad concreta**
sí lleva señal sobre la veracidad del reporte. Segundo, los **corredores más grandes
confirman *por debajo* del global** (en clusters densos ~59 % vs ~65 % en el ruido y 63.5 %
global): las vialidades de altísimo flujo acumulan **proporcionalmente más reportes
falsos/informativos** (saturación, llamadas repetidas, sobre-reporte en hora pico). Como
DBSCAN **nunca vio el target**, esta separación es **validación externa**. *Implicación:*
refuerza que la geografía es una variable valiosa para el clasificador del notebook 2 —y
que un reporte sobre una vialidad saturada merece **más** verificación, no menos.
"""
)

# --------------------------------------------------------------------------- #
# 9. Cierre
# --------------------------------------------------------------------------- #
md(
    r"""
## 9. Cierre: corredores estrella, aportes y limitaciones

A continuación imprimimos los **3 corredores estrella** (los más grandes) con su nombre,
tamaño, tipo dominante y % confirmado, listos para la presentación.
"""
)

code(
    r"""
print("=== Vialidades estrella (top 3 por tamano) ===\n")
for _, r in caracter.head(3).iterrows():
    print(f"- {r['vialidad']} ({r['alcaldia_dom']}) [{r['clase']}]")
    print(f"    n = {r['n']:,} incidentes | {r['pct_real']:.1f}% afirmativos "
          f"(global {GLOB_REAL:.1f}%)")
    print(f"    tipo dominante: {r['incidente_dom']} ({r['pct_inc_dom']:.1f}%) "
          f"| franja pico: {r['franja_pico']}\n")
"""
)

md(
    r"""
**Aportes.**

- **Nodos viales nombrados y accionables.** DBSCAN (`eps`=150 m, `min_samples`=200)
  identificó ~170 cúmulos densos; entre los más grandes, puntos reconocibles sobre
  **Circuito Interior**, **Avenida Río Churubusco**, **Calzada Ignacio Zaragoza** y el
  **Distribuidor Vial San Antonio**, más cruceros en Venustiano Carranza, GAM e
  Iztapalapa. Saber *qué vialidad* concentra incidentes (y de qué tipo) es insumo directo
  para decidir **dónde patrullar o dónde verificar reportes**.
- **El ruido como información.** Que ~77 % de los incidentes sea ruido dice que la mayor
  parte de la ciudad sufre incidentes *dispersos*: la estrategia ahí no es por corredor
  sino de cobertura general.
- **Tres entregables visuales:** el PNG estático `figures/figura_dbscan_vialidades.png`,
  el **mapa interactivo INLINE** (Plotly + OpenStreetMap, con hover) y su respaldo
  standalone `figures/mapa_corredores.html` para abrir en el navegador.
- **Validación externa con giro útil:** el % afirmativo varía fuerte entre vialidades
  (~41 %–90 %) y, además, las **vialidades más saturadas confirman por debajo del global**
  — señal de que ahí conviene verificar **más**, no menos.

**Limitaciones.**

- **DBSCAN es sensible a la densidad variable.** Con un único `eps`/`min_samples`, las
  zonas céntricas (muy densas) se fragmentan o se funden distinto que la periferia; un
  DBSCAN jerárquico (HDBSCAN) podría adaptarse mejor, pero añade complejidad.
- **Ruido alto y pocos corredores lineales.** El ~77 % de ruido es una elección
  deliberada (nodos con masa crítica); tras quitar duplicados quedan sobre todo
  cruceros/zonas y pocos corredores lineales. Bajar `min_samples` recuperaría cúmulos
  pequeños a costa de muchos micro-clusters poco accionables.
- **Nombres heredados de otra corrida.** El JSON geocodificado se calculó sobre una
  corrida previa de DBSCAN; aplicamos un umbral de 1,000 m y caemos a etiqueta genérica
  cuando el cruce es lejano, pero un re-geocodificado sobre esta corrida daría nombres más
  precisos.
- **Geografía aproximada.** La proyección equirectangular local es válida a escala de
  ciudad, pero introduce una distorsión menor en los bordes; para este análisis es
  despreciable frente a la incertidumbre de las propias coordenadas reportadas.
"""
)

# --------------------------------------------------------------------------- #
# Ensamble
# --------------------------------------------------------------------------- #
nb["cells"] = celdas
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
SALIDA.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, SALIDA)
print(f"Notebook escrito: {SALIDA}  ({len(celdas)} celdas)")
