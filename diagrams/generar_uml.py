"""Dibuja el diagrama UML de clases de src/ con matplotlib (sin dependencias
externas de render) y lo guarda en diagrams/uml_clases.png.

Uso: python diagrams/generar_uml.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

AQUI = Path(__file__).resolve().parent

# Paleta por módulo
C_PREP = "#e3f0fb"   # preparacion (azul claro)
C_MOD = "#e7f6e7"    # modelado (verde claro)
C_CLU = "#fdf0e0"    # clustering (naranja claro)
C_ABS = "#f3e8fb"    # clases abstractas (lila)


def clase(ax, x, y, w, titulo, attrs, methods, color, abstracta=False):
    """Dibuja una caja de clase UML (nombre / atributos / metodos).
    (x, y) es la esquina superior izquierda. Devuelve (cx_top, y, w, alto)."""
    lh = 0.32  # alto por linea
    head = 0.62 if abstracta else 0.46
    h = head + lh * (len(attrs) + len(methods)) + 0.2
    y0 = y - h
    ax.add_patch(Rectangle((x, y0), w, h, facecolor=color,
                           edgecolor="#333", linewidth=1.3, zorder=2))
    # nombre
    ty = y - 0.30
    if abstracta:
        ax.text(x + w / 2, y - 0.22, "«abstract»", ha="center", va="top",
                fontsize=7.5, style="italic", color="#555", zorder=3)
        ty = y - 0.46
    ax.text(x + w / 2, ty, titulo, ha="center", va="top",
            fontsize=9.5, fontweight="bold", family="monospace", zorder=3)
    yline = y - head
    ax.plot([x, x + w], [yline, yline], color="#333", lw=0.9, zorder=3)
    yy = yline - 0.04
    for a in attrs:
        ax.text(x + 0.08, yy - lh + 0.07, a, ha="left", va="bottom",
                fontsize=7.6, family="monospace", zorder=3)
        yy -= lh
    ax.plot([x, x + w], [yy, yy], color="#333", lw=0.9, zorder=3)
    yy -= 0.04
    for m in methods:
        ax.text(x + 0.08, yy - lh + 0.07, m, ha="left", va="bottom",
                fontsize=7.6, family="monospace", zorder=3)
        yy -= lh
    return (x, y0, w, h)


def herencia(ax, hijo, padre):
    """Flecha de herencia (triangulo hueco) del hijo al padre."""
    hx = hijo[0] + hijo[2] / 2
    hy = hijo[1] + hijo[3]              # top del hijo
    px = padre[0] + padre[2] / 2
    py = padre[1]                       # bottom del padre
    ax.add_patch(FancyArrowPatch((hx, hy), (px, py),
                 arrowstyle="-|>", mutation_scale=22, lw=1.2,
                 color="#333", shrinkA=2, shrinkB=2,
                 connectionstyle="arc3,rad=0",
                 fc="white", zorder=1))


def composicion(ax, todo, parte):
    """Linea de composicion (rombo lleno) de 'todo' a 'parte'."""
    tx = todo[0] + todo[2]             # derecha del 'todo'
    ty = todo[1] + todo[3] / 2
    px = parte[0]                      # izquierda de la 'parte'
    py = parte[1] + parte[3] / 2
    ax.add_patch(FancyArrowPatch((tx, ty), (px, py),
                 arrowstyle="-", lw=1.4, color="#333", zorder=1))
    ax.plot([(tx + px) / 2], [(ty + py) / 2], marker="D", markersize=9,
            color="#333", zorder=3)
    ax.text((tx + px) / 2, (ty + py) / 2 + 0.18, "compone 1..*",
            ha="center", fontsize=7, style="italic", color="#444", zorder=3)


fig, ax = plt.subplots(figsize=(14.5, 8.8))
ax.set_xlim(0, 15); ax.set_ylim(0, 9.2); ax.axis("off")
ax.text(7.5, 9.05, "Diagrama UML de clases — código fuente (src/)",
        ha="center", fontsize=13, fontweight="bold")

# --- Módulo preparacion ---
pipe = clase(ax, 0.3, 8.5, 3.5, "PipelinePreparacion",
             ["+ pasos: list[PasoLimpieza]", "+ bitacora: list"],
             ["+ aplicar(df, verbose) df"], C_PREP)
paso = clase(ax, 0.5, 6.2, 3.0, "PasoLimpieza",
             ["+ nombre: str"], ["+ aplicar(df) df"], C_ABS, abstracta=True)
qd = clase(ax, -0.05, 3.3, 2.35, "QuitarDuplicados", [], ["+ aplicar(df)"], C_PREP)
qa = clase(ax, 2.45, 3.3, 2.35, "QuitarAnioRuido", [], ["+ aplicar(df)"], C_PREP)
ct = clase(ax, 1.2, 1.5, 2.4, "ConstruirTarget", [], ["+ aplicar(df)"], C_PREP)
composicion(ax, pipe, paso)
for h in (qd, qa, ct):
    herencia(ax, h, paso)

# --- Módulo modelado ---
mc = clase(ax, 6.0, 8.2, 4.3, "ModeloClasificador",
           ["+ numericas: list", "+ categoricas: list"],
           ["+ construir(**p) Pipeline",
            "# _ramas()  [abstracto]",
            "# _crear_estimador(**p) [abstr.]",
            "# _preprocesador() CT"], C_ABS, abstracta=True)
rf = clase(ax, 5.2, 4.6, 3.1, "ModeloRandomForest",
           ["+ imputar_num: bool"],
           ["# _ramas()", "# _crear_estimador(**p)"], C_MOD)
mlp = clase(ax, 8.7, 4.6, 2.9, "ModeloMLP", [],
            ["# _ramas()", "# _crear_estimador(**p)"], C_MOD)
herencia(ax, rf, mc)
herencia(ax, mlp, mc)

# --- Módulo clustering ---
clu = clase(ax, 11.4, 7.6, 3.4, "ClusteringEspacial",
            ["+ lat0, lon0: float", "+ eps: float", "+ min_samples: int"],
            ["+ proyectar(lat, lon)",
             "+ desproyectar(x, y)",
             "+ matriz_coords(df)",
             "+ curva_kdistance(X)",
             "+ ajustar(X) labels"], C_CLU)

# Leyenda
ax.text(11.5, 3.6,
        "Relaciones:\n  ▷  herencia (is-a)\n  ◆—  composición (has-a)\n\n"
        "Módulos (color):\n  azul = src/preparacion\n  verde = src/modelado\n"
        "  naranja = src/clustering\n  lila = clase abstracta",
        fontsize=8, va="top", family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#bbb"))

fig.tight_layout()
out = AQUI / "uml_clases.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("UML guardado en", out)
