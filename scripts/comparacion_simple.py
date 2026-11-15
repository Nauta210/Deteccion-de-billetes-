"""
Metodo simple de comparacion, previo a cualquier clasificador.

Antes de entrenar un modelo conviene verificar si las clases se
separan con un estadistico elemental. Este script calcula, sobre el
billete YA SEGMENTADO Y SIN FONDO, un conjunto de medidas de primer y
segundo orden, y compara su distribucion entre billetes autenticos y
reproducciones mediante diagramas de caja y bigotes.

El razonamiento es el siguiente. Si una medida tan simple como la
media de intensidad ya separa las clases, no hace falta un descriptor
de textura; y si no las separa, el diagrama muestra cuanto se solapan
y por que se necesita algo mas fino. En ambos casos se obtiene
informacion que un numero de exactitud aislado no da.

Todas las medidas se calculan unicamente sobre los pixeles del
soporte del billete. El fondo se excluye explicitamente: de otro modo
la comparacion mediria la mesa sobre la que se tomo la foto y no el
billete.

Uso:
    python scripts/comparacion_simple.py
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))
sys.path.insert(0, os.path.join(RAIZ, "src"))

from pdi import io_utils as IO          # noqa: E402
from pdi import color as C              # noqa: E402
from pdi import segmentacion as SEG     # noqa: E402
from pdi import filtros as F            # noqa: E402

MEDIDAS = [
    ("media", "Media de intensidad"),
    ("desviacion", "Desviacion estandar"),
    ("asimetria", "Asimetria del histograma"),
    ("energia_borde", "Energia de bordes"),
    ("nitidez", "Varianza del laplaciano"),
    ("saturacion", "Saturacion media"),
]


def medidas_de(img_rgb):
    """
    Estadisticos elementales sobre el billete sin fondo.

        media       mu   = (1/n) sum_i x_i
        desviacion  sd   = sqrt( (1/n) sum_i (x_i - mu)^2 )
        asimetria   g1   = (1/n) sum_i ((x_i - mu)/sd)^3
        energia     E    = media de |grad I| sobre el soporte
        nitidez     V_L  = varianza del laplaciano sobre el soporte
        saturacion  S    = media del canal de saturacion

    La asimetria indica si la distribucion de intensidades se inclina
    hacia los tonos claros o los oscuros, que es una de las
    diferencias esperables entre una impresion de seguridad y una
    reproduccion de consumo.
    """
    et = SEG.procesar(img_rgb)
    rect = et["rectificada"]
    m = et["mascara_rect"]
    if m.sum() < 1000:
        return None, et

    g = C.rgb_a_gris(rect) / 255.0
    x = g[m]
    mu = float(x.mean())
    sd = float(x.std())
    g1 = float(np.mean(((x - mu) / max(sd, 1e-9)) ** 3))

    mag, _, _ = F.gradiente(g)
    lap = F.laplaciano(g)
    _, S, _ = C.rgb_a_hsv(rect)

    return {
        "media": mu,
        "desviacion": sd,
        "asimetria": g1,
        "energia_borde": float(mag[m].mean()),
        "nitidez": float(np.var(lap[m])),
        "saturacion": float(S[m].mean()),
    }, et


def cohen_d(a, b):
    """
    Tamano del efecto de Cohen, que expresa la separacion entre dos
    grupos en unidades de desviacion estandar combinada:

        d = (mu_1 - mu_2) / s_p,
        s_p = sqrt( ((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2) )

    Un valor cercano a cero indica solapamiento total; por encima de
    0.8 se considera una separacion grande. A diferencia de una prueba
    de significancia, no depende del tamano muestral, lo que importa
    aqui porque las muestras son pequenas.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp = np.sqrt((((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1))
                  / (n1 + n2 - 2)))
    return float((a.mean() - b.mean()) / max(sp, 1e-12))


def recolectar(carpeta, etiqueta):
    filas = []
    for r in IO.listar_imagenes(carpeta):
        try:
            m, _ = medidas_de(IO.cargar_rgb(r))
        except Exception as e:
            print(f"   fallo {os.path.basename(r)}: {e}")
            continue
        if m is None:
            continue
        m["clase"] = os.path.basename(os.path.dirname(r))
        m["etiqueta"] = etiqueta
        m["archivo"] = os.path.basename(r)
        filas.append(m)
        print(f"   {etiqueta:12s} {m['clase']:7s} "
              f"{os.path.basename(r)[:28]:30s} media={m['media']:.3f}")
    return filas


def main():
    print("Billetes autenticos")
    aut = recolectar(os.path.join(RAIZ, "datos", "autentico"), "autentico")
    print("\nReproducciones")
    rep = recolectar(os.path.join(RAIZ, "datos", "reproduccion"),
                     "reproduccion")

    if not aut or not rep:
        print("\nHacen falta imagenes de ambas clases.")
        return

    # Solo las denominaciones presentes en ambas clases, para que la
    # comparacion no mezcle billetes de colores distintos.
    comunes = ({f["clase"] for f in aut} & {f["clase"] for f in rep})
    print(f"\nDenominaciones comparables: {sorted(comunes)}")
    A = [f for f in aut if f["clase"] in comunes]
    R = [f for f in rep if f["clase"] in comunes]

    resumen = {}
    print(f"\n{'Medida':<26}{'Autentico':>22}{'Reproduccion':>22}"
          f"{'Cohen d':>10}")
    print("-" * 80)
    for clave, nombre in MEDIDAS:
        a = [f[clave] for f in A]
        b = [f[clave] for f in R]
        d = cohen_d(a, b)
        resumen[clave] = {
            "autentico_media": float(np.mean(a)),
            "autentico_sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "reproduccion_media": float(np.mean(b)),
            "reproduccion_sd": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
            "cohen_d": d, "n_aut": len(a), "n_rep": len(b),
        }
        print(f"{nombre:<26}{np.mean(a):>12.4f} +-{np.std(a,ddof=1) if len(a)>1 else 0:>7.4f}"
              f"{np.mean(b):>12.4f} +-{np.std(b,ddof=1) if len(b)>1 else 0:>7.4f}"
              f"{d:>10.2f}")

    # Diagramas de caja y bigotes
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for k, (clave, nombre) in enumerate(MEDIDAS):
        a = ax.ravel()[k]
        datos = [[f[clave] for f in A], [f[clave] for f in R]]
        bp = a.boxplot(datos, labels=["Auténtico", "Reproducción"],
                       patch_artist=True, widths=0.55)
        for parche, col in zip(bp["boxes"], ["#5b8def", "#ef6f5b"]):
            parche.set_facecolor(col)
            parche.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.6)
        # Puntos individuales, indispensables con muestras pequenas
        for i, serie in enumerate(datos, start=1):
            x = np.random.default_rng(k).normal(i, 0.045, len(serie))
            a.plot(x, serie, "o", ms=3.2, color="#222", alpha=0.55)
        a.set_title(f"{nombre}\n$d$ = {resumen[clave]['cohen_d']:.2f}",
                    fontsize=10)
        a.grid(alpha=0.25, axis="y")

    fig.suptitle("Comparacion elemental sobre el billete segmentado sin fondo",
                 fontsize=12)
    fig.tight_layout()
    destino = os.path.join(RAIZ, "resultados", "figuras")
    os.makedirs(destino, exist_ok=True)
    fig.savefig(os.path.join(destino, "cajas_bigotes.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(RAIZ, "resultados", "comparacion_simple.json"),
              "w") as f:
        json.dump({"resumen": resumen,
                   "n_autentico": len(A), "n_reproduccion": len(R),
                   "denominaciones": sorted(comunes)}, f, indent=2)

    print(f"\nFigura en resultados/figuras/cajas_bigotes.png")
    print("Resumen en resultados/comparacion_simple.json")


if __name__ == "__main__":
    main()
