"""
Recorte automatico de capturas.

Motivacion
----------
El modelo de fondo de la segmentacion estima media y covarianza sobre
la franja perimetral del cuadro, bajo el supuesto de que esa franja
contiene unicamente fondo homogeneo. Cuando la fotografia incluye
varios materiales distintos ---un tapete oscuro, un mueble, otros
objetos--- la franja mezcla poblaciones y la covarianza estimada se
dispara, de modo que ningun pixel resulta anomalo y la segmentacion
falla.

Este script acota el cuadro a la zona util antes de procesar. No es un
recorte manual: se determina automaticamente en dos pasos.

  1. Recorte grueso. Se busca la banda horizontal del cuadro cuya
     cromaticidad sea mas homogenea, es decir, la superficie de apoyo
     sobre la que descansa el billete. Para cada fila i se calcula la
     traza de la covarianza cromatica de esa fila,

         v(i) = tr( Cov{ c(i, j) : j = 1..N } )

     y se conserva el intervalo contiguo de filas de menor v que cubra
     al menos una fraccion minima del alto.

  2. Recorte fino. Sobre la banda anterior se ejecuta la segmentacion
     y, si tiene exito, se recorta alrededor del billete dejando un
     margen proporcional a su tamano. El margen es necesario: sin
     fondo alrededor el modelo perimetral no tendria nada que estimar.

Uso:
    python scripts/recortar_capturas.py <entrada> <salida> [margen]
"""
import os
import sys
import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "src"))

from pdi import io_utils as IO          # noqa: E402
from pdi import segmentacion as SEG     # noqa: E402

FRACCION_MIN = 0.40        # la banda util cubre al menos este alto
MARGEN = 0.22              # margen alrededor del billete, en fracciones


def banda_homogenea(img_rgb, fraccion_min=FRACCION_MIN):
    """
    Devuelve (fila_inicial, fila_final) de la banda horizontal cuya
    cromaticidad es mas homogenea.

    Se recorre con una ventana deslizante de alto fijo y se conserva la
    posicion que minimiza la traza de la covarianza cromatica
    acumulada, que es una medida escalar de la dispersion total de
    color dentro de la ventana.
    """
    crom = SEG.cromaticidad(img_rgb)
    M = crom.shape[0]
    alto = max(8, int(round(fraccion_min * M)))
    if alto >= M:
        return 0, M

    # Sumas acumuladas para evaluar la covarianza de cada ventana en
    # tiempo constante.
    X = crom.reshape(M, -1, 2)
    s1 = np.cumsum(np.vstack([np.zeros((1, 2)), X.sum(axis=1)]), axis=0)
    s2 = np.cumsum(np.vstack([np.zeros((1, 2)), (X ** 2).sum(axis=1)]), axis=0)
    n_fila = X.shape[1]

    mejor, mejor_v = 0, np.inf
    for i in range(0, M - alto + 1, max(1, alto // 16)):
        n = n_fila * alto
        suma = s1[i + alto] - s1[i]
        suma2 = s2[i + alto] - s2[i]
        var = suma2 / n - (suma / n) ** 2
        v = float(np.sum(np.maximum(var, 0.0)))
        if v < mejor_v:
            mejor_v, mejor = v, i
    return mejor, mejor + alto


def recortar(img_rgb, margen=MARGEN):
    """
    Devuelve la imagen recortada y un diccionario de diagnostico.
    """
    i0, i1 = banda_homogenea(img_rgb)
    banda = img_rgb[i0:i1]
    diag = {"banda": (int(i0), int(i1)), "fino": False}

    try:
        et = SEG.procesar(banda)
    except Exception:
        return banda, diag

    if et["diagnostico"]["puntaje_forma"] < 0.45:
        return banda, diag

    ys, xs = np.nonzero(et["componente"])
    h, w = banda.shape[:2]
    mh = int(round(margen * (ys.max() - ys.min() + 1)))
    mw = int(round(margen * (xs.max() - xs.min() + 1)))

    y0 = max(0, ys.min() - mh)
    y1 = min(h, ys.max() + 1 + mh)
    x0 = max(0, xs.min() - mw)
    x1 = min(w, xs.max() + 1 + mw)

    diag.update({"fino": True,
                 "puntaje": round(et["diagnostico"]["puntaje_forma"], 3),
                 "caja": (int(x0), int(y0), int(x1), int(y1))})
    return banda[y0:y1, x0:x1], diag


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    entrada, salida = sys.argv[1], sys.argv[2]
    margen = float(sys.argv[3]) if len(sys.argv) > 3 else MARGEN

    rutas = IO.listar_imagenes(entrada)
    print(f"Recortando {len(rutas)} imagenes (margen {margen:.0%})\n")
    n_fino = 0

    for k, r in enumerate(rutas):
        img = IO.cargar_rgb(r)   # sin reducir: conservar todo el detalle
        rec, diag = recortar(img, margen)

        rel = os.path.relpath(os.path.dirname(r), entrada)
        destino = os.path.join(salida, rel)
        os.makedirs(destino, exist_ok=True)
        base = os.path.splitext(os.path.basename(r))[0]
        Image.fromarray(np.clip(rec, 0, 255).astype(np.uint8)).save(
            os.path.join(destino, base + ".jpg"), quality=95, subsampling=0)

        n_fino += int(diag["fino"])
        marca = "fino" if diag["fino"] else "solo banda"
        print(f"  [{k+1:2d}/{len(rutas)}] {base[:34]:36s} "
              f"{img.shape[1]}x{img.shape[0]} -> "
              f"{rec.shape[1]}x{rec.shape[0]}  ({marca})")

    print(f"\nRecorte fino logrado en {n_fino}/{len(rutas)} imagenes.")
    print(f"Salida en {salida}/")


if __name__ == "__main__":
    main()
