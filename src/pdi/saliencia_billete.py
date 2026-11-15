"""
Mapas de saliencia sobre el billete segmentado.

Distincion respecto de los mapas de segmentacion.mapas_de_saliencia
-------------------------------------------------------------------
Aquellos responden a la pregunta *donde esta el billete* y se calculan
sobre la fotografia completa, comparando cada pixel contra un modelo
del fondo de la escena.

Los de este modulo responden a otra pregunta: *que zonas del billete
diferencian un ejemplar autentico de una reproduccion*. Se calculan
DESPUES de la segmentacion, sobre el billete ya rectificado y con el
fondo retirado, de modo que ninguna medida puede contaminarse con la
superficie sobre la que se tomo la fotografia.

El orden importa y no es intercambiable:

    fotografia -> mascara binaria -> billete sin fondo -> ESTOS mapas

Calcularlos antes de retirar el fondo produciria valores altos en la
mesa, en las sombras y en los bordes de la escena, que nada tienen que
ver con la impresion del billete.

Estrategia de calculo
---------------------
Las tres magnitudes son estadisticos locales, de modo que se evaluan
por bloques: la imagen se divide en una retícula y en cada celda se
calcula el estadistico sobre los pixeles validos de esa celda. El
resultado es un mapa de baja resolucion, que es lo adecuado aqui
porque interesa la distribucion espacial gruesa y no el detalle pixel
a pixel.
"""
import numpy as np

from . import color as C
from . import filtros as F


def _por_bloques(campo, mascara, lado):
    """
    Promedia un campo escalar por bloques, considerando solo los
    pixeles validos de cada bloque:

        g(i, j) = ( sum_{x in B_ij} m(x) f(x) ) / ( sum_{x in B_ij} m(x) )

    Las celdas sin pixeles validos quedan como no definidas.
    """
    M, N = campo.shape
    Mi, Ni = M // lado, N // lado
    if Mi == 0 or Ni == 0:
        return np.full((1, 1), np.nan)

    f = campo[:Mi * lado, :Ni * lado].reshape(Mi, lado, Ni, lado)
    m = mascara[:Mi * lado, :Ni * lado].reshape(Mi, lado, Ni, lado)

    suma = (f * m).sum(axis=(1, 3))
    cuenta = m.sum(axis=(1, 3))
    salida = np.full((Mi, Ni), np.nan)
    ok = cuenta > (lado * lado * 0.5)
    salida[ok] = suma[ok] / cuenta[ok]
    return salida


def _normalizar(mapa):
    """Lleva a [0, 1] usando los percentiles 2 y 98 de las celdas validas."""
    v = mapa[np.isfinite(mapa)]
    if v.size < 4:
        return mapa
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    if hi - lo < 1e-12:
        return np.clip(mapa - lo, 0, 1)
    return np.clip((mapa - lo) / (hi - lo), 0.0, 1.0)


def mapa_microtextura(gris, mascara, lado=32):
    """
    Energia local del laplaciano.

        V_L(i,j) = varianza de  lap I  dentro del bloque (i,j)

    Es la version espacialmente resuelta del descriptor con que se
    decide la autenticidad. Muestra en que zonas del billete se
    concentra el detalle fino: en un ejemplar autentico, sobre las
    tramas de guilloche y el microtexto; en una reproduccion, en
    ninguna.
    """
    lap = F.laplaciano(gris)
    media = _por_bloques(lap, mascara, lado)
    media2 = _por_bloques(lap ** 2, mascara, lado)
    return np.maximum(media2 - media ** 2, 0.0)


def mapa_bordes(gris, mascara, lado=32):
    """
    Magnitud media del gradiente por bloque. Responde al contraste de
    los trazos impresos, es decir a la definicion de la linea, mientras
    que el laplaciano responde al detalle mas fino.
    """
    mag, _, _ = F.gradiente(gris)
    return _por_bloques(mag, mascara, lado)


def mapa_saturacion(rect_rgb, mascara, lado=32):
    """
    Saturacion media por bloque. Una reproduccion pierde saturacion
    porque la gama de la impresora de consumo es menor que la de las
    tintas de seguridad, y porque el sustrato dispersa mas la luz.
    """
    _, S, _ = C.rgb_a_hsv(rect_rgb)
    return _por_bloques(S, mascara, lado)


def mapas_del_billete(rect_rgb, mascara, lado=32):
    """
    Calcula las tres evidencias sobre el billete sin fondo y las
    combina.

    La combinacion es el promedio de las tres normalizadas,

        s(i,j) = w1 * t(i,j) + w2 * b(i,j) + w3 * c(i,j)

    con pesos que reflejan la capacidad discriminante medida: la
    microtextura es la evidencia que mejor separa las clases, seguida
    de los bordes y de la saturacion. Un valor alto de s indica una
    zona con caracteristicas propias de impresion de seguridad; uno
    bajo, una zona compatible con una reproduccion.

    Devuelve un diccionario con los mapas crudos, los normalizados y
    el combinado, mas el promedio de cada uno sobre el billete.
    """
    gris = C.rgb_a_gris(rect_rgb) / 255.0

    t = mapa_microtextura(gris, mascara, lado)
    b = mapa_bordes(gris, mascara, lado)
    c = mapa_saturacion(rect_rgb, mascara, lado)

    tn, bn, cn = _normalizar(t), _normalizar(b), _normalizar(c)
    combinada = np.nansum(
        np.dstack([0.50 * tn, 0.30 * bn, 0.20 * cn]), axis=2)
    combinada[~np.isfinite(tn)] = np.nan

    def _media(x):
        v = x[np.isfinite(x)]
        return float(v.mean()) if v.size else float("nan")

    return {
        "microtextura": t, "bordes": b, "saturacion": c,
        "microtextura_n": tn, "bordes_n": bn, "saturacion_n": cn,
        "combinada": combinada, "lado_bloque": lado,
        "promedios": {"microtextura": _media(t), "bordes": _media(b),
                      "saturacion": _media(c),
                      "combinada": _media(combinada)},
    }


def comparar(mapas_a, mapas_b):
    """
    Cociente celda a celda entre dos conjuntos de mapas del mismo
    tamano, util para contrastar un ejemplar autentico con una
    reproduccion de la misma denominacion:

        r(i,j) = s_b(i,j) / s_a(i,j)

    Valores muy por debajo de 1 senalan las zonas en que la
    reproduccion pierde caracteristicas respecto del original.
    """
    salida = {}
    for clave in ("microtextura", "bordes", "saturacion"):
        A, B = mapas_a[clave], mapas_b[clave]
        if A.shape != B.shape:
            n = (min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1]))
            A, B = A[:n[0], :n[1]], B[:n[0], :n[1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            salida[clave] = np.where(np.abs(A) > 1e-12, B / A, np.nan)
    return salida
