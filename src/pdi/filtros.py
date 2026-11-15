"""
Filtrado espacial. La convolucion se evalua explicitamente como suma
ponderada de vecindades, desplazando el arreglo termino a termino.
"""
import numpy as np


def rellenar(img, r, modo="replicar"):
    """Extiende la imagen r pixeles en cada borde."""
    if modo == "replicar":
        return np.pad(img, ((r, r), (r, r)), mode="edge")
    return np.pad(img, ((r, r), (r, r)), mode="constant", constant_values=0.0)


def convolucionar(img, nucleo):
    """
    Convolucion discreta bidimensional:

        g(x, y) = sum_{u=-a}^{a} sum_{v=-b}^{b} w(u, v) * f(x - u, y - v)

    Se implementa acumulando versiones desplazadas de la imagen
    rellenada, es decir, evaluando directamente la suma anterior.
    """
    nucleo = np.asarray(nucleo, dtype=np.float64)
    kh, kw = nucleo.shape
    a, b = kh // 2, kw // 2
    r = max(a, b)
    pad = rellenar(img, r)
    M, N = img.shape
    out = np.zeros((M, N), dtype=np.float64)
    for u in range(kh):
        for v in range(kw):
            w = nucleo[u, v]
            if w == 0.0:
                continue
            # f(x - u, y - v) con el signo de la convolucion
            du = r - (u - a)
            dv = r - (v - b)
            out += w * pad[du:du + M, dv:dv + N]
    return out


def nucleo_gaussiano(sigma, tam=None):
    """
    Nucleo gaussiano bidimensional separable, normalizado:

        G(u, v) = (1 / (2*pi*sigma^2)) * exp( -(u^2 + v^2) / (2*sigma^2) )
    """
    if tam is None:
        tam = int(2 * np.ceil(3 * sigma) + 1)
    if tam % 2 == 0:
        tam += 1
    c = tam // 2
    u = np.arange(tam) - c
    U, V = np.meshgrid(u, u, indexing="ij")
    g = np.exp(-(U ** 2 + V ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def suavizado_gaussiano(img, sigma=1.5):
    """
    Suavizado gaussiano aprovechando la separabilidad del nucleo:

        G(u, v) = g(u) * g(v)

    lo que reduce el costo de O(k^2) a O(2k) por pixel.
    """
    tam = int(2 * np.ceil(3 * sigma) + 1)
    c = tam // 2
    u = np.arange(tam) - c
    g = np.exp(-(u ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    tmp = convolucionar(img, g.reshape(1, -1))
    return convolucionar(tmp, g.reshape(-1, 1))


# Operadores diferenciales clasicos
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float64)

LAPLACIANO = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float64)


def gradiente(img):
    """
    Magnitud del gradiente aproximada con los operadores de Sobel:

        |grad f| = sqrt( (df/dx)^2 + (df/dy)^2 )
    """
    gx = convolucionar(img, SOBEL_X)
    gy = convolucionar(img, SOBEL_Y)
    return np.sqrt(gx ** 2 + gy ** 2), gx, gy


def laplaciano(img):
    """
    Laplaciano discreto:

        lap f = d2f/dx2 + d2f/dy2
    """
    return convolucionar(img, LAPLACIANO)
