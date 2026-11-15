"""
Umbralizacion automatica y morfologia matematica binaria.
Ambas se derivan de su definicion, sin llamar a rutinas externas.
"""
import numpy as np


def histograma(img, niveles=256, rango=(0.0, 1.0)):
    lo, hi = rango
    q = np.clip(((img - lo) / (hi - lo) * (niveles - 1)).round(), 0, niveles - 1)
    h = np.bincount(q.astype(int).ravel(), minlength=niveles).astype(np.float64)
    return h / h.sum()


def otsu(img, niveles=256, rango=(0.0, 1.0)):
    """
    Umbral optimo de Otsu.

    Para un umbral T que separa las clases C0 y C1 se definen

        w0(T) = sum_{i<=T} p(i)          w1(T) = 1 - w0(T)
        mu0(T) = (1/w0) sum_{i<=T} i p(i)
        mu1(T) = (1/w1) sum_{i>T}  i p(i)

    y la varianza entre clases

        sigma_B^2(T) = w0(T) w1(T) [mu0(T) - mu1(T)]^2

    El umbral elegido maximiza sigma_B^2. Maximizar la varianza entre
    clases equivale a minimizar la varianza intraclase, porque la
    varianza total es constante:

        sigma_T^2 = sigma_W^2(T) + sigma_B^2(T)
    """
    p = histograma(img, niveles, rango)
    i = np.arange(niveles, dtype=np.float64)

    w0 = np.cumsum(p)
    mu = np.cumsum(i * p)
    mu_T = mu[-1]

    w1 = 1.0 - w0
    valido = (w0 > 1e-12) & (w1 > 1e-12)

    sigma_b = np.zeros(niveles)
    sigma_b[valido] = ((mu_T * w0[valido] - mu[valido]) ** 2
                       / (w0[valido] * w1[valido]))

    k = int(np.argmax(sigma_b))
    lo, hi = rango
    T = lo + (k / (niveles - 1.0)) * (hi - lo)
    return T, sigma_b[k]


def binarizar(img, T):
    return (img > T)


# --------------------------------------------------------------------
# Morfologia matematica binaria.
#
#   Dilatacion:  A (+) B = { z : (B_z) ∩ A ≠ vacio }
#   Erosion:     A (-) B = { z : (B_z) ⊆ A }
#   Apertura:    A o B = (A (-) B) (+) B
#   Cierre:      A • B = (A (+) B) (-) B
#
# Se implementan desplazando la imagen binaria segun cada elemento
# activo del elemento estructurante B, lo que corresponde
# literalmente a la union (dilatacion) o interseccion (erosion) de
# las traslaciones.
# --------------------------------------------------------------------
def elemento_rect(alto, ancho):
    return np.ones((alto, ancho), dtype=bool)


def elemento_elipse(radio):
    d = 2 * radio + 1
    u = np.arange(d) - radio
    U, V = np.meshgrid(u, u, indexing="ij")
    return (U ** 2 + V ** 2) <= radio ** 2


def _desplazar(B, du, dv, relleno):
    out = np.full_like(B, relleno)
    M, N = B.shape
    i0, i1 = max(0, du), min(M, M + du)
    j0, j1 = max(0, dv), min(N, N + dv)
    if i0 >= i1 or j0 >= j1:
        return out
    out[i0:i1, j0:j1] = B[i0 - du:i1 - du, j0 - dv:j1 - dv]
    return out


def dilatar(B, ee):
    kh, kw = ee.shape
    a, b = kh // 2, kw // 2
    out = np.zeros_like(B, dtype=bool)
    for u in range(kh):
        for v in range(kw):
            if ee[u, v]:
                out |= _desplazar(B, u - a, v - b, False)
    return out


def erosionar(B, ee):
    kh, kw = ee.shape
    a, b = kh // 2, kw // 2
    out = np.ones_like(B, dtype=bool)
    for u in range(kh):
        for v in range(kw):
            if ee[u, v]:
                out &= _desplazar(B, u - a, v - b, False)
    return out


def abrir(B, ee):
    return dilatar(erosionar(B, ee), ee)


def cerrar(B, ee):
    return erosionar(dilatar(B, ee), ee)


def dilatacion_geodesica(marcador, mascara, ee=None):
    """
    Dilatacion geodesica elemental de tamano uno:

        delta^{(1)}_X(M) = (M (+) B) ∩ X

    El marcador M se dilata con el elemento estructurante B y el
    resultado se restringe a la mascara X, llamada region geodesica.
    La restriccion impide que la propagacion atraviese los limites de
    X, que es lo que distingue esta operacion de una dilatacion
    ordinaria.

    La dilatacion geodesica de tamano n es la composicion n veces:

        delta^{(n)}_X(M) = delta^{(1)}_X( delta^{(n-1)}_X(M) )
    """
    if ee is None:
        ee = np.array([[False, True, False],
                       [True, True, True],
                       [False, True, False]])
    return dilatar(marcador, ee) & mascara


def reconstruccion_morfologica(marcador, mascara, ee=None, max_iter=10000):
    """
    Reconstruccion morfologica por dilatacion.

        R_X(M) = delta^{(n)}_X(M),  con n tal que
                 delta^{(n)}_X(M) = delta^{(n+1)}_X(M)

    Se itera la dilatacion geodesica hasta alcanzar la idempotencia,
    es decir hasta que la propagacion no gane ningun pixel nuevo. El
    resultado son exactamente las componentes conexas de la mascara X
    que contienen al menos un pixel del marcador M.

    La convergencia esta garantizada porque la sucesion es creciente y
    esta acotada superiormente por X, que es finita.
    """
    M = np.asarray(marcador, dtype=bool) & np.asarray(mascara, dtype=bool)
    for _ in range(max_iter):
        nuevo = dilatacion_geodesica(M, mascara, ee)
        if np.array_equal(nuevo, M):
            return M
        M = nuevo
    return M


def rellenar_huecos(B):
    """
    Relleno de huecos por reconstruccion desde el borde.

    Se propaga el fondo conectado al marco de la imagen mediante
    dilataciones condicionadas al complemento de B:

        X_{k+1} = (X_k (+) B_4) ∩ A^c ,   X_0 = marco ∩ A^c

    Los pixeles de fondo no alcanzados son huecos interiores.
    """
    comp = ~B
    marcador = np.zeros_like(B, dtype=bool)
    marcador[0, :] = comp[0, :]
    marcador[-1, :] = comp[-1, :]
    marcador[:, 0] = comp[:, 0]
    marcador[:, -1] = comp[:, -1]

    # El fondo exterior es la reconstruccion morfologica del marco
    # dentro del complemento del objeto. Lo que queda del complemento
    # sin alcanzar son los huecos interiores.
    marcador = reconstruccion_morfologica(marcador, comp)
    return B | (~marcador & comp)
