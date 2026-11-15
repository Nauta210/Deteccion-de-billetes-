"""
Geometria proyectiva: localizacion de esquinas, estimacion de la
homografia y rectificacion por remapeo inverso.

Este modulo sustituye el uso de rutinas de vision artificial para el
calculo y la aplicacion de transformaciones de perspectiva. La matriz
de homografia se obtiene resolviendo explicitamente el sistema lineal
que plantea el algoritmo DLT, y la imagen rectificada se construye por
mapeo inverso con interpolacion bilineal.
"""
import numpy as np


# --------------------------------------------------------------------
# 1. Localizacion de las cuatro esquinas del cuadrilatero
# --------------------------------------------------------------------
def esquinas_por_proyecciones(mascara):
    """
    Un cuadrilatero convexo aproximadamente rectangular tiene la
    propiedad de que sus vertices son los extremos de las proyecciones
    sobre las dos diagonales del plano. Para cada pixel (x, y) del
    objeto se evaluan las formas lineales

        s(x, y) = x + y        d(x, y) = x - y

    y se toman los pixeles que minimizan y maximizan cada una:

        superior izquierda  ->  min s
        inferior derecha    ->  max s
        superior derecha    ->  max d
        inferior izquierda  ->  min d

    El resultado es exacto para un rectangulo rotado y una buena
    aproximacion bajo perspectiva moderada.

    Devuelve un arreglo (4, 2) en coordenadas (columna, fila), es decir
    (x, y), ordenado como [SI, SD, ID, II].
    """
    ys, xs = np.nonzero(mascara)
    if len(xs) < 4:
        raise ValueError("Mascara vacia o demasiado pequena.")

    s = xs.astype(np.int64) + ys.astype(np.int64)
    d = xs.astype(np.int64) - ys.astype(np.int64)

    si = np.array([xs[np.argmin(s)], ys[np.argmin(s)]], dtype=np.float64)
    id_ = np.array([xs[np.argmax(s)], ys[np.argmax(s)]], dtype=np.float64)
    sd = np.array([xs[np.argmax(d)], ys[np.argmax(d)]], dtype=np.float64)
    ii = np.array([xs[np.argmin(d)], ys[np.argmin(d)]], dtype=np.float64)

    return np.vstack([si, sd, id_, ii])


def ejes_principales(mascara):
    """
    Analisis de componentes principales de la nube de pixeles del
    objeto. Con el centroide

        mu = (1/n) sum_i p_i

    y la matriz de covarianza

        C = (1/n) sum_i (p_i - mu)(p_i - mu)^T

    los vectores propios de C dan las direcciones de maxima y minima
    dispersion, y la razon entre las raices de los valores propios
    estima la relacion de aspecto del objeto. Se usa como verificacion
    independiente de la deteccion de esquinas.
    """
    ys, xs = np.nonzero(mascara)
    P = np.vstack([xs, ys]).astype(np.float64)
    mu = P.mean(axis=1, keepdims=True)
    Q = P - mu
    C = (Q @ Q.T) / Q.shape[1]
    valores, vectores = np.linalg.eigh(C)
    orden = np.argsort(valores)[::-1]
    valores = valores[orden]
    vectores = vectores[:, orden]
    razon = float(np.sqrt(valores[0] / max(valores[1], 1e-9)))
    angulo = float(np.degrees(np.arctan2(vectores[1, 0], vectores[0, 0])))
    return mu.ravel(), valores, vectores, razon, angulo


def area_poligono(pts):
    """
    Area por la formula del polinomio del area (formula del cordon):

        A = (1/2) | sum_i (x_i * y_{i+1} - x_{i+1} * y_i) |
    """
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


# --------------------------------------------------------------------
# 2. Estimacion de la homografia (DLT)
# --------------------------------------------------------------------
def homografia_dlt(origen, destino):
    """
    Estima H tal que, en coordenadas homogeneas,

        [x' y' w']^T = H [x y 1]^T,    H = [[h11 h12 h13],
                                            [h21 h22 h23],
                                            [h31 h32  1 ]]

    En coordenadas cartesianas cada correspondencia aporta

        x' = (h11 x + h12 y + h13) / (h31 x + h32 y + 1)
        y' = (h21 x + h22 y + h23) / (h31 x + h32 y + 1)

    Multiplicando por el denominador se obtienen dos ecuaciones
    lineales en las ocho incognitas:

        h11 x + h12 y + h13 - h31 x x' - h32 y x' = x'
        h21 x + h22 y + h23 - h31 x y' - h32 y y' = y'

    Con cuatro correspondencias se arma el sistema A h = b con A de
    8 x 8, que se resuelve por eliminacion gaussiana. La matriz A es no
    singular siempre que no haya tres puntos colineales.
    """
    origen = np.asarray(origen, dtype=np.float64)
    destino = np.asarray(destino, dtype=np.float64)
    if origen.shape != (4, 2) or destino.shape != (4, 2):
        raise ValueError("Se requieren exactamente cuatro correspondencias.")

    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)

    for i in range(4):
        x, y = origen[i]
        xp, yp = destino[i]
        A[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp]
        b[2 * i] = xp
        A[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp]
        b[2 * i + 1] = yp

    h = np.linalg.solve(A, b)
    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7], 1.0]], dtype=np.float64)
    return H


def aplicar_homografia(H, pts):
    """Transforma puntos (n, 2) por H, normalizando la coordenada homogenea."""
    pts = np.asarray(pts, dtype=np.float64)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    q = homog @ H.T
    w = q[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return q[:, :2] / w


def error_reproyeccion(H, origen, destino):
    """
    Error cuadratico medio de reproyeccion en pixeles:

        e = sqrt( (1/n) sum_i || proy_H(p_i) - p_i' ||^2 )
    """
    proy = aplicar_homografia(H, origen)
    d2 = np.sum((proy - np.asarray(destino, dtype=np.float64)) ** 2, axis=1)
    return float(np.sqrt(d2.mean()))


# --------------------------------------------------------------------
# 3. Rectificacion por mapeo inverso con interpolacion bilineal
# --------------------------------------------------------------------
def interpolar_bilineal(img, x, y):
    """
    Interpolacion bilineal. Con x0 = floor(x), y0 = floor(y),
    a = x - x0 y b = y - y0:

        g(x, y) = (1-a)(1-b) f(y0,   x0)   + a(1-b) f(y0,   x0+1)
                + (1-a)b     f(y0+1, x0)   + a b     f(y0+1, x0+1)

    Los puntos fuera del dominio se fijan al borde mas cercano.
    """
    if img.ndim == 2:
        img = img[:, :, None]
        plano = True
    else:
        plano = False

    M, N, C = img.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    a = (x - x0)[..., None]
    b = (y - y0)[..., None]

    x0c = np.clip(x0, 0, N - 1)
    x1c = np.clip(x0 + 1, 0, N - 1)
    y0c = np.clip(y0, 0, M - 1)
    y1c = np.clip(y0 + 1, 0, M - 1)

    f00 = img[y0c, x0c]
    f01 = img[y0c, x1c]
    f10 = img[y1c, x0c]
    f11 = img[y1c, x1c]

    out = ((1 - a) * (1 - b) * f00
           + a * (1 - b) * f01
           + (1 - a) * b * f10
           + a * b * f11)
    return out[:, :, 0] if plano else out


def rectificar(img, esquinas, ancho, alto):
    """
    Construye la imagen rectificada de tamano (alto, ancho).

    Se trabaja con el mapeo INVERSO: para cada pixel (u, v) de la
    imagen de salida se calcula su preimagen

        (x, y) = proy_{H^{-1}}(u, v)

    y se interpola el valor alli. El mapeo directo dejaria huecos en la
    salida, porque la imagen de una retícula regular bajo una
    transformacion proyectiva no es regular.

    Devuelve (imagen_rectificada, H, error_reproyeccion).
    """
    destino = np.array([[0.0, 0.0],
                        [ancho - 1.0, 0.0],
                        [ancho - 1.0, alto - 1.0],
                        [0.0, alto - 1.0]], dtype=np.float64)

    H = homografia_dlt(esquinas, destino)
    err = error_reproyeccion(H, esquinas, destino)
    Hinv = np.linalg.inv(H)

    u = np.arange(ancho, dtype=np.float64)
    v = np.arange(alto, dtype=np.float64)
    U, V = np.meshgrid(u, v)

    den = Hinv[2, 0] * U + Hinv[2, 1] * V + Hinv[2, 2]
    den[np.abs(den) < 1e-12] = 1e-12
    X = (Hinv[0, 0] * U + Hinv[0, 1] * V + Hinv[0, 2]) / den
    Y = (Hinv[1, 0] * U + Hinv[1, 1] * V + Hinv[1, 2]) / den

    return interpolar_bilineal(img, X, Y), H, err


def ordenar_para_horizontal(esquinas):
    """
    Un billete es mas ancho que alto. Si las esquinas detectadas
    describen un cuadrilatero mas alto que ancho, se rota el orden un
    lugar, lo que equivale a girar el marco canonico 90 grados.
    """
    p = np.asarray(esquinas, dtype=np.float64)
    lado_sup = np.linalg.norm(p[1] - p[0])
    lado_der = np.linalg.norm(p[2] - p[1])
    if lado_sup < lado_der:
        return np.vstack([p[1], p[2], p[3], p[0]])
    return p
