"""
Envolvente convexa y rasterizacion de poligonos.

La envolvente convexa del conjunto de pixeles del objeto permite
recuperar la forma cuadrangular completa del billete aunque la
umbralizacion haya dejado huecos o haya perdido zonas de bajo
contraste con el fondo.
"""
import numpy as np


def _producto_cruz(o, a, b):
    """
    Componente z del producto cruz (a - o) x (b - o).

        cruz = (a_x - o_x)(b_y - o_y) - (a_y - o_y)(b_x - o_x)

    Su signo indica si el recorrido o -> a -> b gira a la izquierda
    (positivo), a la derecha (negativo) o es colineal (cero).
    """
    return ((a[0] - o[0]) * (b[1] - o[1])
            - (a[1] - o[1]) * (b[0] - o[0]))


def envolvente_convexa(puntos):
    """
    Algoritmo de la cadena monotona de Andrew, de costo O(n log n)
    dominado por el ordenamiento.

    Se ordenan los puntos lexicograficamente y se construyen la cadena
    inferior y la superior, descartando todo vertice que produzca un
    giro no levogiro. La concatenacion de ambas cadenas es la
    envolvente en sentido antihorario.
    """
    P = np.unique(np.asarray(puntos, dtype=np.float64), axis=0)
    if len(P) <= 3:
        return P
    orden = np.lexsort((P[:, 1], P[:, 0]))
    P = P[orden]

    inferior = []
    for p in P:
        while len(inferior) >= 2 and _producto_cruz(inferior[-2], inferior[-1], p) <= 0:
            inferior.pop()
        inferior.append(p)

    superior = []
    for p in P[::-1]:
        while len(superior) >= 2 and _producto_cruz(superior[-2], superior[-1], p) <= 0:
            superior.pop()
        superior.append(p)

    return np.array(inferior[:-1] + superior[:-1])


def contorno_mascara(mascara, paso=1):
    """
    Pixeles de frontera de la mascara: los del objeto que tienen al
    menos un vecino de fondo en conectividad 4. Se obtienen restando
    la erosion, que es la definicion morfologica del borde interno:

        borde(A) = A - (A (-) B_4)
    """
    B = np.asarray(mascara, dtype=bool)
    interior = (B
                & np.roll(B, 1, 0) & np.roll(B, -1, 0)
                & np.roll(B, 1, 1) & np.roll(B, -1, 1))
    borde = B & ~interior
    ys, xs = np.nonzero(borde)
    pts = np.vstack([xs, ys]).T.astype(np.float64)
    return pts[::paso] if paso > 1 else pts


def rasterizar_poligono(poligono, alto, ancho):
    """
    Rellena un poligono convexo evaluando, para cada pixel, si queda a
    la izquierda de todas sus aristas orientadas. Para un poligono
    convexo recorrido en sentido antihorario, el interior es la
    interseccion de los semiplanos

        (x - x_i)(y_{i+1} - y_i) - (y - y_i)(x_{i+1} - x_i) <= 0
    """
    P = np.asarray(poligono, dtype=np.float64)
    ys, xs = np.mgrid[0:alto, 0:ancho]
    dentro = np.ones((alto, ancho), dtype=bool)
    n = len(P)
    for i in range(n):
        x0, y0 = P[i]
        x1, y1 = P[(i + 1) % n]
        lado = (xs - x0) * (y1 - y0) - (ys - y0) * (x1 - x0)
        dentro &= (lado <= 1e-9)
    return dentro


def cuadrilatero_desde_envolvente(hull, max_vertices=72):
    """
    Cuadrilatero de area maxima inscrito en la envolvente convexa.

    Se recorre cada par de vertices (i, j) como diagonal candidata.
    Fijada la diagonal, el cuadrilatero de area maxima que la contiene
    se obtiene tomando, a cada lado de la recta que pasa por P_i y
    P_j, el vertice mas alejado de ella. La distancia con signo de un
    punto P a esa recta es

        s(P) = ( (P - P_i) x (P_j - P_i) ) / ||P_j - P_i||

    y el area resultante es

        A = (1/2) ||P_j - P_i|| ( max_{s>0} s  +  max_{s<0} |s| )

    El costo es O(n^2) pares, con la evaluacion de distancias
    vectorizada sobre los n vertices. Este procedimiento es exacto
    sobre la envolvente, a diferencia de una busqueda exhaustiva sobre
    un subconjunto de vertices.
    """
    P = np.asarray(hull, dtype=np.float64)
    n = len(P)
    if n <= 4:
        return P
    if n > max_vertices:
        P = P[np.linspace(0, n - 1, max_vertices).astype(int)]
        n = len(P)

    mejor_area, mejor = -1.0, None
    for i in range(n):
        for j in range(i + 2, n):
            d = P[j] - P[i]
            L = np.hypot(d[0], d[1])
            if L < 1e-9:
                continue
            rel = P - P[i]
            s = (rel[:, 0] * d[1] - rel[:, 1] * d[0]) / L
            pos, neg = s.max(), s.min()
            if pos <= 0 or neg >= 0:
                continue
            area = 0.5 * L * (pos - neg)
            if area > mejor_area:
                mejor_area = area
                mejor = np.vstack([P[i], P[int(np.argmax(s))],
                                   P[j], P[int(np.argmin(s))]])
    return mejor if mejor is not None else P[:4]
