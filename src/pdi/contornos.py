"""
Trazado de contornos y ajuste de rectangulo.

Este modulo implementa, sin recurrir a librerias de vision artificial,
las dos operaciones que el analisis de forma requiere:

  1. El seguimiento del contorno de una region binaria mediante el
     algoritmo de Moore, que recorre la frontera pixel a pixel en
     orden, a diferencia del etiquetado de componentes conexas, que
     solo indica a que region pertenece cada pixel.

  2. El rectangulo de area minima que encierra la region, obtenido por
     el metodo de calibradores rotantes sobre la envolvente convexa.

El rectangulo es la clave del criterio de seleccion: un billete es un
rectangulo de relacion de aspecto conocida, de modo que entre varias
regiones candidatas puede elegirse la que mejor se ajusta a esa forma.
"""
import numpy as np

# Vecindad de Moore en sentido horario, empezando por el oeste.
_MOORE = [(0, -1), (-1, -1), (-1, 0), (-1, 1),
          (0, 1), (1, 1), (1, 0), (1, -1)]


def trazar_contorno(B, max_pasos=200000):
    """
    Seguimiento de frontera por el algoritmo de Moore.

    Se parte del primer pixel del objeto en orden de escaneo, que por
    construccion pertenece a la frontera exterior. Desde el pixel
    actual se recorre su vecindad de Moore en sentido horario,
    comenzando por la posicion siguiente a aquella desde la que se
    llego, hasta encontrar el siguiente pixel del objeto. El recorrido
    termina al volver al punto de partida con la misma direccion de
    entrada, que es el criterio de parada de Jacob.

    Devuelve un arreglo (n, 2) de coordenadas (x, y) en el orden del
    recorrido, o None si la region esta vacia.
    """
    B = np.asarray(B, dtype=bool)
    M, N = B.shape
    idx = np.flatnonzero(B)
    if idx.size == 0:
        return None

    p0 = (int(idx[0] // N), int(idx[0] % N))     # (fila, columna)
    contorno = [p0]

    # Direccion de entrada inicial: se llego desde el oeste.
    dir_entrada = 0
    actual = p0
    anterior_dir = dir_entrada
    pasos = 0

    while pasos < max_pasos:
        pasos += 1
        encontrado = False
        # Se empieza a buscar en la posicion siguiente a la de entrada
        inicio = (anterior_dir + 1) % 8
        for k in range(8):
            d = (inicio + k) % 8
            di, dj = _MOORE[d]
            i, j = actual[0] + di, actual[1] + dj
            if 0 <= i < M and 0 <= j < N and B[i, j]:
                # La direccion de entrada al nuevo pixel es la opuesta
                anterior_dir = (d + 4) % 8
                actual = (i, j)
                encontrado = True
                break
        if not encontrado:
            break                                 # pixel aislado
        if actual == p0 and len(contorno) > 2:
            break                                 # criterio de Jacob
        contorno.append(actual)

    pts = np.array([[c[1], c[0]] for c in contorno], dtype=np.float64)
    return pts


def perimetro(contorno):
    """
    Longitud del contorno como suma de las distancias entre puntos
    consecutivos, cerrando el polinomio:

        P = sum_i || p_{i+1} - p_i ||
    """
    P = np.asarray(contorno, dtype=np.float64)
    d = np.diff(np.vstack([P, P[:1]]), axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def area_contorno(contorno):
    """
    Area encerrada por la formula del cordon:

        A = (1/2) | sum_i (x_i y_{i+1} - x_{i+1} y_i) |
    """
    P = np.asarray(contorno, dtype=np.float64)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def circularidad(contorno):
    """
    Medida adimensional de compacidad:

        C = 4 pi A / P^2

    Vale 1 para un circulo y aproximadamente 0.785 para un cuadrado.
    Un rectangulo alargado da valores mucho menores.
    """
    P = perimetro(contorno)
    if P < 1e-9:
        return 0.0
    return float(4.0 * np.pi * area_contorno(contorno) / (P ** 2))


def rectangulo_area_minima(hull):
    """
    Rectangulo de area minima que encierra un poligono convexo, por el
    metodo de calibradores rotantes.

    El teorema en que se apoya establece que al menos un lado del
    rectangulo de area minima es colineal con una arista de la
    envolvente convexa. Basta entonces recorrer las aristas: para cada
    una se define la base ortonormal

        u = (P_{i+1} - P_i) / ||P_{i+1} - P_i||,   v = u perpendicular

    se proyectan todos los vertices sobre ella,

        a_k = p_k . u ,     b_k = p_k . v

    y el area del rectangulo alineado con esa arista es

        A = (max a - min a)(max b - min b)

    Se conserva la orientacion que minimiza A. El costo es O(n) por
    arista con proyeccion vectorizada, es decir O(n^2) en total, o
    O(n) amortizado si se explota la monotonia de los extremos.

    Devuelve (esquinas, ancho, alto, angulo_grados).
    """
    P = np.asarray(hull, dtype=np.float64)
    n = len(P)
    if n < 3:
        return P, 0.0, 0.0, 0.0

    mejor = None
    mejor_area = np.inf

    for i in range(n):
        d = P[(i + 1) % n] - P[i]
        L = np.hypot(d[0], d[1])
        if L < 1e-9:
            continue
        u = d / L
        v = np.array([-u[1], u[0]])

        a = P @ u
        b = P @ v
        amin, amax = a.min(), a.max()
        bmin, bmax = b.min(), b.max()
        area = (amax - amin) * (bmax - bmin)

        if area < mejor_area:
            mejor_area = area
            esquinas = np.array([
                amin * u + bmin * v,
                amax * u + bmin * v,
                amax * u + bmax * v,
                amin * u + bmax * v,
            ])
            ancho, alto = amax - amin, bmax - bmin
            ang = float(np.degrees(np.arctan2(u[1], u[0])))
            mejor = (esquinas, float(ancho), float(alto), ang)

    return mejor if mejor is not None else (P[:4], 0.0, 0.0, 0.0)


def puntaje_billete(hull, aspecto_nominal=140.0 / 66.0,
                    area_imagen=None, tol_aspecto=0.45,
                    frac_min=0.06, frac_max=0.90):
    """
    Puntaje de cuan bien una region se ajusta a la forma de un billete.

    Combina tres evidencias independientes, todas adimensionales y
    acotadas en [0, 1]:

      1. Relacion de aspecto. Con r el cociente entre el lado mayor y
         el menor del rectangulo de area minima y r0 el valor nominal
         del billete, se mide la discrepancia relativa en escala
         logaritmica, que es simetrica ante inversion del cociente:

             s_a = exp( - (ln(r/r0))^2 / (2 tol^2) )

      2. Rectangularidad. Cociente entre el area de la region y la del
         rectangulo que la encierra. Vale 1 para un rectangulo
         perfecto y decae para formas irregulares:

             s_r = A_region / A_rectangulo

      3. Plausibilidad de tamano. Se evalua sobre el area de la
         propia region, no sobre la del rectangulo que la encierra,
         porque una franja delgada y diagonal puede tener un
         rectangulo grande y un area minuscula. Se anula el puntaje
         fuera del intervalo [frac_min, frac_max]: por debajo la
         region es un detalle interno del billete y no el billete; por
         encima es el fondo capturado por error.

    El puntaje final es el producto, de modo que basta que una
    evidencia falle para descartar la region.
    """
    esquinas, ancho, alto, _ = rectangulo_area_minima(hull)
    if ancho < 1e-6 or alto < 1e-6:
        return 0.0, None, {}

    lado_may, lado_men = max(ancho, alto), min(ancho, alto)
    r = lado_may / lado_men
    s_aspecto = float(np.exp(-(np.log(r / aspecto_nominal) ** 2)
                             / (2.0 * tol_aspecto ** 2)))

    a_hull = area_contorno(hull)
    a_rect = ancho * alto
    s_rect = float(np.clip(a_hull / max(a_rect, 1e-9), 0.0, 1.0))

    s_tam, frac = 1.0, None
    if area_imagen:
        frac = a_hull / float(area_imagen)
        if frac < frac_min or frac > frac_max:
            s_tam = 0.0
        else:
            # Rampas suaves en los extremos del intervalo admisible
            s_tam = float(min(np.clip((frac - frac_min) / 0.04, 0.0, 1.0),
                              np.clip((frac_max - frac) / 0.10, 0.0, 1.0)))

    diag = {"aspecto": r, "s_aspecto": s_aspecto,
            "s_rectangularidad": s_rect, "s_tamano": s_tam,
            "fraccion_cuadro": frac}
    return s_aspecto * s_rect * s_tam, esquinas, diag
