"""
Etiquetado de componentes conexas.

Esta rutina sustituye el uso de funciones de extraccion de contornos de
librerias de vision artificial. Se implementa el algoritmo clasico de
dos pasadas de Rosenfeld y Pfaltz con una estructura de conjuntos
disjuntos (union-find) para resolver las equivalencias entre etiquetas
provisionales.

Modelo de conectividad: se emplea vecindad de 8, es decir, dos pixeles
p y q del primer plano pertenecen a la misma componente si existe un
camino p = p_0, p_1, ..., p_n = q tal que cada par consecutivo cumple

    max( |x_i - x_{i+1}| , |y_i - y_{i+1}| ) = 1
"""
import numpy as np


class ConjuntosDisjuntos:
    """Union-find con compresion de caminos y union por rango."""

    def __init__(self, n=0):
        self.padre = list(range(n))
        self.rango = [0] * n

    def nuevo(self):
        self.padre.append(len(self.padre))
        self.rango.append(0)
        return len(self.padre) - 1

    def buscar(self, x):
        raiz = x
        while self.padre[raiz] != raiz:
            raiz = self.padre[raiz]
        while self.padre[x] != raiz:      # compresion de caminos
            self.padre[x], x = raiz, self.padre[x]
        return raiz

    def unir(self, a, b):
        ra, rb = self.buscar(a), self.buscar(b)
        if ra == rb:
            return ra
        if self.rango[ra] < self.rango[rb]:
            ra, rb = rb, ra
        self.padre[rb] = ra
        if self.rango[ra] == self.rango[rb]:
            self.rango[ra] += 1
        return ra


def etiquetar(B):
    """
    Devuelve (etiquetas, n) donde etiquetas es un arreglo entero del
    mismo tamano que B, con 0 en el fondo y 1..n en las componentes.

    Primera pasada: recorrido en orden de escaneo. Para cada pixel del
    primer plano se examinan los vecinos ya visitados (mascara causal):

        (i-1, j-1)  (i-1, j)  (i-1, j+1)
        (i,   j-1)  (i,   j)

    Si ninguno esta etiquetado se crea una etiqueta nueva; si hay varios
    se toma la menor y se registran las equivalencias.

    Segunda pasada: cada etiqueta provisional se reemplaza por el
    representante de su clase de equivalencia, y estas se renumeran
    de forma consecutiva.
    """
    B = np.asarray(B, dtype=bool)
    M, N = B.shape
    etiq = np.zeros((M, N), dtype=np.int32)
    uf = ConjuntosDisjuntos(1)          # indice 0 reservado al fondo

    vecinos = ((-1, -1), (-1, 0), (-1, 1), (0, -1))

    for i in range(M):
        fila = B[i]
        if not fila.any():
            continue
        for j in np.flatnonzero(fila):
            cand = []
            for di, dj in vecinos:
                ii, jj = i + di, j + dj
                if 0 <= ii < M and 0 <= jj < N and etiq[ii, jj] > 0:
                    cand.append(etiq[ii, jj])
            if not cand:
                etiq[i, j] = uf.nuevo()
            else:
                e = min(cand)
                etiq[i, j] = e
                for c in cand:
                    if c != e:
                        uf.unir(e, c)

    # Segunda pasada: resolucion de equivalencias y renumeracion
    tabla = np.zeros(len(uf.padre), dtype=np.int32)
    siguiente = 0
    mapa = {}
    for e in range(1, len(uf.padre)):
        r = uf.buscar(e)
        if r not in mapa:
            siguiente += 1
            mapa[r] = siguiente
        tabla[e] = mapa[r]

    return tabla[etiq], siguiente


def areas(etiquetas, n):
    """Area en pixeles de cada componente (indice 0 = fondo)."""
    return np.bincount(etiquetas.ravel(), minlength=n + 1)


def componente_mayor(B):
    """
    Devuelve la mascara booleana de la componente conexa de mayor area.
    Si no hay primer plano devuelve una mascara vacia.
    """
    etiq, n = etiquetar(B)
    if n == 0:
        return np.zeros_like(B, dtype=bool)
    a = areas(etiq, n)
    a[0] = 0
    return etiq == int(np.argmax(a))


def componente_mayor_multiescala(B, lado_max=320):
    """
    Estrategia multiresolucion.

    El etiquetado exacto tiene costo lineal en el numero de pixeles,
    pero la constante es alta al recorrer pixel a pixel. Dado que la
    region de interes ocupa una fraccion grande del cuadro, se etiqueta
    sobre una version decimada de la mascara, se selecciona alli la
    componente de mayor area y se proyecta de vuelta a la resolucion
    original intersecandola con la mascara original. Esto conserva la
    precision de los bordes, que es lo que importa para localizar las
    esquinas, y reduce el costo del etiquetado.
    """
    M, N = B.shape
    L = max(M, N)
    if L <= lado_max:
        return componente_mayor(B)

    k = int(np.ceil(L / float(lado_max)))
    M2, N2 = (M // k) * k, (N // k) * k
    reducida = B[:M2, :N2].reshape(M2 // k, k, N2 // k, k).mean(axis=(1, 3)) > 0.5

    sel = componente_mayor(reducida)
    if not sel.any():
        return componente_mayor(B)

    # Expansion por replicacion de cada pixel decimado
    grande = np.kron(sel, np.ones((k, k), dtype=bool))
    salida = np.zeros((M, N), dtype=bool)
    salida[:M2, :N2] = grande[:M2, :N2]

    # Se dilata un paso para no perder el borde por la decimacion
    salida[:M2, :N2] |= np.roll(grande, 1, axis=0)[:M2, :N2]
    salida[:M2, :N2] |= np.roll(grande, -1, axis=0)[:M2, :N2]
    return salida & B
