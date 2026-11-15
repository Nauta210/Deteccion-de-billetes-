"""
Descriptores usados por el modulo de verificacion de autenticidad.

Todos se calculan a partir de su definicion matematica sobre arreglos
numericos. La transformada de Fourier se evalua con la implementacion
de FFT de la libreria numerica, que es un algoritmo de calculo de la
DFT y no una rutina de vision artificial.
"""
import numpy as np
from .filtros import laplaciano


# --------------------------------------------------------------------
# Patrones Binarios Locales (LBP)
# --------------------------------------------------------------------
def lbp(img, P=8, R=1):
    """
    Operador LBP con P vecinos sobre una circunferencia de radio R:

        LBP_{P,R}(x_c) = sum_{p=0}^{P-1} s( g_p - g_c ) 2^p

        s(z) = 1 si z >= 0,  0 en otro caso

    donde g_c es la intensidad del pixel central y g_p la del vecino
    p-esimo, situado en

        (x_c + R cos(2*pi*p/P),  y_c - R sin(2*pi*p/P))

    Los vecinos no centrados en la retícula se obtienen por
    interpolacion bilineal.
    """
    M, N = img.shape
    centro = img[R:M - R, R:N - R]
    codigo = np.zeros_like(centro, dtype=np.int32)

    for p in range(P):
        ang = 2.0 * np.pi * p / P
        dx = R * np.cos(ang)
        dy = -R * np.sin(ang)
        vec = _muestrear(img, dx, dy, R)
        codigo |= ((vec >= centro).astype(np.int32) << p)

    return codigo


def _muestrear(img, dx, dy, R):
    """Vecino desplazado (dx, dy) con interpolacion bilineal."""
    M, N = img.shape
    ys, xs = np.mgrid[R:M - R, R:N - R]
    x = xs + dx
    y = ys + dy
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    a = x - x0
    b = y - y0
    x0 = np.clip(x0, 0, N - 2)
    y0 = np.clip(y0, 0, M - 2)
    return ((1 - a) * (1 - b) * img[y0, x0]
            + a * (1 - b) * img[y0, x0 + 1]
            + (1 - a) * b * img[y0 + 1, x0]
            + a * b * img[y0 + 1, x0 + 1])


def _transiciones(c, P):
    """Numero de transiciones binarias circulares U(LBP)."""
    bits = [(c >> p) & 1 for p in range(P)]
    u = 0
    for p in range(P):
        u += bits[p] ^ bits[(p + 1) % P]
    return u


def hist_lbp_uniforme(img, P=8, R=1):
    """
    Histograma de patrones uniformes invariantes a rotacion.

    Un patron es uniforme si U(LBP) <= 2, donde U cuenta las
    transiciones 0-1 y 1-0 al recorrer circularmente el codigo. Para
    estos patrones se define

        LBP^{riu2}_{P,R} = sum_p s(g_p - g_c)   si U <= 2
                         = P + 1                 en otro caso

    lo que produce P + 2 clases. Los patrones uniformes concentran la
    mayor parte de la informacion discriminante de la textura y el
    descriptor resultante es invariante a rotacion y a cambios
    monotonicos de la escala de grises.
    """
    codigo = lbp(img, P, R)
    tabla = np.zeros(1 << P, dtype=np.int32)
    for c in range(1 << P):
        if _transiciones(c, P) <= 2:
            tabla[c] = bin(c).count("1")
        else:
            tabla[c] = P + 1
    mapeado = tabla[codigo]
    h = np.bincount(mapeado.ravel(), minlength=P + 2).astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else h


# --------------------------------------------------------------------
# Matriz de co-ocurrencia de niveles de gris (GLCM)
# --------------------------------------------------------------------
def glcm(img, niveles=32, dx=1, dy=0, simetrica=True):
    """
    Matriz de co-ocurrencia P_{d,theta}(i, j): frecuencia relativa con
    que un pixel de nivel i aparece separado del desplazamiento
    (dx, dy) de un pixel de nivel j.

    La imagen se cuantiza previamente a un numero reducido de niveles
    para que la matriz sea estadisticamente estable con el numero de
    muestras disponible en la region de interes.
    """
    q = np.clip((img * (niveles - 1)).round(), 0, niveles - 1).astype(np.int32)
    M, N = q.shape

    i0 = slice(max(0, -dy), M - max(0, dy))
    j0 = slice(max(0, -dx), N - max(0, dx))
    i1 = slice(max(0, dy), M - max(0, -dy))
    j1 = slice(max(0, dx), N - max(0, -dx))

    a = q[i0, j0].ravel()
    b = q[i1, j1].ravel()

    P = np.bincount(a * niveles + b, minlength=niveles * niveles)
    P = P.reshape(niveles, niveles).astype(np.float64)
    if simetrica:
        P = P + P.T
    s = P.sum()
    return P / s if s > 0 else P


def descriptores_haralick(P):
    """
    Medidas de Haralick derivadas de la GLCM:

        Contraste     = sum_ij (i - j)^2 P(i, j)
        Homogeneidad  = sum_ij P(i, j) / (1 + |i - j|)
        Energia       = sum_ij P(i, j)^2
        Entropia      = - sum_ij P(i, j) log( P(i, j) )
        Correlacion   = sum_ij (i - mu_i)(j - mu_j) P(i, j) / (sd_i sd_j)

    La energia y la homogeneidad crecen con la regularidad de la
    textura, mientras que la entropia crece con su desorden. Una trama
    de semitono regular y un guilloche continuo difieren precisamente
    en estas propiedades.
    """
    n = P.shape[0]
    i, j = np.mgrid[0:n, 0:n]

    contraste = float(np.sum(((i - j) ** 2) * P))
    homog = float(np.sum(P / (1.0 + np.abs(i - j))))
    energia = float(np.sum(P ** 2))

    Pp = P[P > 0]
    entropia = float(-np.sum(Pp * np.log(Pp)))

    mu_i = float(np.sum(i * P))
    mu_j = float(np.sum(j * P))
    sd_i = float(np.sqrt(np.sum(((i - mu_i) ** 2) * P)))
    sd_j = float(np.sqrt(np.sum(((j - mu_j) ** 2) * P)))
    if sd_i > 1e-9 and sd_j > 1e-9:
        corr = float(np.sum((i - mu_i) * (j - mu_j) * P) / (sd_i * sd_j))
    else:
        corr = 0.0

    return np.array([contraste, homog, energia, entropia, corr])


# --------------------------------------------------------------------
# Analisis espectral: deteccion de trama de semitono
# --------------------------------------------------------------------
def espectro_magnitud(img):
    """
    Espectro de magnitud de la DFT bidimensional:

        F(u, v) = sum_x sum_y f(x, y) exp(-j 2 pi (ux/M + vy/N))

    Se aplica previamente una ventana de Hann separable para atenuar
    la discontinuidad en los bordes del bloque, que de otro modo
    introduce energia espuria en forma de cruz a lo largo de los ejes.
    El espectro se centra desplazando el origen de frecuencias.
    """
    M, N = img.shape
    wm = np.hanning(M)[:, None]
    wn = np.hanning(N)[None, :]
    f = (img - img.mean()) * (wm * wn)
    F = np.fft.fftshift(np.fft.fft2(f))
    return np.abs(F)


def energia_radial(img, r1=0.25, r2=0.45):
    """
    Razon de energia contenida en un anillo de frecuencias:

        E_rho = sum_{(u,v) in A} |F(u,v)|^2  /  sum_{(u,v)} |F(u,v)|^2

        A = { (u, v) : r1 <= rho(u, v) <= r2 }

    donde rho es el radio normalizado respecto de la frecuencia de
    Nyquist. Una impresora de consumo reproduce los tonos continuos
    mediante una trama periodica de puntos, cuya frecuencia fundamental
    aparece como picos concentrados en el anillo correspondiente. La
    impresion de seguridad, basada en lineas continuas de guilloche,
    no presenta esa concentracion.
    """
    A = espectro_magnitud(img)
    M, N = A.shape
    v = (np.arange(M) - M // 2) / (M / 2.0)
    u = (np.arange(N) - N // 2) / (N / 2.0)
    U, V = np.meshgrid(u, v)
    rho = np.sqrt(U ** 2 + V ** 2)

    pot = A ** 2
    total = pot.sum()
    if total <= 0:
        return 0.0, 0.0
    anillo = (rho >= r1) & (rho <= r2)
    razon = float(pot[anillo].sum() / total)

    # Pico espectral dentro del anillo, relativo a la media del anillo.
    # Cuantifica cuan concentrada (periodica) es la energia del anillo.
    if anillo.sum() > 0:
        media = pot[anillo].mean()
        pico = float(pot[anillo].max() / (media + 1e-12))
    else:
        pico = 0.0
    return razon, pico


def perfil_radial(img, nbins=64):
    """Perfil de potencia promedio por banda radial de frecuencia."""
    A = espectro_magnitud(img)
    M, N = A.shape
    v = (np.arange(M) - M // 2) / (M / 2.0)
    u = (np.arange(N) - N // 2) / (N / 2.0)
    U, V = np.meshgrid(u, v)
    rho = np.clip(np.sqrt(U ** 2 + V ** 2), 0, 1.0)
    idx = np.clip((rho * (nbins - 1)).astype(int), 0, nbins - 1)
    pot = (A ** 2).ravel()
    suma = np.bincount(idx.ravel(), weights=pot, minlength=nbins)
    cuenta = np.bincount(idx.ravel(), minlength=nbins)
    return suma / np.maximum(cuenta, 1)


# --------------------------------------------------------------------
# Nitidez local
# --------------------------------------------------------------------
def varianza_laplaciano(img):
    """
    Medida de nitidez basada en la energia del laplaciano:

        V_L = (1/|Omega|) sum_x ( lap f(x) - media(lap f) )^2

    Toda reproduccion impresa atraviesa un muestreo adicional que actua
    como filtro pasabajas, de modo que el microtexto pierde definicion
    y V_L disminuye.
    """
    L = laplaciano(img)
    return float(np.var(L))


def contraste_local(img, ventana=9):
    """Desviacion estandar local promedio, como medida complementaria."""
    from .filtros import convolucionar
    k = np.ones((ventana, ventana)) / (ventana * ventana)
    mu = convolucionar(img, k)
    mu2 = convolucionar(img ** 2, k)
    var = np.maximum(mu2 - mu ** 2, 0.0)
    return float(np.mean(np.sqrt(var)))


# --------------------------------------------------------------------
# Vector de caracteristicas completo
# --------------------------------------------------------------------
NOMBRES = (
    [f"lbp_{i}" for i in range(10)]
    + ["glcm_contraste", "glcm_homogeneidad", "glcm_energia",
       "glcm_entropia", "glcm_correlacion"]
    + ["fft_razon", "fft_pico", "lap_var", "contraste_local"]
)


def vector_caracteristicas(roi_gris):
    """
    Concatena las cuatro familias de descriptores sobre una region de
    interes en escala de grises normalizada a [0, 1].

    Dimension total: 10 (LBP) + 5 (Haralick) + 2 (espectral)
                     + 1 (laplaciano) + 1 (contraste) = 19
    """
    x = np.clip(roi_gris, 0.0, 1.0)
    h = hist_lbp_uniforme(x, P=8, R=1)
    P = glcm(x, niveles=32, dx=1, dy=0)
    har = descriptores_haralick(P)
    razon, pico = energia_radial(x)
    vl = varianza_laplaciano(x)
    cl = contraste_local(x)
    return np.concatenate([h, har, [razon, pico, vl, cl]])


# --------------------------------------------------------------------
# Regiones de interes sobre el marco canonico
# --------------------------------------------------------------------
# Centros normalizados (cx, cy) de cada region, medidos sobre el
# billete rectificado. Se ubican sobre zonas de fondo con trama de
# guilloche y microtexto, evitando el retrato, que esta impreso en
# calcografia y cuya textura responde a otra fisica.
#
# El conjunto es simetrico respecto del centro: bajo un giro de media
# vuelta, (cx, cy) -> (1-cx, 1-cy) lleva cada region aproximadamente
# sobre otra del mismo conjunto. Por eso las medidas promediadas sobre
# todas las regiones son casi invariantes a la orientacion del billete,
# y no hace falta resolver la ambiguedad de media vuelta.
ROIS_NORM = [
    (0.09, 0.22), (0.09, 0.78),
    (0.50, 0.12), (0.50, 0.88),
    (0.88, 0.25), (0.88, 0.72),
]
LADO_ROI = 144


def extraer_rois(imagen_gris, lado=LADO_ROI):
    """Recorta las regiones fijas del marco canonico."""
    M, N = imagen_gris.shape
    r = lado // 2
    salida = []
    for cx, cy in ROIS_NORM:
        x = int(np.clip(int(round(cx * N)), r, N - r - 1))
        y = int(np.clip(int(round(cy * M)), r, M - r - 1))
        salida.append(imagen_gris[y - r:y + r, x - r:x + r])
    return salida
