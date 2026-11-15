"""
Conversiones de espacio de color implementadas a partir de sus
definiciones matematicas, sin recurrir a rutinas de vision artificial.
"""
import numpy as np


def rgb_a_gris(img):
    """
    Luminancia segun la recomendacion ITU-R BT.601:

        I(x, y) = 0.2989 R + 0.5870 G + 0.1140 B

    Los pesos reflejan la sensibilidad relativa del sistema visual
    humano a cada primario.
    """
    return (0.2989 * img[:, :, 0]
            + 0.5870 * img[:, :, 1]
            + 0.1140 * img[:, :, 2])


def rgb_a_hsv(img):
    """
    Conversion RGB -> HSV sobre el cilindro de color.

    Con r, g, b en [0, 1], Cmax = max(r,g,b), Cmin = min(r,g,b)
    y delta = Cmax - Cmin:

        H = 60 * ((g-b)/delta mod 6)      si Cmax = r
        H = 60 * ((b-r)/delta + 2)        si Cmax = g
        H = 60 * ((r-g)/delta + 4)        si Cmax = b
        S = delta / Cmax                  (0 si Cmax = 0)
        V = Cmax

    Devuelve H en [0, 360), S y V en [0, 1].
    """
    a = img / 255.0
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    cmax = a.max(axis=2)
    cmin = a.min(axis=2)
    delta = cmax - cmin

    H = np.zeros_like(cmax)
    nz = delta > 1e-12

    m_r = nz & (cmax == r)
    m_g = nz & (cmax == g) & (~m_r)
    m_b = nz & (cmax == b) & (~m_r) & (~m_g)

    H[m_r] = 60.0 * (((g[m_r] - b[m_r]) / delta[m_r]) % 6.0)
    H[m_g] = 60.0 * (((b[m_g] - r[m_g]) / delta[m_g]) + 2.0)
    H[m_b] = 60.0 * (((r[m_b] - g[m_b]) / delta[m_b]) + 4.0)

    S = np.zeros_like(cmax)
    pos = cmax > 1e-12
    S[pos] = delta[pos] / cmax[pos]

    return H, S, cmax


def balance_de_blancos(img, p=6.0):
    """
    Constancia de color por la hipotesis de Minkowski (shades of gray).

    Para cada canal c se estima el iluminante mediante la norma p:

        e_c = ( (1/|Omega|) * sum_x f_c(x)^p )^(1/p)

    y se corrige con la ganancia  k / e_c, con k la media de los e_c.
    Con p = 1 se recupera la hipotesis del mundo gris y con p -> inf
    la del parche blanco. Esta etapa reduce la dependencia del tono
    respecto de la temperatura de color de la iluminacion.
    """
    eps = 1e-9
    e = np.array([
        (np.mean(img[:, :, c] ** p)) ** (1.0 / p) + eps for c in range(3)
    ])
    k = e.mean()
    out = img * (k / e)[None, None, :]
    return np.clip(out, 0, 255)


def hist_matiz(H, S, V, mascara=None, bins=36, s_min=0.12, v_min=0.15):
    """
    Histograma circular normalizado del matiz.

        h(k) = (1/|Omega_k|) * sum_{x in Omega} delta[ Q(H(x)) - k ]

    Solo se acumulan pixeles con saturacion y valor suficientes, ya que
    el matiz queda indefinido cuando S -> 0 y es ruidoso cuando V -> 0.
    """
    val = (S >= s_min) & (V >= v_min)
    if mascara is not None:
        val &= mascara.astype(bool)
    if val.sum() < 16:
        return np.zeros(bins)
    idx = np.floor(H[val] / 360.0 * bins).astype(int) % bins
    h = np.bincount(idx, minlength=bins).astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else h


def suavizar_circular(h, nucleo=(1.0, 2.0, 3.0, 2.0, 1.0)):
    """Suaviza un histograma circular por convolucion periodica."""
    k = np.asarray(nucleo, dtype=np.float64)
    k = k / k.sum()
    r = len(k) // 2
    ext = np.concatenate([h[-r:], h, h[:r]])
    out = np.convolve(ext, k, mode="same")[r:-r]
    s = out.sum()
    return out / s if s > 0 else out
