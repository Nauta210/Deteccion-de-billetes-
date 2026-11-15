"""
Segmentacion del billete y rectificacion geometrica.

Estructura de la cadena, en el orden en que se ejecuta:

    imagen
      -> varios mapas de saliencia (evidencias complementarias)
      -> umbralizacion de Otsu + histeresis por reconstruccion geodesica
      -> morfologia matematica
      -> etiquetado de componentes conexas
      -> trazado de contorno y rectangulo de area minima
      -> SELECCION de la region que mejor se ajusta a un billete
      -> homografia y rectificacion
      -> billete segmentado SIN FONDO

La seleccion por forma es lo que da robustez. Ningun mapa de saliencia
funciona en todas las condiciones: la distancia cromatica separa bien
un billete de color sobre una mesa neutra, pero falla si el fondo es
oscuro y texturizado; la distancia de luminancia hace lo contrario. En
lugar de elegir una de antemano, se generan regiones candidatas con
cada mapa y se conserva la que mejor se ajusta a la forma conocida del
objeto: un rectangulo de relacion de aspecto 140 a 66.
"""
import numpy as np

from . import color as C
from . import filtros as F
from . import umbral as U
from . import etiquetado as E
from . import geometria as G
from . import convexa as CV
from . import contornos as CT

ASPECTO_BILLETE = 140.0 / 66.0
ANCHO_CANONICO = 880
ALTO_CANONICO = int(round(ANCHO_CANONICO / ASPECTO_BILLETE))


# ====================================================================
# 1. Mapas de saliencia
# ====================================================================
def cromaticidad(img_rgb):
    """
    Coordenadas de cromaticidad normalizadas:

        r = R / (R+G+B),    g = G / (R+G+B)

    Bajo el modelo lambertiano la intensidad observada es el producto
    del albedo por la irradiancia. Al dividir por la suma de los tres
    canales el factor de irradiancia se cancela, de modo que la
    cromaticidad es invariante a la intensidad de la iluminacion. Esa
    propiedad permite que una sombra proyectada conserve la
    cromaticidad del fondo y no se confunda con el objeto.
    """
    suma = img_rgb.sum(axis=2) + 1e-6
    return np.dstack([img_rgb[:, :, 0] / suma, img_rgb[:, :, 1] / suma])


def modelo_de_fondo(campo, franja=0.06):
    """
    Media y covarianza del fondo, estimadas sobre una franja perimetral
    del cuadro, bajo el supuesto de que el objeto no toca los bordes.
    """
    M, N = campo.shape[:2]
    d = campo.shape[2] if campo.ndim == 3 else 1
    X = campo.reshape(M, N, d)
    m = max(2, int(round(franja * min(M, N))))
    banda = np.concatenate([
        X[:m, :, :].reshape(-1, d), X[-m:, :, :].reshape(-1, d),
        X[:, :m, :].reshape(-1, d), X[:, -m:, :].reshape(-1, d),
    ], axis=0)
    mu = banda.mean(axis=0)
    Q = banda - mu
    Sigma = np.atleast_2d((Q.T @ Q) / max(len(banda) - 1, 1)) + np.eye(d) * 1e-5
    return mu, Sigma


def distancia_mahalanobis(campo, mu, Sigma):
    """
    Distancia de Mahalanobis de cada pixel al modelo de fondo:

        d(x) = sqrt( (f(x) - mu)^T Sigma^{-1} (f(x) - mu) )

    A diferencia de la euclidiana, pondera cada direccion del espacio
    por la dispersion que el fondo presenta en ella. Un fondo uniforme
    tiene covarianza pequena, de modo que desviaciones leves producen
    distancias grandes.
    """
    M, N = campo.shape[:2]
    d = campo.shape[2] if campo.ndim == 3 else 1
    Si = np.linalg.inv(Sigma)
    D = campo.reshape(-1, d) - mu
    d2 = np.einsum("ij,jk,ik->i", D, Si, D)
    return np.sqrt(np.maximum(d2, 0.0)).reshape(M, N)


def _norm(x, percentil=99.0, sigma=None):
    y = np.clip(x / max(np.percentile(x, percentil), 1e-9), 0.0, 1.0)
    return F.suavizado_gaussiano(y, sigma=sigma) if sigma else y


def mapas_de_saliencia(img_rgb):
    """
    Evidencias complementarias que alimentan la segmentacion. Cada una
    responde a un fenomeno distinto:

      cromatica : distancia de Mahalanobis de la cromaticidad al modelo
                  de fondo. Separa un billete de color sobre superficie
                  neutra y es insensible a sombras.
      luminancia: distancia de Mahalanobis de la luminancia. Separa un
                  billete claro sobre fondo oscuro, caso en que la
                  cromatica es inestable.
      brillo    : luminancia normalizada. Es el mapa mas simple de
                  todos y precisamente por eso el mas robusto cuando
                  el fondo es oscuro y texturizado: alli la franja
                  perimetral mezcla zonas muy oscuras con reflejos
                  especulares, la covarianza estimada se dispara y las
                  dos distancias de Mahalanobis dejan de discriminar,
                  mientras que el billete sigue siendo simplemente la
                  region mas clara del cuadro.
      borde     : magnitud del gradiente suavizada. Delimita el borde
                  del papel cuando el fondo es parecido en color y en
                  brillo.
      combinada : promedio ponderado de las anteriores.
    """
    gris = C.rgb_a_gris(img_rgb) / 255.0

    crom = cromaticidad(img_rgb)
    mu_c, S_c = modelo_de_fondo(crom)
    s_crom = _norm(distancia_mahalanobis(crom, mu_c, S_c), sigma=2.0)

    lum = gris[:, :, None]
    mu_l, S_l = modelo_de_fondo(lum)
    s_lum = _norm(distancia_mahalanobis(lum, mu_l, S_l), sigma=2.0)

    mag, _, _ = F.gradiente(F.suavizado_gaussiano(gris, sigma=1.2))
    s_borde = _norm(F.suavizado_gaussiano(_norm(mag, 100.0), sigma=5.0), 100.0)

    s_brillo = _norm(gris, 99.0, sigma=2.0)

    return {"cromatica": s_crom, "luminancia": s_lum, "borde": s_borde,
            "brillo": s_brillo,
            "combinada": np.clip(0.45 * s_crom + 0.35 * s_lum
                                 + 0.20 * s_borde, 0.0, 1.0)}


# ====================================================================
# 2. Mascara binaria
# ====================================================================
def mascara_binaria(sal, factor_bajo=0.40, radio_morf=3, lado_max=320,
                    radio_cierre=None):
    """
    Umbralizacion e higiene morfologica de un mapa de saliencia.

    Otsu da el umbral alto y se define un umbral bajo proporcional. Los
    pixeles sobre el alto son semillas seguras; los que superan solo el
    bajo son candidatos. La mascara final es la reconstruccion
    morfologica de las semillas dentro de los candidatos,

        R_X(M) = delta^{(n)}_X(M),

    es decir, las componentes de la mascara laxa que contienen al menos
    una semilla. Esto recupera las zonas palidas del billete, que un
    umbral unico descartaria, sin incorporar regiones del fondo no
    conectadas a ninguna semilla.

    El radio del cierre es un parametro y no una constante porque
    determina la separacion minima que el operador puede preservar:
    un cierre de radio r une dos regiones separadas por menos de 2r
    pixeles. Un radio grande consolida un billete cuya mascara quedo
    fragmentada, pero fusiona dos billetes contiguos en una sola
    region. Explorar varios radios y quedarse con las regiones que
    mejor se ajusten a la forma esperada evita tener que acertar el
    valor de antemano.
    """
    T, _ = U.otsu(sal, niveles=256, rango=(0.0, 1.0))
    alto, bajo = sal > T, sal > (factor_bajo * T)

    M, N = sal.shape
    k = max(1, int(np.ceil(max(M, N) / float(lado_max))))
    M2, N2 = (M // k) * k, (N // k) * k

    def _dec(B):
        return B[:M2, :N2].reshape(M2 // k, k, N2 // k, k).mean(axis=(1, 3)) > 0.3

    rec = U.reconstruccion_morfologica(_dec(alto), _dec(bajo))
    B = np.zeros((M, N), dtype=bool)
    B[:M2, :N2] = np.kron(rec, np.ones((k, k), dtype=bool))[:M2, :N2]
    B &= bajo

    if radio_cierre is None:
        radio_cierre = radio_morf * 4
    B = U.abrir(B, U.elemento_elipse(radio_morf))
    B = U.cerrar(B, U.elemento_elipse(radio_cierre))
    return U.rellenar_huecos(B)


def _decimar(B, lado_max=320):
    M, N = B.shape
    k = max(1, int(np.ceil(max(M, N) / float(lado_max))))
    M2, N2 = (M // k) * k, (N // k) * k
    return B[:M2, :N2].reshape(M2 // k, k, N2 // k, k).mean(axis=(1, 3)) > 0.5


def _ampliar(B, forma, lado_max=320):
    M, N = forma
    k = max(1, int(np.ceil(max(M, N) / float(lado_max))))
    grande = np.kron(B, np.ones((k, k), dtype=bool))
    out = np.zeros(forma, dtype=bool)
    h, w = min(M, grande.shape[0]), min(N, grande.shape[1])
    out[:h, :w] = grande[:h, :w]
    out |= np.roll(out, 1, 0) | np.roll(out, -1, 0)
    out |= np.roll(out, 1, 1) | np.roll(out, -1, 1)
    return out


# ====================================================================
# 3. Candidatos y seleccion por forma
# ====================================================================
def candidatos(B, area_imagen, max_regiones=3):
    """
    Extrae las regiones mayores de la mascara binaria y evalua cuan
    bien se ajusta cada una a la forma de un billete: se traza el
    contorno, se calcula la envolvente convexa y el rectangulo de area
    minima, y se puntua por relacion de aspecto, rectangularidad y
    plausibilidad de tamano.
    """
    etiq, n = E.etiquetar(_decimar(B))
    if n == 0:
        return []
    a = np.bincount(etiq.ravel(), minlength=n + 1)
    a[0] = 0

    salida = []
    for e in np.argsort(a)[::-1][:max_regiones]:
        if a[e] == 0:
            continue
        sel = _ampliar(etiq == e, B.shape) & B
        if sel.sum() < 64:
            continue
        cont = CT.trazar_contorno(sel)
        if cont is None or len(cont) < 8:
            continue
        hull = CV.envolvente_convexa(cont)
        if len(hull) < 3:
            continue
        punt, esquinas, diag = CT.puntaje_billete(hull, ASPECTO_BILLETE,
                                                  area_imagen)
        salida.append({"mascara": sel, "contorno": cont, "hull": hull,
                       "rect": esquinas, "puntaje": punt, "diag": diag})
    return sorted(salida, key=lambda d: -d["puntaje"])


def segmentar(img_rgb):
    """
    Cadena completa con seleccion entre mapas. Se prueban los mapas en
    orden de preferencia y se conserva globalmente la region de mayor
    puntaje de forma; si alguna supera el umbral de confianza se detiene
    la busqueda.
    """
    mapas = mapas_de_saliencia(img_rgb)
    area = float(img_rgb.shape[0] * img_rgb.shape[1])

    mejor = mejor_nombre = mejor_B = None
    for nombre in ("combinada", "cromatica", "brillo", "luminancia"):
        for radio in (12, 4):
            B = mascara_binaria(mapas[nombre], radio_cierre=radio)
            for cand in candidatos(B, area):
                if mejor is None or cand["puntaje"] > mejor["puntaje"]:
                    mejor, mejor_nombre, mejor_B = cand, nombre, B
            if mejor is not None and mejor["puntaje"] > 0.75:
                break
        if mejor is not None and mejor["puntaje"] > 0.75:
            break

    if mejor is None:
        raise ValueError("No se encontro ninguna region candidata.")

    T, _ = U.otsu(mapas[mejor_nombre], niveles=256, rango=(0.0, 1.0))
    return {"mapas": mapas, "mapa_elegido": mejor_nombre,
            "saliencia": mapas[mejor_nombre], "umbral": float(T),
            "binaria": mejor_B, "componente": mejor["mascara"],
            "contorno": mejor["contorno"], "hull": mejor["hull"],
            "rect_minimo": mejor["rect"], "puntaje_forma": mejor["puntaje"],
            "forma_diag": mejor["diag"]}


# ====================================================================
# 4. Rectificacion y aislamiento del billete
# ====================================================================
def _ordenar_horario(pts):
    """
    Ordena cuatro vertices en sentido horario comenzando por el
    superior izquierdo, usando el angulo polar respecto del centroide:

        theta_i = atan2( y_i - ybar, x_i - xbar )
    """
    P = np.asarray(pts, dtype=np.float64)
    c = P.mean(axis=0)
    P = P[np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))]
    return np.roll(P, -int(np.argmin(P.sum(axis=1))), axis=0)


def rectificar_billete(img_rgb, et, ancho=ANCHO_CANONICO, alto=ALTO_CANONICO):
    """
    Estima la homografia desde el cuadrilatero de area maxima inscrito
    en la envolvente y rectifica al marco canonico.

    Se usa el cuadrilatero inscrito y no el rectangulo de area minima
    porque bajo perspectiva el billete no es un rectangulo en la imagen
    sino un cuadrilatero general; el rectangulo sirve para puntuar la
    forma, no para rectificar.
    """
    quad = CV.cuadrilatero_desde_envolvente(et["hull"])
    esq = G.ordenar_para_horizontal(_ordenar_horario(quad))
    rect, Hm, err = G.rectificar(img_rgb, esq, ancho, alto)

    m = et["componente"].astype(np.float64)[:, :, None]
    m_rect, _, _ = G.rectificar(m, esq, ancho, alto)
    m_rect = m_rect[:, :, 0] > 0.5

    comp = et["componente"]
    mu, val, vec, razon, ang = G.ejes_principales(comp)
    diag = {"centroide": mu, "razon_pca": razon, "angulo_pca": ang,
            "area_mascara": float(comp.sum()),
            "area_cuadrilatero": float(G.area_poligono(esq)),
            "rectangularidad": float(G.area_poligono(esq)
                                     / max(CT.area_contorno(et["hull"]), 1.0)),
            "cobertura": float(comp.sum()) / float(comp.size),
            "puntaje_forma": et["puntaje_forma"],
            "aspecto_medido": et["forma_diag"].get("aspecto"),
            "mapa_elegido": et["mapa_elegido"]}
    return rect, m_rect, esq, Hm, err, diag


def segmentar_sin_fondo(rect_rgb, mascara_rect):
    """
    Billete aislado. Los pixeles fuera de la mascara se marcan como no
    validos, de modo que todo estadistico posterior se calcula
    unicamente sobre el soporte del billete y el fondo de la escena no
    puede influir en los descriptores ni en las medias comparadas.
    """
    out = rect_rgb.copy()
    out[~mascara_rect] = np.nan
    return out


def procesar(img_rgb):
    """Cadena completa. Devuelve todas las etapas."""
    et = segmentar(img_rgb)
    rect, m_rect, esq, Hm, err, diag = rectificar_billete(img_rgb, et)
    et.update({"rectificada": rect, "mascara_rect": m_rect,
               "sin_fondo": segmentar_sin_fondo(rect, m_rect),
               "esquinas": esq, "homografia": Hm,
               "error_reproyeccion": err, "diagnostico": diag})
    return et
