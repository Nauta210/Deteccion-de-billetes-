"""
Reglas de decision.

Este modulo sustituye por completo al clasificador entrenado. No hay
ninguna etapa de aprendizaje, ningun archivo de modelo y ningun
parametro ajustado a partir de datos por un algoritmo de optimizacion.
La decision se toma con dos reglas explicitas cuyas constantes estan
escritas en el codigo fuente y documentadas.

Por que reglas y no un clasificador
-----------------------------------
Un clasificador estimado a partir de un conjunto reducido tiene mas
parametros libres que muestras disponibles, de modo que su desempeno
aparente no es verificable. Una regla explicita, en cambio, es
auditable: ante cualquier decision puede senalarse la magnitud medida y
la constante con la que se comparo. Ademas, las constantes aqui
empleadas no son arbitrarias sino que corresponden a propiedades
fisicas del objeto: el color dominante de cada denominacion y la
energia de microtextura que aporta la impresion calcografica.

Las dos reglas son independientes entre si: la denominacion se decide
por color y la autenticidad por microtextura. Un error en una no
arrastra a la otra.
"""
import numpy as np

from . import color as C
from . import filtros as F


# ====================================================================
# Constantes de denominacion
# ====================================================================
# Matiz dominante de cada denominacion, en grados sobre el circulo
# cromatico, medido tras aplicar constancia de color. Corresponden al
# color caracteristico del diseno: el billete de cincuenta mil es
# violeta y el de cien mil verde azulado.
#
# Son constantes de diseno del algoritmo, del mismo modo que los pesos
# del operador de Sobel o los coeficientes de la luminancia: se fijan
# en el codigo y no se recalculan al ejecutar.
PROTOTIPOS_MATIZ = {
    "50mil": 270.0,      # violeta
    "100mil": 160.0,     # verde azulado
}

# Apertura angular admitida alrededor de cada prototipo. Fuera de ella
# la denominacion se declara no identificada en lugar de forzar la
# asignacion al prototipo mas cercano.
TOLERANCIA_MATIZ = 55.0

# Separacion angular minima entre el mejor y el segundo candidato. Si
# dos prototipos quedan casi a la misma distancia la decision no es
# confiable y se prefiere no decidir.
MARGEN_MINIMO = 15.0


# ====================================================================
# Constante de autenticidad
# ====================================================================
# Umbral sobre la varianza del laplaciano medida en el marco canonico.
#
# Origen del valor: sobre el conjunto disponible las dos poblaciones no
# se solapan. El ejemplar autentico de menor microtextura mide
# 3,34 x 10^-3 y la reproduccion de mayor microtextura 6,6 x 10^-4. El
# umbral se sitúa en la media geometrica de ambos extremos, que es el
# punto que maximiza el margen relativo a ambos lados cuando la
# magnitud se distribuye de forma multiplicativa:
#
#     tau = sqrt( 3,34e-3 x 6,6e-4 ) = 1,48e-3
#
# El margen resultante es de un factor 2,3 hacia cada lado.
#
# La magnitud se mide sobre el billete ya rectificado al marco
# canonico, de modo que es independiente de la distancia de captura:
# todos los billetes se comparan al mismo tamano en pixeles.
UMBRAL_LAPLACIANO = 1.48e-3

# Descriptor secundario, informativo. No participa en la decision pero
# se reporta porque permite detectar situaciones limite.
UMBRAL_ENERGIA_BORDES = 0.104


# ====================================================================
# Constancia de color
# ====================================================================
def balance_minkowski(rect_rgb, mascara, p=6.0):
    """
    Correccion del iluminante por la hipotesis de Minkowski.

    Para cada canal c se estima el iluminante mediante la norma p

        e_c = ( (1/|Omega|) * sum_{x in Omega} f_c(x)^p )^(1/p)

    y se corrige dividiendo por e_c. Con p = 1 se recupera la hipotesis
    del mundo gris y con p tendiendo a infinito la del parche blanco;
    el valor intermedio p = 6 es el que mejor se comporta en escenas
    con pocas superficies.

    El iluminante se estima UNICAMENTE sobre los pixeles del billete.
    Si se estimara sobre toda la imagen, el color de la mesa entraria
    en el calculo y la correccion dependeria de donde se apoyo el
    billete.

    Esta etapa es indispensable aqui: sin ella el matiz medido de un
    mismo billete se desplaza decenas de grados al cambiar la
    temperatura de color de la iluminacion, y ninguna constante fija
    seria valida.
    """
    salida = rect_rgb.copy()
    for c in range(3):
        canal = rect_rgb[:, :, c][mascara]
        if canal.size == 0:
            continue
        e = (np.mean(canal ** p)) ** (1.0 / p) + 1e-9
        salida[:, :, c] = rect_rgb[:, :, c] / e
    k = salida[mascara].mean() if mascara.any() else 1.0
    return np.clip(salida / max(k, 1e-9) * 128.0, 0.0, 255.0)


# ====================================================================
# Denominacion
# ====================================================================
def matiz_dominante(rect_rgb, mascara, s_min=0.10, v_min=0.12):
    """
    Matiz dominante por media circular.

    El matiz es una magnitud angular, de modo que su promedio
    aritmetico carece de sentido: la media de 350 y 10 grados no es 180
    sino 0. Se promedian los vectores unitarios correspondientes,

        Hbar = arg( (1/n) * sum_k exp( j * H_k ) )

    Solo se acumulan pixeles con saturacion y valor suficientes, porque
    el matiz queda indefinido cuando la saturacion tiende a cero y es
    ruidoso cuando el valor tiende a cero.

    Devuelve (matiz en grados, concentracion en [0,1]). La
    concentracion es el modulo del vector resultante: vale 1 si todos
    los pixeles comparten el matiz y 0 si estan uniformemente
    repartidos por el circulo.
    """
    H, S, V = C.rgb_a_hsv(rect_rgb)
    valido = mascara & (S >= s_min) & (V >= v_min)
    if valido.sum() < 500:
        return None, 0.0
    z = np.exp(1j * np.radians(H[valido]))
    r = z.mean()
    return float(np.degrees(np.angle(r)) % 360.0), float(abs(r))


def distancia_angular(a, b):
    """
    Diferencia minima entre dos angulos sobre el circulo, en grados:

        d(a,b) = min( |a-b| , 360 - |a-b| )
    """
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def decidir_denominacion(matiz):
    """
    Asigna la denominacion por minima distancia angular al prototipo,
    con rechazo por tolerancia y por margen.

    Devuelve (clase, confianza, distancias). La clase es None cuando
    ningun prototipo queda dentro de la tolerancia o cuando dos quedan
    demasiado cerca entre si.
    """
    if matiz is None:
        return None, 0.0, {}
    d = {c: distancia_angular(matiz, h) for c, h in PROTOTIPOS_MATIZ.items()}
    orden = sorted(d.items(), key=lambda kv: kv[1])
    mejor, d1 = orden[0]
    d2 = orden[1][1] if len(orden) > 1 else 180.0

    if d1 > TOLERANCIA_MATIZ:
        return None, 0.0, d
    if (d2 - d1) < MARGEN_MINIMO:
        return None, 0.0, d
    # Confianza: 1 en el prototipo exacto, 0 en el limite de tolerancia
    conf = float(np.clip(1.0 - d1 / TOLERANCIA_MATIZ, 0.0, 1.0))
    return mejor, conf, d


# ====================================================================
# Autenticidad
# ====================================================================
def medir_microtextura(rect_rgb, mascara):
    """
    Varianza del laplaciano y energia de bordes sobre el soporte del
    billete.

        lap I = d2I/dx2 + d2I/dy2
        V_L   = (1/|Omega|) * sum_x ( lap I(x) - media(lap I) )^2
        E     = media de |grad I| sobre Omega

    El laplaciano responde a las variaciones de segundo orden de la
    intensidad, es decir al detalle fino. La impresion calcografica de
    un billete genuino deja un relieve y una trama de linea continua
    que producen mucha energia en esa banda; una reproduccion impresa
    atraviesa un muestreo adicional que actua como filtro pasabajas y
    la reduce en uno o dos ordenes de magnitud.

    El fondo se excluye explicitamente: de otro modo la medida
    dependeria de la superficie sobre la que se tomo la fotografia.
    """
    g = C.rgb_a_gris(rect_rgb) / 255.0
    if mascara.sum() < 1000:
        return None, None
    lap = F.laplaciano(g)
    mag, _, _ = F.gradiente(g)
    return float(np.var(lap[mascara])), float(mag[mascara].mean())


def decidir_autenticidad(v_lap):
    """
    Regla explicita de decision:

        V_L < tau  ->  posible reproduccion impresa
        V_L >= tau ->  sin anomalias de microtextura

    Devuelve (es_reproduccion, holgura). La holgura es el cociente
    V_L / tau expresado en factores: un valor de 3 significa que el
    billete tiene el triple de microtextura que el umbral, y uno de
    0,3 que tiene menos de un tercio. Se reporta porque informa cuan
    lejos quedo la medida del limite, cosa que un simple si o no oculta.
    """
    if v_lap is None:
        return None, None
    return bool(v_lap < UMBRAL_LAPLACIANO), float(v_lap / UMBRAL_LAPLACIANO)


# ====================================================================
# Evaluacion
# ====================================================================
def matriz_confusion(y_real, y_pred, clases):
    idx = {c: i for i, c in enumerate(clases)}
    M = np.zeros((len(clases), len(clases)), dtype=int)
    for a, b in zip(y_real, y_pred):
        if a in idx and b in idx:
            M[idx[a], idx[b]] += 1
    return M


def metricas_binarias(y_real, y_pred):
    """Clase positiva = True (reproduccion)."""
    y_real = np.asarray(y_real, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    VP = int(np.sum(y_real & y_pred))
    VN = int(np.sum(~y_real & ~y_pred))
    FP = int(np.sum(~y_real & y_pred))
    FN = int(np.sum(y_real & ~y_pred))
    prec = VP / max(VP + FP, 1)
    sens = VP / max(VP + FN, 1)
    return {"VP": VP, "VN": VN, "FP": FP, "FN": FN,
            "exactitud": (VP + VN) / max(VP + VN + FP + FN, 1),
            "precision": prec, "sensibilidad": sens,
            "f1": 2 * prec * sens / max(prec + sens, 1e-9),
            "FNR": FN / max(FN + VP, 1), "FPR": FP / max(FP + VN, 1)}
