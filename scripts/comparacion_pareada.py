"""
Comparacion pareada dentro de un mismo encuadre.

Por que este experimento es distinto
------------------------------------
Todas las comparaciones anteriores contrastaban fotografias tomadas en
momentos, lugares y con camaras distintas. En ese diseno, cualquier
diferencia medida entre clases puede provenir de la impresion o de la
condicion de captura, y no hay forma de separarlas.

Aqui los dos billetes aparecen en la MISMA fotografia. Comparten
exactamente el mismo sensor, la misma exposicion, el mismo balance de
blancos, la misma distancia focal, el mismo instante y la misma
iluminacion. La condicion de captura queda controlada por diseno, no
por argumentacion. Toda diferencia que persista es atribuible al
objeto.

Es el equivalente fotografico de un experimento pareado: en lugar de
comparar dos grupos independientes se compara dentro de cada unidad
experimental, lo que elimina la varianza entre unidades.

Identificacion de cada billete
------------------------------
Las reproducciones didacticas de curso legal deben diferir en tamano
del billete que representan. Dentro de un mismo encuadre la escala es
comun, de modo que el area en pixeles es directamente comparable y la
region mayor corresponde al ejemplar autentico. El script reporta la
razon de tamanos para que esa identificacion quede verificable y no
sea una suposicion.

Uso:
    python scripts/comparacion_pareada.py <imagen> [imagen2 ...]
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))
sys.path.insert(0, os.path.join(RAIZ, "src"))

from pdi import io_utils as IO          # noqa: E402
from pdi import color as C              # noqa: E402
from pdi import filtros as F            # noqa: E402
from pdi import geometria as G          # noqa: E402
from pdi import convexa as CV           # noqa: E402
from pdi import contornos as CT         # noqa: E402
from pdi import segmentacion as SEG     # noqa: E402

MEDIDAS = [
    ("media", "Media de intensidad"),
    ("desviacion", "Desviacion estandar"),
    ("energia_borde", "Energia de bordes"),
    ("nitidez", "Varianza del laplaciano"),
    ("saturacion", "Saturacion media"),
    ("agudeza_borde", "Agudeza del borde fisico"),
]


def toca_borde(mascara, fraccion=0.015):
    """
    Indica si la region alcanza el borde del cuadro.

    El margen se toma proporcional al tamano de la imagen y no como un
    numero fijo de pixeles, porque un billete cortado por el encuadre
    suele quedar a unos pocos pixeles del limite y esa distancia
    escala con la resolucion.
    """
    m = mascara
    k = max(3, int(round(fraccion * min(m.shape))))
    return bool(m[:k, :].any() or m[-k:, :].any()
                or m[:, :k].any() or m[:, -k:].any())


ASPECTO_MIN, ASPECTO_MAX = 1.45, 2.80
SALTO_MIN = 0.04


def dos_regiones(img_rgb, puntaje_min=0.35):
    """
    Localiza las dos regiones del cuadro que mejor se ajustan a la
    forma de un billete, probando los distintos mapas de saliencia y
    quedandose con las dos de mayor puntaje que no se solapen.
    """
    mapas = SEG.mapas_de_saliencia(img_rgb)
    area = float(img_rgb.shape[0] * img_rgb.shape[1])

    # Se explora ademas una rejilla de escalas morfologicas.
    #
    # Con dos billetes contiguos en el mismo encuadre la separacion
    # entre ellos puede ser de pocos pixeles. El cierre con elemento
    # de radio r une regiones separadas por menos de 2r, de modo que
    # un cierre amplio los fusiona en una sola region. La apertura
    # hace lo contrario: erosiona los puentes delgados y los separa,
    # a costa de redondear las esquinas.
    #
    # No existe un par de radios correcto para toda imagen, porque la
    # separacion depende de como se colocaron los billetes. Se generan
    # candidatos con varias combinaciones y se deja que el puntaje de
    # forma decida cual region conservar, que es el mismo criterio con
    # que se elige entre mapas de saliencia.
    todos = []
    for nombre in ("cromatica", "combinada", "brillo", "luminancia"):
        for r_ap, r_ci in ((3, 12), (6, 2), (12, 2), (16, 2)):
            B = SEG.mascara_binaria(mapas[nombre], radio_morf=r_ap,
                                    radio_cierre=r_ci)
            for c in SEG.candidatos(B, area, max_regiones=4):
                c["mapa"] = f"{nombre}/a{r_ap}c{r_ci}"
                todos.append(c)
    todos.sort(key=lambda d: -d["puntaje"])

    elegidas = []
    for c in todos:
        if c["puntaje"] < puntaje_min:
            continue
        # Se rechazan las regiones truncadas por el borde del cuadro:
        # un billete cortado no tiene sus cuatro esquinas, de modo que
        # la homografia se estimaria sobre un cuadrilatero que no
        # corresponde al objeto y todas las medidas quedarian sesgadas.
        if toca_borde(c["mascara"]):
            continue
        # La relacion de aspecto de un billete completo visto de forma
        # aproximadamente frontal debe acercarse a la nominal de
        # 140:66 = 2.12. Una region que se aparta mucho corresponde a
        # un billete truncado por el encuadre, a dos billetes
        # fusionados o a un objeto que no es un billete, y medir sobre
        # ella produciria cifras sin sentido.
        if not (ASPECTO_MIN <= c["diag"]["aspecto"] <= ASPECTO_MAX):
            continue
        # Se descartan las que se solapan con una ya elegida
        if any((c["mascara"] & e["mascara"]).sum()
               > 0.25 * min(c["mascara"].sum(), e["mascara"].sum())
               for e in elegidas):
            continue
        elegidas.append(c)
        if len(elegidas) == 2:
            break
    return elegidas


def medir(img_rgb, cand):
    """Rectifica una region y calcula las medidas sobre el billete."""
    quad = CV.cuadrilatero_desde_envolvente(cand["hull"])
    esq = G.ordenar_para_horizontal(SEG._ordenar_horario(quad))
    rect, _, err = G.rectificar(img_rgb, esq,
                                SEG.ANCHO_CANONICO, SEG.ALTO_CANONICO)
    m = cand["mascara"].astype(np.float64)[:, :, None]
    m_rect, _, _ = G.rectificar(m, esq, SEG.ANCHO_CANONICO, SEG.ALTO_CANONICO)
    m_rect = m_rect[:, :, 0] > 0.5
    if m_rect.sum() < 1000:
        return None, None

    g = C.rgb_a_gris(rect) / 255.0
    x = g[m_rect]
    mag, _, _ = F.gradiente(g)
    lap = F.laplaciano(g)
    _, S, _ = C.rgb_a_hsv(rect)

    # Agudeza del borde fisico, en la imagen original
    gimg = C.rgb_a_gris(img_rgb) / 255.0
    magi, _, _ = F.gradiente(gimg)
    from pdi import umbral as U
    ee = U.elemento_elipse(3)
    mk = cand["mascara"]
    banda = U.dilatar(mk, ee) & ~U.erosionar(mk, ee)
    dentro = gimg[U.erosionar(mk, ee)].mean()
    fuera = gimg[~U.dilatar(mk, U.elemento_elipse(11))].mean()
    salto = abs(dentro - fuera)
    # Validacion del contorno. Si la intensidad media dentro y fuera
    # de la region son casi iguales, la frontera detectada no
    # corresponde a un borde fisico del objeto sino a un limite
    # arbitrario dentro de una zona homogenea. Medir la agudeza en ese
    # caso divide por un numero casi nulo y produce valores enormes
    # sin significado. La region se rechaza.
    if salto < SALTO_MIN:
        return None, None
    agudeza = float(magi[banda].mean() / salto)

    med = {
        "media": float(x.mean()),
        "desviacion": float(x.std()),
        "energia_borde": float(mag[m_rect].mean()),
        "nitidez": float(np.var(lap[m_rect])),
        "saturacion": float(S[m_rect].mean()),
        "agudeza_borde": agudeza,
        "area_px": float(mk.sum()),
        "lado_mayor_px": float(max(CT.rectangulo_area_minima(cand["hull"])[1:3])),
        "aspecto": float(cand["diag"]["aspecto"]),
        "puntaje_forma": float(cand["puntaje"]),
        "error_reproyeccion": float(err),
    }
    return med, rect


def procesar_imagen(ruta):
    img = IO.cargar_rgb(ruta)
    img = IO.reducir_lado_mayor(img, 1600)
    regs = dos_regiones(img)
    if len(regs) < 2:
        print(f"  Solo se hallaron {len(regs)} regiones completas "
              f"(sin truncar por el borde). Se omite esta imagen.")
        return None

    resultados = []
    for c in regs:
        med, rect = medir(img, c)
        if med is None:
            print("  Una region no presenta un borde fisico valido. "
                  "Se omite esta imagen.")
            return None
        resultados.append((med, rect))

    # El de mayor lado es el autentico: la reproduccion didactica debe
    # diferir en tamano y aqui la escala es comun.
    resultados.sort(key=lambda t: -t[0]["lado_mayor_px"])
    aut, rep = resultados[0], resultados[1]
    razon = rep[0]["lado_mayor_px"] / aut[0]["lado_mayor_px"]
    return {"autentico": aut[0], "didactico": rep[0],
            "razon_tamano": razon,
            "rect_aut": aut[1], "rect_rep": rep[1],
            "archivo": os.path.basename(ruta)}


def main():
    rutas = sys.argv[1:]
    if not rutas:
        print(__doc__)
        sys.exit(1)

    pares = []
    for r in rutas:
        print(f"\n{os.path.basename(r)}")
        p = procesar_imagen(r)
        if p is None:
            continue
        pares.append(p)
        a, d = p["autentico"], p["didactico"]
        print(f"  lado mayor    autentico {a['lado_mayor_px']:7.1f} px   "
              f"didactico {d['lado_mayor_px']:7.1f} px   "
              f"razon {p['razon_tamano']:.3f}")
        print(f"  {'medida':<26}{'autentico':>12}{'didactico':>12}"
              f"{'cociente':>11}")
        for clave, nombre in MEDIDAS:
            va, vd = a[clave], d[clave]
            coc = vd / va if abs(va) > 1e-12 else float("nan")
            print(f"  {nombre:<26}{va:>12.5f}{vd:>12.5f}{coc:>11.2f}")

    if not pares:
        print("\nNo se pudo analizar ninguna imagen.")
        return

    # Figura: recortes rectificados y barras pareadas
    n = len(pares)
    fig = plt.figure(figsize=(13, 3.1 * n + 3.4))
    for k, p in enumerate(pares):
        for j, (clave, titulo) in enumerate((("rect_aut", "Autentico"),
                                             ("rect_rep", "Didactico"))):
            a = fig.add_subplot(n + 1, 2, 2 * k + j + 1)
            a.imshow(np.clip(p[clave], 0, 255).astype(np.uint8))
            a.set_title(f"{p['archivo']} — {titulo}", fontsize=9)
            a.set_xticks([]); a.set_yticks([])

    ax = fig.add_subplot(n + 1, 1, n + 1)
    etiquetas = [nom for _, nom in MEDIDAS]
    va = [np.mean([p["autentico"][c] for p in pares]) for c, _ in MEDIDAS]
    vd = [np.mean([p["didactico"][c] for p in pares]) for c, _ in MEDIDAS]
    # Normalizado al autentico, para poder mostrar escalas distintas juntas
    rel = [d / a if abs(a) > 1e-12 else np.nan for a, d in zip(va, vd)]
    x = np.arange(len(etiquetas))
    ax.bar(x, np.ones_like(x, dtype=float), width=0.38, label="Autentico",
           color="#5b8def", alpha=0.85)
    ax.bar(x + 0.38, rel, width=0.38, label="Didactico",
           color="#ef6f5b", alpha=0.85)
    ax.axhline(1.0, color="#333", lw=0.8)
    ax.set_xticks(x + 0.19)
    ax.set_xticklabels(etiquetas, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Valor relativo al autentico")
    ax.set_title("Comparacion pareada en el mismo encuadre "
                 "(condicion de captura identica)", fontsize=11)
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    for i, v in enumerate(rel):
        ax.text(i + 0.38, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    destino = os.path.join(RAIZ, "resultados", "figuras")
    os.makedirs(destino, exist_ok=True)
    fig.savefig(os.path.join(destino, "comparacion_pareada.png"), dpi=130)
    plt.close(fig)

    salida = {"pares": [{k: v for k, v in p.items()
                         if not k.startswith("rect_")} for p in pares]}
    with open(os.path.join(RAIZ, "resultados",
                           "comparacion_pareada.json"), "w") as f:
        json.dump(salida, f, indent=2)
    print("\nFigura en resultados/figuras/comparacion_pareada.png")


if __name__ == "__main__":
    main()
