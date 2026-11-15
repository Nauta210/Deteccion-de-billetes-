"""
Analisis de una imagen individual con generacion de figuras por etapa.

Uso:
    python src/analizar.py <imagen> [--salida resultados/figuras]
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdi import io_utils as IO          # noqa: E402
from pdi import color as C              # noqa: E402
from pdi import segmentacion as SEG     # noqa: E402
from pdi import descriptores as D       # noqa: E402
from pdi import saliencia_billete as SB  # noqa: E402


def panel_etapas(img, et, destino):
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    ax = ax.ravel()

    ax[0].imshow(img.astype(np.uint8)); ax[0].set_title("(a) Original")
    ax[1].imshow(et["saliencia"], cmap="inferno")
    ax[1].set_title(f"(b) Saliencia (Mahalanobis + gradiente)")
    ax[2].imshow(et["binaria"], cmap="gray")
    ax[2].set_title(f"(c) Histeresis, T={et['umbral']:.3f}")
    ax[3].imshow(et["componente"], cmap="gray")
    ax[3].set_title("(d) Componente conexa mayor")

    ax[4].imshow(img.astype(np.uint8))
    if et.get("hull") is not None:
        h = np.vstack([et["hull"], et["hull"][:1]])
        ax[4].plot(h[:, 0], h[:, 1], "-", lw=1.2, color="#37e")
    q = np.vstack([et["esquinas"], et["esquinas"][:1]])
    ax[4].plot(q[:, 0], q[:, 1], "-o", lw=2, ms=6, color="#f33")
    ax[4].set_title("(e) Envolvente y cuadrilatero")

    ax[5].imshow(et["rectificada"].astype(np.uint8))
    ax[5].set_title("(f) Rectificada por homografia")

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def panel_espectro(rect, destino):
    g = C.rgb_a_gris(rect) / 255.0
    rois = D.extraer_rois(g)
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.4))

    for k in range(4):
        r = rois[k]
        ax[0, k].imshow(r, cmap="gray"); ax[0, k].set_title(f"ROI {k+1}")
        A = D.espectro_magnitud(r)
        ax[1, k].imshow(np.log1p(A), cmap="magma")
        razon, pico = D.energia_radial(r)
        ax[1, k].set_title(f"E={razon:.3f}  pico={pico:.0f}", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Regiones de interes y espectro de magnitud (escala log)")
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def panel_perfil(rect_aut, rect_rep, destino):
    """Compara el perfil radial de potencia de un autentico y una copia."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for rect, etq, est in ((rect_aut, "Autentico", "-"),
                           (rect_rep, "Reproduccion", "--")):
        if rect is None:
            continue
        g = C.rgb_a_gris(rect) / 255.0
        perfiles = [D.perfil_radial(r) for r in D.extraer_rois(g)]
        p = np.mean(perfiles, axis=0)
        rho = np.linspace(0, 1, len(p))
        ax.semilogy(rho, p / p.max(), est, lw=1.8, label=etq)
    ax.set_xlabel(r"Frecuencia radial normalizada $\rho$")
    ax.set_ylabel("Potencia relativa")
    ax.set_title("Perfil radial de potencia")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(destino, dpi=130); plt.close(fig)


def panel_saliencia_billete(rect, mascara, destino, rect2=None, masc2=None):
    """
    Mapas de saliencia calculados SOBRE EL BILLETE YA SEGMENTADO, es
    decir despues de retirar el fondo. Si se entrega un segundo
    ejemplar se muestran ambos para comparar.
    """
    casos = [("Auténtico", rect, mascara)]
    if rect2 is not None:
        casos.append(("Reproducción", rect2, masc2))

    calculados = [(etq, r, m, SB.mapas_del_billete(r, m, lado=32))
                  for etq, r, m in casos]

    # Escala comun. Si cada mapa se normalizara por separado, ambos
    # ocuparian todo el rango de color y la comparacion visual seria
    # enganosa: un billete sin microtextura se veria tan brillante
    # como uno que si la tiene. Se toma como referencia el maximo del
    # primer caso, que es el autentico.
    escalas = {}
    for clave in ("microtextura", "bordes", "saturacion"):
        v = calculados[0][3][clave]
        v = v[np.isfinite(v)]
        escalas[clave] = float(np.percentile(v, 98)) if v.size else 1.0

    fig, ax = plt.subplots(len(casos), 5,
                           figsize=(15, 3.1 * len(casos)), squeeze=False)
    for f, (etq, r, m, mp) in enumerate(calculados):
        sf = r.copy()
        sf[~m] = 255.0                    # fondo retirado, en blanco
        comb = np.nansum(np.dstack([
            0.50 * np.clip(mp["microtextura"] / max(escalas["microtextura"], 1e-12), 0, 1),
            0.30 * np.clip(mp["bordes"] / max(escalas["bordes"], 1e-12), 0, 1),
            0.20 * np.clip(mp["saturacion"] / max(escalas["saturacion"], 1e-12), 0, 1),
        ]), axis=2)
        comb[~np.isfinite(mp["microtextura"])] = np.nan

        paneles = [
            ("Billete sin fondo", sf, None, None),
            ("Microtextura", mp["microtextura"], "inferno", escalas["microtextura"]),
            ("Bordes", mp["bordes"], "inferno", escalas["bordes"]),
            ("Saturación", mp["saturacion"], "inferno", escalas["saturacion"]),
            ("Combinada", comb, "viridis", 1.0),
        ]
        for k, (titulo, dato, cmap, vmax) in enumerate(paneles):
            a = ax[f, k]
            if cmap is None:
                a.imshow(np.clip(dato, 0, 255).astype(np.uint8))
            else:
                a.imshow(dato, cmap=cmap, vmin=0, vmax=vmax)
            a.set_title(f"{etq} — {titulo}" if k == 0 else titulo,
                        fontsize=9)
            a.set_xticks([]); a.set_yticks([])
        print(f"   {etq:14s} microtextura={mp['promedios']['microtextura']:.5f}"
              f"  bordes={mp['promedios']['bordes']:.4f}"
              f"  saturacion={mp['promedios']['saturacion']:.4f}")
    fig.suptitle("Mapas de saliencia sobre el billete segmentado, sin "
                 "fondo. Ambas filas comparten escala de color.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagen")
    ap.add_argument("--comparar", default=None,
                    help="Imagen de reproduccion para el perfil radial")
    ap.add_argument("--salida", default="resultados/figuras")
    args = ap.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.imagen))[0]

    img = IO.reducir_lado_mayor(IO.cargar_rgb(args.imagen), 1024)
    et = SEG.procesar(img)

    panel_etapas(img, et, os.path.join(args.salida, f"etapas_{base}.png"))
    panel_espectro(et["rectificada"],
                   os.path.join(args.salida, f"espectro_{base}.png"))

    et2 = None
    if args.comparar:
        img2 = IO.reducir_lado_mayor(IO.cargar_rgb(args.comparar), 1024)
        et2 = SEG.procesar(img2)
        panel_perfil(et["rectificada"], et2["rectificada"],
                     os.path.join(args.salida, f"perfil_{base}.png"))

    print("\n  Mapas de saliencia sobre el billete sin fondo:")
    panel_saliencia_billete(
        et["rectificada"], et["mascara_rect"],
        os.path.join(args.salida, f"saliencia_billete_{base}.png"),
        et2["rectificada"] if et2 else None,
        et2["mascara_rect"] if et2 else None)

    d = et["diagnostico"]
    print(json.dumps({
        "umbral_otsu": round(et["umbral"], 4),
        "error_reproyeccion_px": round(et["error_reproyeccion"], 4),
        "rectangularidad": round(d["rectangularidad"], 4),
        "cobertura": round(d["cobertura"], 4),
        "razon_pca": round(d["razon_pca"], 3),
        "esquinas": et["esquinas"].astype(int).tolist(),
        "homografia": np.round(et["homografia"], 6).tolist(),
    }, indent=2))
    print(f"\nFiguras en {args.salida}/")


if __name__ == "__main__":
    main()
