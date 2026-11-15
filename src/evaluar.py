"""
Evaluacion de las reglas de decision sobre el banco de datos.

Este script NO entrena nada. Las constantes estan fijadas en
src/pdi/reglas.py y no se modifican aqui: lo unico que se hace es
aplicarlas a cada imagen y contabilizar aciertos y errores.

Uso:
    python src/evaluar.py
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdi import io_utils as IO          # noqa: E402
from pdi import segmentacion as SEG     # noqa: E402
from pdi import reglas as R             # noqa: E402

LADO_TRABAJO = 1024


def procesar(ruta):
    img = IO.reducir_lado_mayor(IO.cargar_rgb(ruta), LADO_TRABAJO)
    et = SEG.procesar(img)
    d = et["diagnostico"]
    ok = (d["puntaje_forma"] >= 0.40
          and d["rectangularidad"] > 0.80
          and et["error_reproyeccion"] < 1.0)

    rect, m = et["rectificada"], et["mascara_rect"]
    if m.sum() < 1000:
        return {"ok": False, "diag": d}

    wb = R.balance_minkowski(rect, m)
    matiz, conc = R.matiz_dominante(wb, m)
    clase, conf, _ = R.decidir_denominacion(matiz)
    v_lap, energia = R.medir_microtextura(rect, m)
    es_rep, holgura = R.decidir_autenticidad(v_lap)

    return {"ok": bool(ok), "diag": d, "matiz": matiz,
            "concentracion": conc, "clase": clase, "confianza": conf,
            "v_lap": v_lap, "energia": energia,
            "es_reproduccion": es_rep, "holgura": holgura}


def recorrer(carpeta):
    salida = {}
    if not os.path.isdir(carpeta):
        return salida
    for clase in sorted(os.listdir(carpeta)):
        sub = os.path.join(carpeta, clase)
        if os.path.isdir(sub):
            r = IO.listar_imagenes(sub)
            if r:
                salida[clase] = r
    return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="datos")
    ap.add_argument("--salida", default="resultados")
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  EVALUACION DE LAS REGLAS DE DECISION")
    print("=" * 70)
    print(f"  Prototipos de matiz : {R.PROTOTIPOS_MATIZ}")
    print(f"  Tolerancia angular  : {R.TOLERANCIA_MATIZ}  grados")
    print(f"  Umbral de laplaciano: {R.UMBRAL_LAPLACIANO:.2e}")
    print("  (constantes fijas en src/pdi/reglas.py; no se ajustan aqui)")
    print("=" * 70)

    grupos = [("autentico", recorrer(os.path.join(args.datos, "autentico"))),
              ("reproduccion", recorrer(os.path.join(args.datos,
                                                     "reproduccion")))]

    registros = []
    print("\n[1/3] Procesando imagenes")
    for etq, g in grupos:
        for clase, rutas in g.items():
            for r in rutas:
                try:
                    d = procesar(r)
                except Exception as e:
                    print(f"   fallo {os.path.basename(r)[:30]}: {e}")
                    continue
                d.update({"origen": etq, "clase_real": clase,
                          "archivo": os.path.basename(r)})
                registros.append(d)
                if d["ok"]:
                    print(f"   {etq:12s} {clase:7s} "
                          f"{d['archivo'][:30]:32s} "
                          f"matiz={d['matiz'] or 0:6.1f} "
                          f"V_L={d['v_lap']:.5f} "
                          f"x{d['holgura']:5.2f}  "
                          f"{'REPRODUCCION' if d['es_reproduccion'] else 'sin anomalias'}")
                else:
                    print(f"   {etq:12s} {clase:7s} "
                          f"{d['archivo'][:30]:32s} SEGMENTACION RECHAZADA")

    n = len(registros)
    n_ok = sum(1 for d in registros if d["ok"])
    print(f"\n   Segmentacion valida: {n_ok}/{n} ({100*n_ok/max(n,1):.1f} %)")

    validos = [d for d in registros if d["ok"]]

    # ---------------- Denominacion -------------------------------
    print("\n[2/3] Denominacion")
    conocidas = set(R.PROTOTIPOS_MATIZ)
    sub = [d for d in validos if d["clase_real"] in conocidas]
    clases = sorted(conocidas)
    etiquetas = clases + ["rechazo"]
    y_r = [d["clase_real"] for d in sub]
    y_p = [d["clase"] if d["clase"] else "rechazo" for d in sub]
    MC = R.matriz_confusion(y_r, y_p, etiquetas)
    ac = float(np.trace(MC[:len(clases), :len(clases)])) / max(len(sub), 1)
    print(f"   filas=real, columnas=predicho -> {etiquetas}")
    print(MC)
    print(f"   Exactitud: {100*ac:.1f} %  (n={len(sub)})")

    fuera = [d for d in validos if d["clase_real"] not in conocidas]
    if fuera:
        rech = sum(1 for d in fuera if d["clase"] is None)
        print(f"   Denominaciones no modeladas: {len(fuera)} imagenes, "
              f"{rech} rechazadas correctamente "
              f"({100*rech/len(fuera):.0f} %)")

    # ---------------- Autenticidad -------------------------------
    print("\n[3/3] Autenticidad")
    y_real = [d["origen"] == "reproduccion" for d in validos]
    y_pred = [bool(d["es_reproduccion"]) for d in validos]
    met = R.metricas_binarias(y_real, y_pred)
    print(f"   VP={met['VP']}  VN={met['VN']}  "
          f"FP={met['FP']}  FN={met['FN']}")
    print(f"   Exactitud {100*met['exactitud']:.1f} %   "
          f"F1 {met['f1']:.3f}   FNR {met['FNR']:.3f}   "
          f"FPR {met['FPR']:.3f}")

    a = np.array([d["v_lap"] for d in validos if d["origen"] == "autentico"])
    b = np.array([d["v_lap"] for d in validos
                  if d["origen"] == "reproduccion"])
    if a.size and b.size:
        print(f"\n   V_L autentico    : [{a.min():.5f}, {a.max():.5f}]")
        print(f"   V_L reproduccion : [{b.min():.5f}, {b.max():.5f}]")
        print(f"   umbral           : {R.UMBRAL_LAPLACIANO:.5f}")
        if b.max() < a.min():
            print(f"   Las poblaciones NO se solapan. "
                  f"Margen: factor {a.min()/b.max():.1f}")

    res = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "constantes": {"prototipos_matiz": R.PROTOTIPOS_MATIZ,
                       "tolerancia_matiz": R.TOLERANCIA_MATIZ,
                       "umbral_laplaciano": R.UMBRAL_LAPLACIANO},
        "n_imagenes": n, "segmentacion_ok": n_ok,
        "tasa_segmentacion": n_ok / max(n, 1),
        "matriz_confusion_denominacion": MC.tolist(),
        "etiquetas_confusion": etiquetas,
        "exactitud_denominacion": ac, "n_denominacion": len(sub),
        "autenticidad": {k: float(v) for k, v in met.items()},
        "rango_vlap_autentico": [float(a.min()), float(a.max())] if a.size else None,
        "rango_vlap_reproduccion": [float(b.min()), float(b.max())] if b.size else None,
    }
    with open(os.path.join(args.salida, "resultados.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\nListo en {time.time()-t0:.1f} s. "
          f"Resultados en {args.salida}/resultados.json")
    print("No se genero ningun archivo de modelo: no hay entrenamiento.")


if __name__ == "__main__":
    main()
