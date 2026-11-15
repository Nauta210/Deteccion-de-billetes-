"""
Prototipo de escritorio.

Aplicacion con interfaz grafica que analiza una fotografia de un
billete y comunica el resultado por pantalla y por voz.

ALCANCE DE ESTE PROTOTIPO
-------------------------
La entrada es una fotografia ya tomada, que el usuario selecciona
desde el disco. La captura en vivo no forma parte de esta etapa: el
objetivo aqui es validar la cadena de procesamiento y las reglas de
decision sobre imagenes reales, de forma reproducible y verificable
por terceros.

La captura en tiempo real se abordara en la etapa siguiente, sobre
dispositivo movil, que es donde el sistema tiene sentido practico: la
verificacion ocurre en el lugar de la transaccion, no frente a un
computador. Esa decision es deliberada y no una limitacion tecnica del
algoritmo, que opera igual sobre cualquier fotografia.

NO HAY ENTRENAMIENTO
--------------------
La decision se toma con reglas explicitas cuyas constantes estan
escritas en src/pdi/reglas.py. No existe etapa de aprendizaje ni
archivo de modelo: la aplicacion se ejecuta directamente.

NO HAY LIBRERIAS DE VISION ARTIFICIAL
-------------------------------------
Las dependencias son NumPy, Pillow y Matplotlib. Toda la segmentacion,
la rectificacion y la descripcion se implementan en src/pdi/ a partir
de sus definiciones matematicas.

Modos de ejecucion
------------------
    python src/prototipo_pc.py
        Interfaz grafica. Es el modo normal.

    python src/prototipo_pc.py --imagen <ruta>
        Analiza una fotografia e imprime el resultado en la terminal,
        sin abrir ventana. Util para verificacion automatizada.
"""
import os
import sys
import glob
import queue
import argparse
import threading
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))
sys.path.insert(0, AQUI)

from pdi import io_utils as IO          # noqa: E402
from pdi import segmentacion as SEG     # noqa: E402
from pdi import reglas as R             # noqa: E402

LADO_TRABAJO = 1024

ETIQUETAS = {"50mil": "cincuenta mil pesos", "100mil": "cien mil pesos",
             "20mil": "veinte mil pesos", "10mil": "diez mil pesos",
             "5mil": "cinco mil pesos", "2mil": "dos mil pesos"}

GUIA = ("Seleccione la fotografía de un billete. Debe estar plano, "
        "fotografiado de frente, entero y sobre un fondo liso de color "
        "distinto al billete.")


# ====================================================================
# Sintesis de voz
# ====================================================================
class Voz:
    """
    Salida hablada. Se intenta pyttsx3, que funciona en los tres
    sistemas, y en Windows el sintetizador SAPI por linea de comandos.
    Si ninguno esta disponible el programa continua en silencio sin
    fallar: la voz complementa la salida en pantalla y no es un
    requisito de funcionamiento.
    """

    def __init__(self):
        self.motor = None
        self.modo = "silencio"
        try:
            import pyttsx3
            self.motor = pyttsx3.init()
            self.motor.setProperty("rate", 170)
            for v in self.motor.getProperty("voices"):
                if "spanish" in v.name.lower() or "es" in str(v.id).lower():
                    self.motor.setProperty("voice", v.id)
                    break
            self.modo = "pyttsx3"
        except Exception:
            if sys.platform.startswith("win"):
                self.modo = "sapi"

    def decir(self, texto):
        if self.modo == "silencio":
            return
        threading.Thread(target=self._decir, args=(texto,),
                         daemon=True).start()

    def _decir(self, texto):
        try:
            if self.modo == "pyttsx3":
                self.motor.say(texto)
                self.motor.runAndWait()
            elif self.modo == "sapi":
                import subprocess
                seguro = texto.replace("'", " ")
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Add-Type -AssemblyName System.Speech; "
                     "$s = New-Object System.Speech.Synthesis."
                     f"SpeechSynthesizer; $s.Speak('{seguro}')"],
                    capture_output=True, timeout=30)
        except Exception:
            pass


# ====================================================================
# Analisis
# ====================================================================
def revisar_encuadre(diag, err):
    """
    Verificacion previa al veredicto.

    Es preferible pedir otra fotografia a dictaminar sobre una imagen
    que la cadena no puede rectificar bien. La relacion de aspecto es
    el indicador mas informativo: la homografia supone que el objeto es
    plano, de modo que un billete plegado o fotografiado muy
    oblicuamente produce un cuadrilatero cuya relacion se aparta de la
    nominal de 140 a 66.
    """
    kappa = diag["cobertura"]
    asp = diag.get("aspecto_medido")

    if kappa < 0.05:
        return ("No encuentro el billete en la fotografía. Debe verse "
                "sobre un fondo liso de color distinto.")
    if kappa < 0.15:
        return "El billete se ve muy pequeño dentro de la fotografía."
    if kappa > 0.92:
        return ("El billete llena toda la fotografía. Hace falta algo de "
                "margen alrededor.")
    if diag["rectangularidad"] < 0.80:
        return ("El billete se ve arrugado o incompleto. Debe estar "
                "extendido y entero dentro del cuadro.")
    if asp is not None and not (1.45 <= asp <= 2.80):
        return ("El billete no se ve plano y de frente. La fotografía "
                "debe tomarse desde arriba, con el billete extendido.")
    if diag.get("puntaje_forma", 1.0) < 0.40:
        return ("No reconozco la forma de un billete en la fotografía.")
    if err > 1.5:
        return "La fotografía salió movida."
    return None


def analizar(img_rgb):
    """
    Cadena completa sobre una imagen. Aplica las reglas explicitas de
    src/pdi/reglas.py; no carga ningun modelo porque no hay ninguno.
    """
    img = IO.reducir_lado_mayor(img_rgb, LADO_TRABAJO)
    try:
        et = SEG.procesar(img)
    except Exception:
        return {"estado": "encuadre", "rectificada": None, "original": img,
                "mensaje": "No encuentro el billete en la fotografía."}

    diag = et["diagnostico"]
    aviso = revisar_encuadre(diag, et["error_reproyeccion"])
    if aviso:
        return {"estado": "encuadre", "mensaje": aviso, "original": img,
                "rectificada": et["rectificada"], "diagnostico": diag}

    rect, mascara = et["rectificada"], et["mascara_rect"]

    # --- Regla 1: denominacion por matiz, tras constancia de color --
    wb = R.balance_minkowski(rect, mascara)
    matiz, concentracion = R.matiz_dominante(wb, mascara)
    clase, conf, _ = R.decidir_denominacion(matiz)

    # --- Regla 2: autenticidad por microtextura ---------------------
    v_lap, energia = R.medir_microtextura(rect, mascara)
    es_rep, holgura = R.decidir_autenticidad(v_lap)

    diag = dict(diag)
    diag.update({"matiz": matiz, "concentracion": concentracion,
                 "v_lap": v_lap, "energia_bordes": energia,
                 "holgura": holgura})

    if clase is None:
        return {"estado": "duda", "rectificada": rect, "original": img,
                "diagnostico": diag,
                "mensaje": "No logro identificar la denominación. Puede "
                           "ser un billete de otra denominación, o la "
                           "iluminación es insuficiente."}

    nombre = ETIQUETAS.get(clase, clase)
    partes = [f"Billete de {nombre}."]
    if es_rep:
        estado = "alerta"
        partes.append("Atención: posible reproducción impresa. "
                      "Verifique manualmente.")
    else:
        estado = "ok"
        partes.append("No se detectaron anomalías de impresión.")
    partes.append("Este resultado no confirma autenticidad. Aplique el "
                  "procedimiento del Banco de la República: toque el alto "
                  "relieve, mire el billete al trasluz y gírelo para ver "
                  "el cambio de color.")

    return {"estado": estado, "denominacion": clase, "nombre": nombre,
            "confianza": float(conf), "v_lap": v_lap, "holgura": holgura,
            "mensaje": " ".join(partes), "rectificada": rect,
            "original": img, "diagnostico": diag}


def imprimir(res):
    print("\n" + "=" * 62)
    print(f"  ESTADO: {res['estado'].upper()}")
    print("=" * 62)
    print(f"  {res['mensaje']}")
    d = res.get("diagnostico")
    if d:
        print(f"\n  SEGMENTACION")
        print(f"    mapa de saliencia  : {d.get('mapa_elegido')}")
        print(f"    puntaje de forma   : {d.get('puntaje_forma', 0):.3f}")
        print(f"    relacion de aspecto: {d.get('aspecto_medido', 0):.2f}")
        print(f"    rectangularidad    : {d.get('rectangularidad', 0):.3f}")
        print(f"    cobertura          : {d.get('cobertura', 0):.3f}")
        if d.get("matiz") is not None:
            print(f"\n  REGLA 1 - denominacion por matiz")
            print(f"    matiz dominante    : {d['matiz']:.1f} grados")
            print(f"    prototipos         : {R.PROTOTIPOS_MATIZ}")
            print(f"    tolerancia         : {R.TOLERANCIA_MATIZ} grados")
        if d.get("v_lap") is not None:
            print(f"\n  REGLA 2 - autenticidad por microtextura")
            print(f"    varianza laplaciano: {d['v_lap']:.5f}")
            print(f"    umbral             : {R.UMBRAL_LAPLACIANO:.5f}")
            print(f"    holgura            : x{d['holgura']:.2f}"
                  "   (>1 sin anomalias, <1 reproduccion)")
    print("=" * 62 + "\n")


def ejemplos():
    """Imagenes del banco de datos, para probar sin buscar archivos."""
    lista = []
    for etq, patron in (("auténtico", "datos/autentico/*/*.jpg"),
                        ("reproducción", "datos/reproduccion/*/*.jpg")):
        for r in sorted(glob.glob(os.path.join(RAIZ, patron)))[:6]:
            lista.append((etq, os.path.basename(os.path.dirname(r)),
                          os.path.basename(r), r))
    return lista


# ====================================================================
# Interfaz grafica
# ====================================================================
def lanzar_gui():
    import tkinter as tk
    from tkinter import filedialog
    from PIL import Image, ImageTk

    FONDO, PANEL = "#12151c", "#1b2029"
    TEXTO, TENUE = "#eef2f8", "#93a0b4"
    OK, ALERTA, ACENTO = "#33c481", "#ff8c42", "#4da3ff"

    raiz = tk.Tk()
    raiz.title("Verificador de billetes colombianos — UMNG")
    raiz.configure(bg=FONDO)
    raiz.geometry("1200x720")
    raiz.minsize(1000, 640)

    voz = Voz()
    cola = queue.Queue()
    estado = {"analizando": False, "ultimo_mensaje": GUIA}

    izq = tk.Frame(raiz, bg=FONDO)
    izq.pack(side="left", fill="both", expand=True, padx=(14, 7), pady=14)
    der = tk.Frame(raiz, bg=PANEL, width=440)
    der.pack(side="right", fill="both", padx=(7, 14), pady=14)
    der.pack_propagate(False)

    tk.Label(izq, text="Fotografía analizada", bg=FONDO, fg=TENUE,
             font=("Segoe UI", 10), anchor="w").pack(fill="x")
    lienzo = tk.Label(izq, bg="#0a0c11", fg=TENUE, font=("Segoe UI", 12),
                      text="Pulse «Seleccionar imagen» para comenzar.")
    lienzo.pack(fill="both", expand=True, pady=(4, 8))

    tk.Label(izq, text="Billete rectificado al marco canónico", bg=FONDO,
             fg=TENUE, font=("Segoe UI", 10), anchor="w").pack(fill="x")
    lbl_rect = tk.Label(izq, bg="#0a0c11", height=9)
    lbl_rect.pack(fill="x", pady=(4, 10))

    barra = tk.Frame(izq, bg=FONDO)
    barra.pack(fill="x")

    tk.Label(der, text="Resultado", bg=PANEL, fg=TENUE,
             font=("Segoe UI", 10)).pack(anchor="w", padx=16, pady=(16, 2))
    lbl_titulo = tk.Label(der, text="Listo", bg=PANEL, fg=TEXTO,
                          font=("Segoe UI", 20, "bold"), anchor="w")
    lbl_titulo.pack(anchor="w", padx=16)
    lbl_msg = tk.Message(der, text=GUIA, bg=PANEL, fg=TEXTO, width=400,
                         font=("Segoe UI", 11), anchor="w", justify="left")
    lbl_msg.pack(anchor="w", padx=16, pady=(8, 12))

    tk.Label(der, text="Diagnóstico", bg=PANEL, fg=TENUE,
             font=("Segoe UI", 10)).pack(anchor="w", padx=16)
    txt_diag = tk.Label(der, text="—", bg=PANEL, fg=TENUE, anchor="w",
                        font=("Consolas", 9), justify="left")
    txt_diag.pack(anchor="w", padx=16, pady=(2, 10))

    tk.Label(der, text="Ejemplos del banco de datos", bg=PANEL, fg=TENUE,
             font=("Segoe UI", 10)).pack(anchor="w", padx=16, pady=(4, 2))
    marco_lista = tk.Frame(der, bg=PANEL)
    marco_lista.pack(fill="both", expand=True, padx=16)
    scroll = tk.Scrollbar(marco_lista)
    scroll.pack(side="right", fill="y")
    lista = tk.Listbox(marco_lista, bg="#0f131a", fg=TEXTO, relief="flat",
                       font=("Consolas", 9), selectbackground=ACENTO,
                       yscrollcommand=scroll.set, activestyle="none")
    lista.pack(side="left", fill="both", expand=True)
    scroll.config(command=lista.yview)

    muestras = ejemplos()
    for etq, den, arch, _ in muestras:
        lista.insert("end", f"{etq[:4]}  {den:7s} {arch[:26]}")

    tk.Label(der, bg=PANEL, fg=TENUE, font=("Segoe UI", 8), justify="left",
             wraplength=400, anchor="w",
             text="Este sistema detecta reproducciones impresas. No "
                  "confirma la autenticidad de un billete ni reemplaza la "
                  "verificación oficial del Banco de la República.")\
        .pack(side="bottom", anchor="w", padx=16, pady=(8, 12))

    # ---- acciones --------------------------------------------------
    def poner(titulo, mensaje, color, diag=None, original=None, rect=None):
        lbl_titulo.config(text=titulo, fg=color)
        lbl_msg.config(text=mensaje)
        estado["ultimo_mensaje"] = mensaje
        txt_diag.config(text=(
            f"mapa       {diag.get('mapa_elegido','—')}\n"
            f"forma      {diag.get('puntaje_forma',0):.3f}\n"
            f"aspecto    {diag.get('aspecto_medido',0):.2f}\n"
            f"matiz      {diag.get('matiz') or 0:.1f} grados\n"
            f"V_lap      {diag.get('v_lap') or 0:.5f}\n"
            f"umbral     {R.UMBRAL_LAPLACIANO:.5f}\n"
            f"holgura    x{diag.get('holgura') or 0:.2f}") if diag else "—")
        for lbl, arr, maxi in ((lienzo, original, (720, 420)),
                               (lbl_rect, rect, (720, 190))):
            if arr is None:
                lbl.config(image="", text="" if lbl is lbl_rect else
                           "Pulse «Seleccionar imagen» para comenzar.")
                lbl.image = None
                continue
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            im.thumbnail(maxi)
            ph = ImageTk.PhotoImage(im)
            lbl.config(image=ph, text="")
            lbl.image = ph

    def tarea(img):
        try:
            cola.put(("resultado", analizar(img)))
        except Exception as e:
            cola.put(("error", str(e)))

    def analizar_ruta(ruta):
        if estado["analizando"]:
            return
        estado["analizando"] = True
        btn_sel.config(state="disabled", text="Analizando...")
        poner("Analizando", f"Procesando {os.path.basename(ruta)}", ACENTO)
        threading.Thread(target=tarea, args=(IO.cargar_rgb(ruta),),
                         daemon=True).start()

    def seleccionar():
        ruta = filedialog.askopenfilename(
            title="Seleccione la fotografía de un billete",
            initialdir=os.path.join(RAIZ, "datos"),
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.heic"),
                       ("Todos los archivos", "*.*")])
        if ruta:
            analizar_ruta(ruta)

    def abrir_ejemplo(event=None):
        sel = lista.curselection()
        if sel:
            analizar_ruta(muestras[sel[0]][3])

    def limpiar():
        estado["analizando"] = False
        btn_sel.config(state="normal", text="Seleccionar imagen")
        poner("Listo", GUIA, ACENTO)

    btn_sel = tk.Button(barra, text="Seleccionar imagen", command=seleccionar,
                        bg=ACENTO, fg="white", relief="flat", height=2,
                        font=("Segoe UI", 14, "bold"), cursor="hand2")
    btn_sel.pack(side="left", fill="x", expand=True, padx=(0, 6))
    tk.Button(barra, text="Analizar otra", command=limpiar, bg="#2b7a52",
              fg="white", relief="flat", height=2, cursor="hand2",
              font=("Segoe UI", 12, "bold")).pack(side="left", padx=3)
    tk.Button(barra, text="Repetir voz",
              command=lambda: voz.decir(estado["ultimo_mensaje"]),
              bg=PANEL, fg=ACENTO, relief="flat", height=2, cursor="hand2",
              font=("Segoe UI", 10)).pack(side="left", padx=3)

    lista.bind("<Double-Button-1>", abrir_ejemplo)
    lista.bind("<Return>", abrir_ejemplo)

    def revisar_cola():
        try:
            while True:
                tipo, dato = cola.get_nowait()
                estado["analizando"] = False
                btn_sel.config(state="normal", text="Seleccionar imagen")
                if tipo == "error":
                    poner("Error", str(dato), ALERTA)
                    continue
                r = dato
                color = {"ok": OK, "alerta": ALERTA}.get(r["estado"], ACENTO)
                titulo = {"ok": r.get("nombre", "Billete").capitalize(),
                          "alerta": "Posible reproducción",
                          "encuadre": "Fotografía no utilizable",
                          "duda": "No identificado"}[r["estado"]]
                poner(titulo, r["mensaje"], color, r.get("diagnostico"),
                      r.get("original"), r.get("rectificada"))
                voz.decir(r["mensaje"])
        except queue.Empty:
            pass
        raiz.after(80, revisar_cola)

    raiz.bind("<o>", lambda e: seleccionar())
    raiz.bind("<n>", lambda e: limpiar())
    raiz.bind("<Escape>", lambda e: raiz.destroy())
    raiz.after(300, lambda: voz.decir(GUIA))
    revisar_cola()
    raiz.mainloop()


# ====================================================================
def main():
    ap = argparse.ArgumentParser(description="Prototipo de escritorio")
    ap.add_argument("--imagen", default=None,
                    help="analiza una fotografia e imprime el resultado")
    args = ap.parse_args()

    if args.imagen:
        if not os.path.exists(args.imagen):
            print(f"No existe el archivo: {args.imagen}")
            sys.exit(1)
        print(f"Analizando {args.imagen}")
        imprimir(analizar(IO.cargar_rgb(args.imagen)))
        return

    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("\nEste equipo no tiene tkinter, la libreria de interfaz")
        print("grafica de Python.")
        print("  Windows y macOS: reinstale Python desde python.org")
        print("                   marcando la opcion tcl/tk.")
        print("  Linux:           sudo apt install python3-tk")
        print("\nMientras tanto puede usar:")
        print("  python src/prototipo_pc.py --imagen <ruta>")
        sys.exit(1)

    lanzar_gui()


if __name__ == "__main__":
    main()
