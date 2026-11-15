"""
Verificador de billetes colombianos - version Android.

Interfaz Kivy sobre la MISMA cadena de procesamiento del prototipo de
escritorio: src/pdi/ y prototipo_pc.analizar(). No se reescribio ni se
modifico ningun algoritmo; solo cambia la forma de obtener la fotografia
(camara/galeria del telefono) y de mostrar el resultado.

Antes de compilar, android/preparar_fuente.py copia src/pdi y
src/prototipo_pc.py a esta carpeta.

Prueba en computador:
    pip install kivy numpy pillow
    python android/preparar_fuente.py
    python android/app/main.py
"""
import os
import sys
import threading
import traceback

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import numpy as np                                       # noqa: E402
from kivy.app import App                                 # noqa: E402
from kivy.clock import Clock                             # noqa: E402
from kivy.core.window import Window                      # noqa: E402
from kivy.graphics.texture import Texture                # noqa: E402
from kivy.properties import BooleanProperty, ListProperty  # noqa: E402
from kivy.uix.boxlayout import BoxLayout                 # noqa: E402
from kivy.uix.button import Button                       # noqa: E402
from kivy.lang import Builder                            # noqa: E402
from kivy.utils import get_color_from_hex as hexa        # noqa: E402

import prototipo_pc as PC                                # noqa: E402
from pdi import reglas as R                              # noqa: E402
from carga import cargar_rgb_movil                       # noqa: E402
from plataforma import Plataforma                        # noqa: E402

# Paleta: fondo gris claro, marcos de un gris un poco mas oscuro para que
# el texto siempre tenga contraste, titulos en rojo oscuro y botones en
# tres tonos de rojo con texto blanco.
FONDO, MARCO = "#E9EBEE", "#D6D9DE"
TEXTO, TENUE = "#22262B", "#4A525C"
TITULO = "#7A1414"
ROJO_BOTON, ROJO_OSCURO, ROJO_CLARO = "#C62828", "#7F1616", "#D23D3D"
# El titulo es siempre rojo oscuro; el estado se indica con la franja
# de color del borde izquierdo de la tarjeta de resultado.
OK, ALERTA, NEUTRO = "#2E7D32", "#E65100", "#7B8794"

KV = """
#:import hexa kivy.utils.get_color_from_hex

# Diseno para Samsung Galaxy S24 FE: 6.7", 1080 x 2340 px (19.5:9).
# Todo se mide en dp / sp y los anchos siguen al ancho de la ventana,
# de modo que tambien se adapta a otras pantallas y tamanos de fuente.

<Tarjeta>:
    canvas.before:
        Color:
            rgba: hexa('#D6D9DE')
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(22)]
        Color:
            rgba: self.acento
        RoundedRectangle:
            pos: self.pos
            size: dp(8), self.height
            radius: [dp(22), 0, 0, dp(22)]

<Bloque>:
    orientation: 'vertical'
    size_hint_y: None
    height: self.minimum_height if self.visible else 0
    opacity: 1 if self.visible else 0
    padding: dp(14)
    spacing: dp(6)
    canvas.before:
        Color:
            rgba: hexa('#D6D9DE')
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(22)]

<Etiqueta@Label>:
    color: hexa('#4A525C')
    font_size: '13sp'
    size_hint_y: None
    height: self.texture_size[1] + dp(4)
    halign: 'left'
    valign: 'middle'
    text_size: self.width, None

<Boton>:
    background_normal: ''
    background_down: ''
    background_disabled_normal: ''
    background_color: 0, 0, 0, 0
    color: 1, 1, 1, 1
    bold: True
    font_size: '18sp'
    opacity: 0.4 if self.disabled else 1
    canvas.before:
        Color:
            rgba: self.fondo if self.state == 'normal' else [self.fondo[0] * 0.75, self.fondo[1] * 0.75, self.fondo[2] * 0.75, 1]
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(20)]

BoxLayout:
    orientation: 'vertical'
    padding: [dp(16), dp(8), dp(16), dp(14)]
    spacing: dp(10)

    ScrollView:
        do_scroll_x: False
        bar_width: dp(3)
        bar_color: hexa('#7A1414')

        BoxLayout:
            id: contenido
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height
            spacing: dp(14)
            padding: [0, dp(4), 0, dp(6)]

            Etiqueta:
                text: 'VERIFICADOR DE BILLETES COLOMBIANOS'
                bold: True
                font_size: '12sp'

            Tarjeta:
                id: tarjeta
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: [dp(26), dp(18), dp(18), dp(18)]
                spacing: dp(8)

                Label:
                    id: titulo
                    text: 'Listo'
                    color: hexa('#7A1414')
                    bold: True
                    font_size: '28sp'
                    size_hint_y: None
                    height: self.texture_size[1] + dp(6)
                    halign: 'left'
                    valign: 'middle'
                    text_size: self.width, None

                Label:
                    id: mensaje
                    text: ''
                    color: hexa('#22262B')
                    font_size: '18sp'
                    line_height: 1.15
                    size_hint_y: None
                    height: self.texture_size[1] + dp(6)
                    halign: 'left'
                    valign: 'top'
                    text_size: self.width, None

            Etiqueta:
                id: aviso
                text: ''
                font_size: '14sp'
                color: hexa('#3A4149')
                opacity: 1 if self.text else 0
                height: (self.texture_size[1] + dp(4)) if self.text else 0

            Bloque:
                id: bloque_original
                Etiqueta:
                    text: 'Fotografía analizada'
                Image:
                    id: img_original
                    size_hint_y: None
                    height: self.width * 0.62
                    fit_mode: 'contain'

            Bloque:
                id: bloque_rect
                Etiqueta:
                    text: 'Billete rectificado'
                Image:
                    id: img_rect
                    size_hint_y: None
                    height: self.width * 0.44
                    fit_mode: 'contain'

            Bloque:
                id: bloque_diag
                Etiqueta:
                    text: 'Diagnóstico'
                Label:
                    id: diagnostico
                    text: ''
                    color: hexa('#4A525C')
                    font_size: '13sp'
                    font_name: 'RobotoMono-Regular'
                    size_hint_y: None
                    height: self.texture_size[1] + dp(6)
                    halign: 'left'
                    valign: 'top'
                    text_size: self.width, None

            Etiqueta:
                text: 'Este sistema detecta reproducciones impresas. No confirma la autenticidad de un billete ni reemplaza la verificación oficial del Banco de la República.'
                font_size: '12sp'

    # Zona del pulgar: botones grandes en la parte baja de la pantalla.
    BoxLayout:
        orientation: 'vertical'
        size_hint_y: None
        height: dp(76) + dp(58) + dp(10)
        spacing: dp(10)

        Boton:
            id: btn_foto
            text: 'Tomar foto'
            font_size: '22sp'
            size_hint_y: None
            height: dp(76)
            on_release: app.tomar_foto()

        BoxLayout:
            size_hint_y: None
            height: dp(58)
            spacing: dp(10)

            Boton:
                id: btn_galeria
                text: 'Galería'
                fondo: hexa('#7F1616')
                on_release: app.elegir_galeria()
            Boton:
                id: btn_voz
                text: 'Repetir voz'
                fondo: hexa('#D23D3D')
                on_release: app.repetir_voz()
"""


class Boton(Button):
    fondo = ListProperty(list(hexa("#C62828")))


class Tarjeta(BoxLayout):
    acento = ListProperty(list(hexa("#7B8794")))


class Bloque(BoxLayout):
    visible = BooleanProperty(False)


def a_textura(arr):
    """Arreglo (M, N) o (M, N, 3) en [0, 255] -> textura de Kivy."""
    a = np.clip(arr, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    a = np.ascontiguousarray(a)
    alto, ancho = a.shape[:2]
    tex = Texture.create(size=(ancho, alto), colorfmt="rgb")
    tex.blit_buffer(a.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
    tex.flip_vertical()
    return tex


class VerificadorApp(App):
    title = "Verificador de billetes"

    def build(self):
        Window.clearcolor = hexa(FONDO)
        self.analizando = False
        self.ultimo_mensaje = PC.GUIA
        self.plat = Plataforma(self.recibir_imagen, self.mostrar_error)
        self.raiz_ui = Builder.load_string(KV)
        self.ids_ui = self.raiz_ui.ids
        self.poner("Listo", PC.GUIA, NEUTRO)
        self.plat.ajustar_barras(FONDO, claro=True)
        Clock.schedule_once(lambda dt: self.plat.hablar(PC.GUIA), 1.5)
        return self.raiz_ui

    def on_pause(self):
        return True     # al abrir la camara la app no debe reiniciarse

    # ---- acciones -----------------------------------------------------
    def tomar_foto(self):
        if not self.analizando:
            self.plat.detener_voz()
            self.plat.tomar_foto()

    def elegir_galeria(self):
        if not self.analizando:
            self.plat.detener_voz()
            self.plat.elegir_galeria()

    def repetir_voz(self):
        self.plat.hablar(self.ultimo_mensaje)

    # ---- flujo de analisis --------------------------------------------
    def recibir_imagen(self, datos):
        if self.analizando:
            return
        self.analizando = True
        self.bloquear(True)
        self.poner("Analizando",
                   "Procesando la fotografía. Puede tardar unos segundos.",
                   NEUTRO, hablar=False)
        threading.Thread(target=self._trabajo, args=(datos,),
                         daemon=True).start()

    def _trabajo(self, datos):
        try:
            img = cargar_rgb_movil(datos)
            res = PC.analizar(img)
            Clock.schedule_once(lambda dt: self.mostrar_resultado(res), 0)
        except Exception as e:
            traceback.print_exc()
            msg = f"No pude analizar la imagen ({type(e).__name__})."
            Clock.schedule_once(lambda dt: self.mostrar_error(msg), 0)

    def mostrar_resultado(self, r):
        self.analizando = False
        self.bloquear(False)
        color = {"ok": OK, "alerta": ALERTA}.get(r["estado"], NEUTRO)
        titulo = {"ok": r.get("nombre", "Billete").capitalize(),
                  "alerta": "Posible reproducción",
                  "encuadre": "Fotografía no utilizable",
                  "duda": "No identificado"}[r["estado"]]
        # Con una fotografia no utilizable no hay medidas ni billete
        # rectificado que valga la pena mostrar.
        util = r["estado"] != "encuadre"
        self.poner(titulo, r["mensaje"], color,
                   r.get("diagnostico") if util else None,
                   r.get("original"), r.get("rectificada") if util else None)

    def mostrar_error(self, texto):
        self.analizando = False
        self.bloquear(False)
        self.poner("Error", texto, ALERTA)

    def bloquear(self, si):
        for k in ("btn_foto", "btn_galeria"):
            self.ids_ui[k].disabled = si

    # ---- presentacion ---------------------------------------------------
    def poner(self, titulo, mensaje, color, diag=None, original=None,
              rect=None, hablar=True):
        ids = self.ids_ui
        ids.titulo.text = titulo
        ids.tarjeta.acento = hexa(color)
        # El procedimiento oficial de verificacion pasa a un texto
        # secundario: el veredicto queda arriba, corto y legible.
        corte = mensaje.find("Este resultado no confirma")
        if corte > 0:
            ids.mensaje.text = mensaje[:corte].strip()
            ids.aviso.text = mensaje[corte:].strip()
        else:
            ids.mensaje.text = mensaje
            ids.aviso.text = ""
        self.ultimo_mensaje = mensaje      # la voz lee el mensaje completo

        ids.bloque_diag.visible = bool(diag)
        ids.bloque_original.visible = original is not None
        ids.bloque_rect.visible = rect is not None

        if diag:
            ids.diagnostico.text = (
                f"mapa       {diag.get('mapa_elegido', '-')}\n"
                f"forma      {diag.get('puntaje_forma', 0):.3f}\n"
                f"aspecto    {diag.get('aspecto_medido', 0):.2f}\n"
                f"matiz      {diag.get('matiz') or 0:.1f} grados\n"
                f"V_lap      {diag.get('v_lap') or 0:.5f}\n"
                f"umbral     {R.UMBRAL_LAPLACIANO:.5f}\n"
                f"holgura    x{diag.get('holgura') or 0:.2f}")
        else:
            ids.diagnostico.text = ""

        for widget, arr in ((ids.img_original, original),
                            (ids.img_rect, rect)):
            widget.texture = a_textura(arr) if arr is not None else None

        if hablar and titulo not in ("Listo",):
            self.plat.hablar(mensaje)


if __name__ == "__main__":
    VerificadorApp().run()
