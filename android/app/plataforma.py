"""
Capa de plataforma: camara, galeria y voz.

En Android se usan las aplicaciones del sistema mediante Intents:

  * Camara  -> MediaStore.ACTION_IMAGE_CAPTURE, escribiendo en un URI
    creado en MediaStore. Desde Android 10 no requiere ningun permiso
    de almacenamiento ni el permiso CAMERA.
  * Galeria -> Intent.ACTION_GET_CONTENT con tipo image/*.
  * Voz     -> android.speech.tts.TextToSpeech, en espanol.

En computador (para probar la interfaz sin telefono) la camara se
sustituye por un selector de archivos de tkinter y la voz usa
prototipo_pc.Voz.

Ninguna de las operaciones de esta clase hace procesamiento de imagen:
solo entrega al resto de la aplicacion la fotografia elegida.
"""
import io
import os
import time
import traceback

from kivy.clock import Clock
from kivy.utils import platform

EN_ANDROID = platform == "android"

REQ_FOTO = 7001
REQ_GALERIA = 7002
RESULT_OK = -1          # Activity.RESULT_OK


class Plataforma:
    def __init__(self, al_recibir, al_error):
        """
        al_recibir(bytes)  se llama (en el hilo de Kivy) con los bytes del
                           archivo de imagen elegido o fotografiado.
        al_error(texto)    se llama si algo falla.
        """
        self.al_recibir = al_recibir
        self.al_error = al_error
        self._uri_foto = None
        self._tts = None
        self._tts_listo = False
        self._pendiente = None

        if EN_ANDROID:
            self._iniciar_android()
        else:
            self._iniciar_escritorio()

    # ------------------------------------------------------------------
    # Interfaz publica
    # ------------------------------------------------------------------
    def tomar_foto(self):
        if EN_ANDROID:
            self._tomar_foto_android()
        else:
            self.elegir_galeria()

    def elegir_galeria(self):
        if EN_ANDROID:
            self._galeria_android()
        else:
            self._archivo_escritorio()

    def hablar(self, texto):
        if EN_ANDROID:
            self._hablar_android(texto)
        else:
            self._voz_escritorio.decir(texto)

    def ajustar_barras(self, color_hex, claro=False):
        """
        Pinta la barra de estado y la de navegacion del color de fondo.
        claro=True: fondo claro, por lo que los iconos del sistema
        (hora, bateria, gestos) se dibujan oscuros.
        """
        if not EN_ANDROID:
            return
        from android.runnable import run_on_ui_thread
        from jnius import autoclass

        @run_on_ui_thread
        def aplicar():
            try:
                Color = autoclass("android.graphics.Color")
                ventana = self._actividad().getWindow()
                ventana.addFlags(-2147483648)   # DRAWS_SYSTEM_BAR_BACKGROUNDS
                c = Color.parseColor(color_hex)
                ventana.setStatusBarColor(c)
                ventana.setNavigationBarColor(c)
                if claro:
                    decor = ventana.getDecorView()
                    # LIGHT_STATUS_BAR (0x2000) | LIGHT_NAVIGATION_BAR (0x10)
                    decor.setSystemUiVisibility(
                        decor.getSystemUiVisibility() | 0x2000 | 0x10)
            except Exception:
                traceback.print_exc()
        aplicar()

    def detener_voz(self):
        if EN_ANDROID and self._tts is not None:
            try:
                self._tts.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Escritorio (solo para pruebas)
    # ------------------------------------------------------------------
    def _iniciar_escritorio(self):
        from prototipo_pc import Voz
        self._voz_escritorio = Voz()

    def _archivo_escritorio(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            raiz = tk.Tk()
            raiz.withdraw()
            ruta = filedialog.askopenfilename(
                title="Seleccione la fotografia de un billete",
                filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp"),
                           ("Todos", "*.*")])
            raiz.destroy()
            if ruta:
                with open(ruta, "rb") as f:
                    datos = f.read()
                Clock.schedule_once(lambda dt: self.al_recibir(datos), 0)
        except Exception as e:
            self.al_error(f"No pude abrir el archivo: {e}")

    # ------------------------------------------------------------------
    # Android
    # ------------------------------------------------------------------
    def _iniciar_android(self):
        from jnius import autoclass
        from android import activity

        self._PythonActivity = autoclass("org.kivy.android.PythonActivity")
        activity.bind(on_activity_result=self._on_activity_result)
        self._iniciar_voz_android()

    def _actividad(self):
        return self._PythonActivity.mActivity

    def _tomar_foto_android(self):
        try:
            from jnius import autoclass, cast
            Intent = autoclass("android.content.Intent")
            MediaStore = autoclass("android.provider.MediaStore")
            ContentValues = autoclass("android.content.ContentValues")
            ImagenesMedia = autoclass("android.provider.MediaStore$Images$Media")

            act = self._actividad()
            valores = ContentValues()
            valores.put("_display_name",
                        "billete_%d.jpg" % int(time.time()))
            valores.put("mime_type", "image/jpeg")
            valores.put("relative_path", "Pictures/VerificadorBilletes")
            uri = act.getContentResolver().insert(
                ImagenesMedia.EXTERNAL_CONTENT_URI, valores)
            if uri is None:
                self.al_error("No pude preparar el archivo de la foto.")
                return
            self._uri_foto = uri

            intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            intent.putExtra(MediaStore.EXTRA_OUTPUT,
                            cast("android.os.Parcelable", uri))
            act.startActivityForResult(intent, REQ_FOTO)
        except Exception:
            traceback.print_exc()
            self.al_error("No pude abrir la camara.")

    def _galeria_android(self):
        try:
            from jnius import autoclass
            Intent = autoclass("android.content.Intent")
            intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            self._actividad().startActivityForResult(intent, REQ_GALERIA)
        except Exception:
            traceback.print_exc()
            self.al_error("No pude abrir la galeria.")

    def _on_activity_result(self, codigo, resultado, intent):
        # Se ejecuta fuera del hilo de Kivy: se delega con Clock.
        try:
            if codigo == REQ_FOTO:
                uri, self._uri_foto = self._uri_foto, None
                if resultado == RESULT_OK and uri is not None:
                    datos = self._leer_uri(uri)
                    # La foto ya se leyo: se elimina de la galeria para no
                    # dejar fotografias de billetes en el telefono.
                    self._borrar_uri(uri)
                    Clock.schedule_once(lambda dt: self.al_recibir(datos), 0)
                elif uri is not None:
                    self._borrar_uri(uri)          # cancelada
            elif codigo == REQ_GALERIA:
                if resultado == RESULT_OK and intent is not None:
                    uri = intent.getData()
                    if uri is not None:
                        datos = self._leer_uri(uri)
                        Clock.schedule_once(
                            lambda dt: self.al_recibir(datos), 0)
        except Exception as e:
            traceback.print_exc()
            msg = f"No pude leer la imagen: {e}"
            Clock.schedule_once(lambda dt: self.al_error(msg), 0)

    def _leer_uri(self, uri):
        """Lee los bytes de un content:// URI mediante su descriptor."""
        resolver = self._actividad().getContentResolver()
        pfd = resolver.openFileDescriptor(uri, "r")
        fd = pfd.detachFd()                 # ahora el descriptor es nuestro
        with os.fdopen(fd, "rb") as f:
            return f.read()

    def _borrar_uri(self, uri):
        try:
            self._actividad().getContentResolver().delete(uri, None, None)
        except Exception:
            pass

    # ---- voz ----------------------------------------------------------
    def _iniciar_voz_android(self):
        try:
            from jnius import autoclass, PythonJavaClass, java_method
            TextToSpeech = autoclass("android.speech.tts.TextToSpeech")
            Locale = autoclass("java.util.Locale")
            plat = self

            class AlIniciarVoz(PythonJavaClass):
                __javainterfaces__ = [
                    "android/speech/tts/TextToSpeech$OnInitListener"]
                __javacontext__ = "app"

                @java_method("(I)V")
                def onInit(self, estado):
                    if estado != 0:             # TextToSpeech.SUCCESS
                        return
                    for loc in (Locale("es", "CO"), Locale("es", "US"),
                                Locale("es")):
                        if plat._tts.setLanguage(loc) >= 0:
                            break
                    plat._tts_listo = True
                    if plat._pendiente:
                        texto, plat._pendiente = plat._pendiente, None
                        plat._decir(texto)

            self._escucha = AlIniciarVoz()      # referencia viva
            self._tts = TextToSpeech(self._actividad(), self._escucha)
        except Exception:
            traceback.print_exc()
            self._tts = None

    def _hablar_android(self, texto):
        if self._tts is None:
            return
        if not self._tts_listo:
            self._pendiente = texto
            return
        self._decir(texto)

    def _decir(self, texto):
        try:
            from jnius import autoclass
            TextToSpeech = autoclass("android.speech.tts.TextToSpeech")
            try:
                self._tts.speak(texto, TextToSpeech.QUEUE_FLUSH, None,
                                "billete")
            except Exception:
                self._tts.speak(texto, TextToSpeech.QUEUE_FLUSH, None)
        except Exception:
            traceback.print_exc()
