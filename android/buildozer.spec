[app]
title = Verificador de Billetes
package.name = verificadorbilletes
package.domain = co.edu.umng

# android/app contiene main.py, plataforma.py, carga.py y (tras ejecutar
# preparar_fuente.py) pdi/ y prototipo_pc.py
source.dir = app
source.include_exts = py,png
source.exclude_dirs = __pycache__
version = 0.1.0

# Solo NumPy y Pillow: el procesamiento no usa OpenCV ni scikit-image.
requirements = python3,kivy==2.3.0,numpy,pillow,pyjnius,android

orientation = portrait
fullscreen = 0

# Icono adaptable (Samsung One UI lo recorta en cuadrado redondeado) y
# pantalla de carga oscura, a juego con la interfaz.
icon.filename = %(source.dir)s/data/icon.png
icon.adaptive_foreground.filename = %(source.dir)s/data/icon_fg.png
icon.adaptive_background.filename = %(source.dir)s/data/icon_bg.png
presplash.filename = %(source.dir)s/data/presplash.png
android.presplash_color = #E9EBEE

# Sin permisos: la camara y la galeria se abren con las aplicaciones del
# sistema (Intents), que no los requieren desde Android 10.
android.permissions =

android.api = 34
android.minapi = 29
android.ndk = 25b
# Cubre practicamente todos los telefonos actuales. Para telefonos muy
# antiguos de 32 bits use:  arm64-v8a, armeabi-v7a  (compila el doble).
android.archs = arm64-v8a
android.accept_sdk_license = True
android.allow_backup = False

[buildozer]
log_level = 2
warn_on_root = 1
