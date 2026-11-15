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
# Sin versiones fijadas: se usan las de python-for-android, que son las que
# esa version probo en conjunto (Python 3.14, Kivy 2.3.1, NumPy via la receta
# local de android/p4a-recipes).
requirements = python3,kivy,numpy,pillow,pyjnius,android

orientation = portrait
fullscreen = 0

# Icono adaptable (Samsung One UI lo recorta en cuadrado redondeado) y
# pantalla de carga, a juego con la interfaz.
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
# NDK: NO se fija. Buildozer usa el que recomienda python-for-android
# (r28c con esta version). Fijar uno mas viejo (r25b) hace fallar la
# compilacion de NumPy.

# Cubre practicamente todos los telefonos actuales. Para telefonos muy
# antiguos de 32 bits use:  arm64-v8a, armeabi-v7a  (compila el doble).
android.archs = arm64-v8a
android.accept_sdk_license = True
android.allow_backup = False

# Version FIJA de python-for-android (en lugar de 'master', que cambia a
# diario y puede romper la compilacion de un dia para otro).
p4a.branch = v2026.05.09
# Recetas propias que reemplazan a las de python-for-android (ruta relativa
# a la carpeta android/, desde donde se ejecuta buildozer).
p4a.local_recipes = ./p4a-recipes

[buildozer]
log_level = 2
warn_on_root = 1
