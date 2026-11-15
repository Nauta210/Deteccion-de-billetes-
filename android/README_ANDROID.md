# Versión Android

La aplicación móvil reutiliza **sin modificar** la cadena de procesamiento
del prototipo de escritorio (`src/pdi/` y `prototipo_pc.analizar`). Solo
cambia la entrada (cámara o galería del teléfono) y la salida (pantalla y
voz del teléfono). Sigue sin usar OpenCV ni scikit-image: las únicas
dependencias son NumPy y Pillow.

```
android/
├── app/
│   ├── main.py          Interfaz (Kivy)
│   ├── plataforma.py    Cámara, galería y voz de Android
│   └── carga.py         Reduce la foto antes de procesarla (evita agotar la memoria)
├── preparar_fuente.py   Copia src/pdi y prototipo_pc.py dentro de app/
└── buildozer.spec       Configuración del APK
```

`android/app/pdi/` y `android/app/prototipo_pc.py` **se generan** con
`preparar_fuente.py`; no se editan ni se suben a Git.

## 1. Probar la interfaz en el computador (opcional, 2 minutos)

```bash
pip install kivy numpy pillow
python android/preparar_fuente.py
python android/app/main.py
```

«Tomar foto» y «Galería» abren un selector de archivos. Sirve para revisar
la interfaz y el flujo antes de compilar.

## 2. Generar el APK con GitHub Actions (recomendado, no instala nada)

Buildozer solo funciona en Linux, por eso la compilación se hace en la
nube. El archivo `.github/workflows/apk.yml` ya está configurado.

1. Subir el proyecto al repositorio (ver `GITHUB.md`), incluyendo
   `android/` y `.github/`.
2. En GitHub: pestaña **Actions** → **Compilar APK de Android** →
   **Run workflow**.
3. Esperar. La primera vez tarda 30–60 minutos porque compila Python y
   NumPy para Android.
4. Al terminar, abrir la ejecución y descargar **verificador-billetes-apk**
   (sección *Artifacts*). Es un `.zip` que contiene el `.apk`.

## 3. Alternativas para compilar

**Google Colab** (sin instalar nada): en un cuaderno nuevo, subir la
carpeta del proyecto y ejecutar

```bash
!apt-get install -y -qq openjdk-17-jdk zip unzip autoconf libtool pkg-config \
    zlib1g-dev libncurses5-dev libtinfo5 cmake libffi-dev libssl-dev
!pip install -q buildozer "cython<3"
%cd ProyectoBilletes
!python android/preparar_fuente.py
%cd android
!BUILDOZER_WARN_ON_ROOT=0 buildozer -v android debug
```

El APK queda en `android/bin/`.

**WSL2 en Windows** (Ubuntu): instalar las mismas dependencias que arriba
y ejecutar `buildozer -v android debug` dentro de `android/`.

## 4. Instalar en el teléfono

1. Pasar el `.apk` al teléfono (cable, Drive, WhatsApp, correo).
2. Abrirlo desde el administrador de archivos. Android pedirá permitir la
   instalación desde esa aplicación: aceptar.
3. Abrir **Verificador de Billetes**.

Es un APK de depuración, firmado con la clave de depuración. Sirve para
usarlo y para la entrega; para publicarlo en Google Play habría que
generar uno `release` firmado con una clave propia.

## 5. Pantalla del Samsung Galaxy S24 FE

La interfaz está diseñada para el S24 FE (6,7", 1080 × 2340 px, 19,5:9):

* Los botones están abajo, en la zona del pulgar: «Tomar foto» ocupa todo
  el ancho (76 dp) y debajo van «Galería» y «Repetir voz».
* Arriba queda lo importante: denominación y veredicto en letra grande;
  el procedimiento oficial del Banco de la República va aparte, en texto
  secundario. Después vienen la foto, el billete rectificado y el
  diagnóstico.
* Colores: fondo gris claro (`#E9EBEE`), marcos de gris un poco más
  oscuro (`#D6D9DE`) detrás del texto para que siempre se lea, títulos en
  rojo oscuro (`#7A1414`) y botones en tres rojos con texto blanco. El
  estado del resultado se indica con la franja del borde izquierdo de la
  tarjeta: verde (sin anomalías), naranja (posible reproducción) o gris.
  Los colores están al inicio de `app/main.py` por si se quieren ajustar.
* La barra de estado y la de navegación se pintan del fondo claro (con
  iconos oscuros), y hay icono adaptable (One UI lo recorta con su forma)
  y pantalla de carga a juego.
* Todo se mide en dp/sp y sigue el ancho de la ventana, así que respeta el
  tamaño de fuente y de pantalla que se elija en Ajustes de Samsung.

Se revisó en una simulación a 1080 × 2340 con densidad 2,625 (≈ 411 × 891 dp)
y también con letra al 130 %. Falta verlo en el teléfono real.

## 6. Notas de uso y límites

* **Requiere Android 10 o superior** (`android.minapi = 29`). A cambio, la
  app no pide ningún permiso: abre la cámara del sistema.
* **Velocidad.** En el computador el análisis tarda ~1 s; en un teléfono
  espere varios segundos más porque el algoritmo está escrito en Python
  puro sobre NumPy (así es el proyecto, sin librerías de visión). Mientras
  analiza, los botones se desactivan.
* **Foto de la cámara.** Se reduce a lado mayor ≤ 1024 por promediado de
  bloques antes de convertirla, igual que hace la cadena en el
  computador. El banco de datos con el que se fijaron los umbrales
  (`reglas.py`) está compuesto por fotos ya de ese orden de tamaño. Con
  fotos directas de la cámara conviene **validar de nuevo** los umbrales
  de matiz y microtextura; los resultados del informe no se transfieren
  automáticamente.
* La foto tomada se elimina de la galería después de leerla.

## 7. Si algo falla

Conectar el teléfono con depuración USB y ver el registro:

```bash
adb logcat -s python
```

Los errores de la app aparecen ahí con su traza completa.
