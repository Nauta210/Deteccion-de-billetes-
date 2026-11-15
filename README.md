# Detección de reproducciones impresas de billetes colombianos

Sistema de procesamiento digital de imágenes que reconoce la denominación
de billetes colombianos, discrimina entre ejemplares auténticos y
reproducciones impresas, y comunica el resultado por voz a través de una
interfaz accesible.

**Universidad Militar Nueva Granada — Ingeniería Multimedia**
Juan Castaño (1202720) · Nicolás Moreno (1202437) · Andrés Navia (1202517)

---

## Restricción de implementación

Este proyecto **no utiliza librerías de visión artificial**. No se emplea
OpenCV ni scikit-image en ninguna etapa del procesamiento. Todas las
operaciones se implementaron a partir de sus definiciones matemáticas
sobre arreglos numéricos:

| Operación | Implementada en | Reemplaza a |
|---|---|---|
| Extracción de la región del objeto | `pdi/etiquetado.py` | `cv2.findContours` |
| Seguimiento de frontera | `pdi/contornos.py` | `cv2.findContours` |
| Rectángulo de área mínima | `pdi/contornos.py` | `cv2.minAreaRect` |
| Transformación de perspectiva | `pdi/geometria.py` | `cv2.getPerspectiveTransform` + `cv2.warpPerspective` |
| Umbralización automática | `pdi/umbral.py` | `cv2.threshold(THRESH_OTSU)` |
| Morfología matemática | `pdi/umbral.py` | `cv2.morphologyEx` |
| Convolución y filtrado | `pdi/filtros.py` | `cv2.filter2D`, `cv2.GaussianBlur` |
| Conversión de color | `pdi/color.py` | `cv2.cvtColor` |
| Envolvente convexa | `pdi/convexa.py` | `cv2.convexHull` |
| Descriptores de textura | `pdi/descriptores.py` | `skimage.feature` |

Las únicas dependencias del procesamiento son NumPy (álgebra sobre
arreglos), Pillow (decodificación de archivos) y Matplotlib (figuras).

Sin excepciones: `grep -rn "cv2" src/ scripts/` no devuelve nada.

El prototipo analiza fotografías seleccionadas por el usuario. La
captura en vivo corresponde a la etapa siguiente, sobre dispositivo
móvil.

---

## Estructura

```
.
├── src/
│   ├── pdi/                  Librería propia de procesamiento
│   │   ├── io_utils.py       Carga de imágenes y remuestreo
│   │   ├── color.py          RGB→gris, RGB→HSV, balance de blancos
│   │   ├── filtros.py        Convolución, gaussiano, Sobel, Laplaciano
│   │   ├── umbral.py         Otsu y morfología matemática
│   │   ├── etiquetado.py     Componentes conexas (union-find)
│   │   ├── convexa.py        Envolvente convexa y cuadrilátero máximo
│   │   ├── geometria.py      Homografía DLT y remapeo bilineal
│   │   ├── descriptores.py   LBP, GLCM, Fourier, Laplaciano
│   │   ├── segmentacion.py   Cadena completa de segmentación
│   │   └── clasificador.py   Denominación, orientación, autenticidad
│   ├── entrenar.py           Entrenamiento y evaluación
│   ├── analizar.py           Figuras por etapa de una imagen
│   ├── app.py                Servidor de la interfaz accesible
│   └── plantillas/index.html Interfaz web
├── scripts/
│   ├── simular_reproduccion.py  Generación de copias simuladas
│   └── barrido_dificultad.py    Barrido de finura de trama
├── datos/
│   ├── autentico/{50mil,100mil}/     Billetes auténticos
│   ├── reproduccion/{50mil,100mil}/  Reproducciones físicas
│   └── simulado/                     Copias simuladas (generadas)
├── resultados/               Modelo, métricas y figuras
├── informe/                  Documento en LaTeX y PDF
└── requirements.txt
```

---

## Instalación

Se requiere Python 3.9 o superior.

```bash
pip install -r requirements.txt
```

El paquete `pillow-heif` solo es necesario si el banco de datos contiene
archivos `.HEIC` (fotografías tomadas con iPhone). Si todas las imágenes
son JPG o PNG, puede omitirse.

---

## Ejecución

Abrir la carpeta en VS Code (*Archivo → Abrir carpeta*), abrir la
terminal integrada y ejecutar:

```bash
pip install -r requirements.txt
python src/prototipo_pc.py      # abre la aplicación
```

Sin cámara, para verificar el algoritmo:

```bash
python src/prototipo_pc.py --imagen datos/autentico/50mil/IMG_2198.jpg
python src/prototipo_pc.py --imagen datos/reproduccion/100mil/100mil_D01_S24FE_tenue_blanco_anverso.jpg
```

Instrucciones detalladas y solución de problemas en `INSTRUCCIONES.txt`.

### Reproducir los resultados del informe

```bash
python scripts/comparacion_simple.py
python scripts/comparacion_pareada.py datos/pareado/*.jpg
python src/analizar.py datos/autentico/50mil/IMG_2198.jpg
python scripts/barrido_dificultad.py
```

## Banco de datos

Las imágenes deben organizarse así:

```
datos/autentico/50mil/      fotografías de billetes auténticos de $50.000
datos/autentico/100mil/     fotografías de billetes auténticos de $100.000
datos/reproduccion/50mil/   fotografías de reproducciones impresas
datos/reproduccion/100mil/
```

**Nomenclatura obligatoria para las capturas nuevas.** El nombre de cada
archivo debe identificar el ejemplar físico fotografiado:

```
<clase>_<ejemplar>_<dispositivo>_<iluminacion>_<fondo>_<cara>.jpg
ejemplo:  50mil_E03_S24FE_natural_claro_anverso.jpg
```

Esto es indispensable: la partición entre entrenamiento y prueba se
realiza **por ejemplar físico y no por imagen**. Si varias fotografías
del mismo billete quedaran repartidas entre ambos conjuntos, el modelo
podría reconocer el ejemplar en lugar de la propiedad de impresión, y
las métricas resultarían artificialmente optimistas.

---

## Estado actual

| Módulo | Estado |
|---|---|
| Segmentación y rectificación | Implementado y evaluado (96.7 %) |
| Reconocimiento de denominación | Implementado y evaluado (100 %, n=29) |
| Verificación de autenticidad | Implementado; validado sobre copias **simuladas** |
| Prototipo de escritorio con voz | Funcional |
| Aplicación móvil con captura en vivo | Etapa siguiente |
| Validación con reproducciones físicas | **Pendiente** |

La clase *reproducción* del conjunto actual es sintética. Los resultados
del módulo de autenticidad son una prueba de concepto del descriptor y
**no constituyen una estimación de su desempeño sobre reproducciones
reales**.
