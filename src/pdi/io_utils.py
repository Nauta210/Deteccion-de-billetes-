"""
E/S de imagenes y remuestreo.

Solo se usa PIL para decodificar el archivo a un arreglo de numeros.
Todo el procesamiento posterior se realiza sobre arreglos numericos.
"""
import os
import numpy as np
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def cargar_rgb(ruta):
    """Devuelve la imagen como arreglo float64 de forma (M, N, 3) en [0, 255]."""
    im = Image.open(ruta).convert("RGB")
    return np.asarray(im, dtype=np.float64)


def guardar_rgb(arr, ruta):
    a = np.clip(arr, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        Image.fromarray(a, mode="L").save(ruta)
    else:
        Image.fromarray(a, mode="RGB").save(ruta)


def listar_imagenes(carpeta):
    salida = []
    for raiz, _, archivos in os.walk(carpeta):
        for f in sorted(archivos):
            if os.path.splitext(f)[1].lower() in EXTS:
                salida.append(os.path.join(raiz, f))
    return salida


# --------------------------------------------------------------------
# Remuestreo por promediado de bloques (decimacion con antialias).
#
#   g(i, j) = (1 / |B_ij|) * sum_{(u,v) in B_ij} f(u, v)
#
# Promediar antes de decimar evita el aliasing que introduciria un
# submuestreo directo, de acuerdo con el criterio de Nyquist.
# --------------------------------------------------------------------
def reducir_lado_mayor(img, lado_max=1024):
    M, N = img.shape[:2]
    L = max(M, N)
    if L <= lado_max:
        return img.copy()
    k = int(np.ceil(L / float(lado_max)))
    return promediar_bloques(img, k)


def promediar_bloques(img, k):
    """Promedia bloques de k x k pixeles."""
    if k <= 1:
        return img.copy()
    M, N = img.shape[:2]
    M2, N2 = (M // k) * k, (N // k) * k
    rec = img[:M2, :N2]
    if rec.ndim == 2:
        return rec.reshape(M2 // k, k, N2 // k, k).mean(axis=(1, 3))
    C = rec.shape[2]
    return rec.reshape(M2 // k, k, N2 // k, k, C).mean(axis=(1, 3))
