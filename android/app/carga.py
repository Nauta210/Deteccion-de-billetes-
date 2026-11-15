"""
Carga de fotografias para el movil.

Una fotografia de telefono moderno tiene entre 12 y 50 megapixeles.
Convertirla directamente a float64, como hace pdi.io_utils.cargar_rgb en
el computador, ocuparia entre 290 MB y 1.2 GB y la aplicacion se cerraria
por falta de memoria. Aqui la imagen se reduce ANTES de convertirla,
con el mismo criterio que usa la cadena de procesamiento: promediado de
bloques (decimacion con antialias) hasta que el lado mayor sea <= 1024.

El resultado entra a prototipo_pc.analizar() igual que en el computador.
"""
import io
import numpy as np
from PIL import Image

LADO_MAX = 1024


def cargar_rgb_movil(origen, lado_max=LADO_MAX):
    """
    origen: ruta de archivo, objeto tipo archivo o bytes.
    Devuelve un arreglo float64 (M, N, 3) en [0, 255] con lado mayor
    <= lado_max.
    """
    if isinstance(origen, (bytes, bytearray)):
        origen = io.BytesIO(origen)
    im = Image.open(origen)

    k = max(1, int(np.ceil(max(im.size) / float(lado_max))))
    if k > 1 and im.format == "JPEG":
        # Decodificacion JPEG reducida (1/2, 1/4 o 1/8): evita construir
        # en memoria la imagen completa.
        im.draft("RGB", (im.size[0] // k, im.size[1] // k))
    im = im.convert("RGB")

    k = max(1, int(np.ceil(max(im.size) / float(lado_max))))
    if k > 1:
        im = im.reduce(k)           # promedio de bloques k x k
    return np.asarray(im, dtype=np.float64)
