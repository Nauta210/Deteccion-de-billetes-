"""
Generacion de reproducciones impresas SIMULADAS.

ADVERTENCIA METODOLOGICA
------------------------
Las imagenes producidas por este script NO son fotografias de billetes
reproducidos fisicamente. Son una simulacion del efecto que introduce
la cadena digitalizacion -> impresion de consumo -> recaptura, y se
emplean unicamente como prueba de concepto del descriptor espectral,
para verificar que la trama de semitono es detectable antes de invertir
esfuerzo en la captura fisica.

Los resultados obtenidos sobre este conjunto se reportan de forma
separada y no sustituyen la validacion sobre ejemplares reales.

Modelo de la simulacion
-----------------------
1. Perdida de resolucion por muestreo del escaner:
   decimacion por promediado de bloques y reconstruccion por
   interpolacion bilineal.

2. Trama de semitono ordenada. Cada canal se compara contra una
   matriz de umbrales periodica girada un angulo distinto:

       s(x, y) = 1  si  f(x, y) > U( x cos a + y sen a mod n,
                                     -x sen a + y cos a mod n )

   Los angulos de pantalla clasicos de la cuatricromia (15, 75, 0 y 45
   grados) evitan el moire entre canales. Esta periodicidad es
   exactamente la que el analisis espectral detecta.

3. Ganancia de punto y ruido de impresion: leve difuminado del patron
   binario y ruido gaussiano aditivo.

4. Restriccion al soporte del billete. La trama se aplica UNICAMENTE
   dentro de la region del billete, no a toda la fotografia. Una
   reproduccion fisica es una hoja impresa fotografiada sobre una mesa
   corriente: el fondo de la escena no atraviesa la impresora. Aplicar
   la trama a todo el cuadro texturiza el fondo, invalida la hipotesis
   de fondo homogeneo en que se apoya el modelo de segmentacion y
   produce un conjunto simulado que no representa el caso real.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pdi import io_utils as IO          # noqa: E402
from pdi import filtros as F            # noqa: E402
from pdi import segmentacion as SEG     # noqa: E402

# Matriz de Bayer de orden 8, normalizada a (0, 1)
BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float64)
BAYER8 = (BAYER8 + 0.5) / 64.0

ANGULOS = {"R": 15.0, "G": 75.0, "B": 0.0}


def _pantalla(M, N, angulo, periodo=8.0):
    """Matriz de umbrales periodica girada 'angulo' grados."""
    y, x = np.mgrid[0:M, 0:N].astype(np.float64)
    a = np.radians(angulo)
    xr = (x * np.cos(a) + y * np.sin(a)) % periodo
    yr = (-x * np.sin(a) + y * np.cos(a)) % periodo
    i = np.clip((yr / periodo * 8).astype(int), 0, 7)
    j = np.clip((xr / periodo * 8).astype(int), 0, 7)
    return BAYER8[i, j]


def _perdida_resolucion(canal, factor=2):
    M, N = canal.shape
    M2, N2 = (M // factor) * factor, (N // factor) * factor
    peq = canal[:M2, :N2].reshape(M2 // factor, factor,
                                  N2 // factor, factor).mean(axis=(1, 3))
    # Reconstruccion por replicacion seguida de suavizado: aproxima la
    # interpolacion del controlador de impresion.
    grande = np.kron(peq, np.ones((factor, factor)))
    out = canal.copy()
    out[:M2, :N2] = grande
    return F.suavizado_gaussiano(out, sigma=0.8)


def mascara_billete(img_rgb):
    """
    Region del billete, usada como soporte de la trama. Si la
    segmentacion falla se devuelve None y la simulacion se aplica a
    todo el cuadro, advirtiendolo.
    """
    try:
        et = SEG.segmentar(img_rgb)
        m = et["componente"]
        cobertura = m.sum() / float(m.size)
        if 0.05 < cobertura < 0.95:
            return m
    except Exception:
        pass
    return None


def simular(img_rgb, periodo=6.0, factor=2, ruido=3.0, semilla=0,
            mascara=None):
    rng = np.random.default_rng(semilla)
    M, N = img_rgb.shape[:2]
    salida = np.zeros_like(img_rgb)

    for c, nombre in enumerate(["R", "G", "B"]):
        canal = img_rgb[:, :, c] / 255.0
        canal = _perdida_resolucion(canal, factor)
        U = _pantalla(M, N, ANGULOS[nombre], periodo)
        binario = (canal > U).astype(np.float64)
        # Ganancia de punto: el toner se extiende mas alla del punto nominal
        difuso = F.suavizado_gaussiano(binario, sigma=0.9)
        mezcla = 0.80 * difuso + 0.20 * canal
        salida[:, :, c] = np.clip(mezcla * 255.0
                                  + rng.normal(0, ruido, size=(M, N)), 0, 255)

    if mascara is not None:
        m = mascara[:, :, None]
        salida = np.where(m, salida, img_rgb)
    return salida


def main():
    if len(sys.argv) < 3:
        print("Uso: python simular_reproduccion.py <entrada> <salida> [periodo]")
        sys.exit(1)
    entrada, salida = sys.argv[1], sys.argv[2]
    periodo = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0

    os.makedirs(salida, exist_ok=True)
    rutas = IO.listar_imagenes(entrada)
    print(f"Simulando {len(rutas)} imagenes con periodo de trama {periodo} px")

    sin_mascara = 0
    for k, r in enumerate(rutas):
        img = IO.cargar_rgb(r)
        img = IO.reducir_lado_mayor(img, 1024)
        m = mascara_billete(img)
        if m is None:
            sin_mascara += 1
        sim = simular(img, periodo=periodo, semilla=k, mascara=m)
        base = os.path.splitext(os.path.basename(r))[0]
        rel = os.path.relpath(os.path.dirname(r), entrada)
        destino = os.path.join(salida, rel)
        os.makedirs(destino, exist_ok=True)
        IO.guardar_rgb(sim, os.path.join(destino, f"{base}_SIM.png"))
        print(f"  [{k+1}/{len(rutas)}] {base}"
              + ("   (sin mascara: trama aplicada a todo el cuadro)"
                 if m is None else ""))

    if sin_mascara:
        print(f"\nAviso: en {sin_mascara} imagenes no se pudo aislar el "
              f"billete y la trama se aplico a todo el cuadro.")


if __name__ == "__main__":
    main()
