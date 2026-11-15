"""
Copia el codigo de procesamiento del proyecto a android/app/ para que
buildozer lo empaquete dentro del APK.

    python android/preparar_fuente.py

Se copia (no se duplica a mano) para que el APK use siempre exactamente
la misma cadena de procesamiento que el prototipo de escritorio:

    src/pdi/             ->  android/app/pdi/
    src/prototipo_pc.py  ->  android/app/prototipo_pc.py

No se copia nada de datos/, resultados/ ni informe/.
"""
import os
import shutil

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))
DESTINO = os.path.join(AQUI, "app")


def main():
    pdi_origen = os.path.join(RAIZ, "src", "pdi")
    pdi_destino = os.path.join(DESTINO, "pdi")
    if os.path.isdir(pdi_destino):
        shutil.rmtree(pdi_destino)
    shutil.copytree(pdi_origen, pdi_destino,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(os.path.join(RAIZ, "src", "prototipo_pc.py"),
                 os.path.join(DESTINO, "prototipo_pc.py"))
    n = sum(len(f) for _, _, f in os.walk(pdi_destino))
    print(f"Listo: {n} archivos de pdi/ + prototipo_pc.py -> {DESTINO}")


if __name__ == "__main__":
    main()
