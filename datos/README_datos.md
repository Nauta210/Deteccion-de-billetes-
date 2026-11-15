# Banco de datos

## Estado actual

| Carpeta | Contenido | n |
|---|---|---|
| `autentico/50mil/`  | Billetes auténticos $50.000  | 20 |
| `autentico/100mil/` | Billetes auténticos $100.000 | 10 |
| `reproduccion/2mil ... 100mil/` | Billetes didácticos comerciales | 12 |
| `simulado/` | Copias simuladas (generadas por script) | 30 |

## Historial de capturas

- **Serie 1** (descartada): didácticos sobre tapete de fibra de carbono,
  luz tenue, ángulo oblicuo. Fondo texturizado incompatible con el
  modelo de fondo perimetral.
- **Serie 2** (actual): didácticos sobre tela blanca, billete plano,
  cámara cenital. Recortadas con `scripts/recortar_capturas.py`.

## ADVERTENCIA: confusión entre clase y condición de captura

Las fotografías auténticas y las didácticas **no se tomaron en las
mismas condiciones**:

| | Auténticos | Didácticos (serie 2) |
|---|---|---|
| Fondo | superficie clara lisa | tela blanca con pliegues |
| Iluminación | natural difusa | interior tenue |
| Dispositivo | dos cámaras (12 Mpx) | S24 FE (archivo de 1,4 Mpx) |
| Estado del billete | circulado, desgastado | nuevo, sin uso |

El último renglón es el más difícil de descartar: el desgaste añade
textura al papel. Un billete auténtico circulado tiene fibras
levantadas y suciedad que un didáctico nuevo no tiene, y eso eleva la
varianza del laplaciano por una razón que no es la impresión.

Cualquier clasificador entrenado con este conjunto puede aprender
**"foto tenue sobre fondo oscuro = falso"** en lugar de aprender la
diferencia de impresión. Las medidas que más separan las clases en
`comparacion_simple.py` (energía de bordes, varianza del laplaciano)
son precisamente las que degrada una foto oscura y oblicua.

### Pruebas de descarte ya realizadas

1. **Compresión.** Se degradaron los auténticos a la misma cadena de
   reescalado y compresión JPEG. La varianza del laplaciano se mantuvo
   en 0,012–0,017 frente a 0,0003 de los didácticos. La compresión
   **no** explica la diferencia.
2. **Enfoque.** Se midió la agudeza del borde físico del papel, que
   depende del enfoque y no de la impresión. Los didácticos dieron
   1,87 ± 1,66 frente a 0,87 ± 0,27 de los auténticos, es decir están
   igual o mejor enfocados. El desenfoque **no** explica la diferencia.

Quedan sin descartar la diferencia de dispositivo, la de iluminación y
sobre todo la de desgaste del billete.

### Experimento decisivo: REALIZADO

Se fotografiaron un billete auténtico y uno didáctico de \$50.000
dentro del **mismo encuadre**, con la misma cámara, exposición,
distancia, iluminación e instante (`datos/pareado/`). En ese diseño la
condición de captura queda controlada por construcción, de modo que
toda diferencia es atribuible al objeto.

Se repitió para \$100.000. Resultado sobre los reversos
(`scripts/comparacion_pareada.py`), con el didáctico expresado como
cociente respecto del auténtico:

| Medida | \$50.000 | \$100.000 | Promedio |
|---|---|---|---|
| Media de intensidad | 1.31 | 1.18 | 1.24 |
| Desviación estándar | 0.78 | 0.53 | 0.65 |
| Energía de bordes | **0.34** | **0.26** | **0.31** |
| Varianza del laplaciano | **0.04** | **0.01** | **0.04** |
| Saturación media | 0.71 | 0.80 | 0.76 |
| Agudeza del borde físico | 2.70 | 1.37 | 2.14 |
| Lado mayor (tamaño) | 0.860 | 0.746 | — |

Las dos denominaciones coinciden en dirección y en orden de magnitud,
lo que descarta que se trate de una peculiaridad de un ejemplar.

**Conclusión.** La diferencia de microtextura es real y no un artefacto
de captura. El didáctico tiene entre 25 y 100 veces menos energía en el
laplaciano pese a que su borde físico está *más* enfocado (2.14× en
promedio), lo que descarta el desenfoque como explicación. La ausencia de calcografía y de trama
de seguridad es medible con cámara convencional.

**Hallazgo adicional:** el didáctico mide entre 14 % y 25 % menos que el
auténtico.
Las reproducciones de curso legal deben diferir en tamaño, y esa
diferencia es detectable, pero solo si hay una referencia de escala en
la escena. La rectificación al marco canónico la elimina por
construcción, de modo que el sistema no puede usarla.

### Encuadres rechazados automáticamente

De las cuatro fotografías pareadas, dos fueron descartadas por el
propio script, sin intervención manual:

- **Anverso de \$50.000**: el billete auténtico está cortado por el
  borde del cuadro y carece de dos esquinas.
- **Anverso de \$100.000**: no se hallaron dos regiones completas.

Los criterios de rechazo son tres, y cada uno responde a un modo de
error distinto:

1. **Contacto con el borde**, con margen proporcional al tamaño de la
   imagen. Un billete truncado no tiene sus cuatro esquinas.
2. **Relación de aspecto** fuera de [1.45, 2.80]. Señala un billete
   truncado, dos billetes fusionados en una región, o un objeto que no
   es un billete.
3. **Escalón de intensidad en el contorno** inferior a 0.04. Si la
   intensidad media dentro y fuera de la región es casi la misma, la
   frontera detectada no es un borde físico sino un límite arbitrario
   dentro de una zona homogénea; medir la agudeza allí divide por un
   número casi nulo y produce valores enormes sin significado.

El tercer criterio surgió de observar valores de agudeza de 19 y de
139 en corridas intermedias, que son físicamente imposibles.

### Pendiente

Repetir los **anversos** de ambas denominaciones dejando los dos
billetes completos dentro del cuadro, con margen en los cuatro lados y
separados unos dos centímetros entre sí. En el par de \$100.000 los
billetes quedaron casi en contacto, lo que obligó a barrer varias
escalas morfológicas para separarlos.

Conviene además enviar los archivos **a resolución completa de la
cámara**, sin pasar por aplicaciones de mensajería, que reescalan a
1–2 Mpx y recomprimen.

## Protocolo para las capturas nuevas

Mismo fondo, misma luz, mismo teléfono, billete plano y cámara cenital
para las dos clases. Nomenclatura:

```
<clase>_<ejemplar>_<dispositivo>_<iluminacion>_<fondo>_<cara>.jpg
50mil_E03_S24FE_natural_claro_anverso.jpg      (auténtico)
50mil_D01_S24FE_natural_claro_anverso.jpg      (didáctico)
```

El campo **ejemplar** es obligatorio: la partición entre entrenamiento
y prueba se hace por ejemplar físico, no por imagen.

Meta razonable: 15 capturas por cara y por ejemplar, variando solo
ángulo y distancia dentro del rango admisible.
