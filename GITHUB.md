# Publicar el proyecto en GitHub

Usuario: **[rammiuwu](https://github.com/rammiuwu)**
Repositorio propuesto: `deteccion-billetes-colombianos`

El repositorio pesa unos **11 MB**, así que no se requiere Git LFS ni
ninguna configuración especial. Las reproducciones simuladas (47 MB de
PNG derivados) quedan fuera por `.gitignore`, porque se regeneran con un
comando.

---

## 1. Crear el repositorio vacío en GitHub

1. Entrar a <https://github.com/new>
2. **Repository name:** `deteccion-billetes-colombianos`
3. **Description:** `Detección de reproducciones impresas de billetes colombianos con interfaz accesible. Procesamiento Digital de Imágenes, UMNG.`
4. Elegir **Public** o **Private**. Ver la nota al final sobre esto.
5. **No marcar** ninguna de las casillas de inicialización (`Add a README`,
   `.gitignore`, `license`). El proyecto ya trae esos archivos y marcarlas
   provoca un conflicto en el primer `push`.
6. Pulsar **Create repository**.

---

## 2. Instalar y configurar Git (solo la primera vez)

Descargar desde <https://git-scm.com/downloads> e instalar. Luego, en una
terminal:

```bash
git config --global user.name "Juan Castaño"
git config --global user.email "correo@unimilitar.edu.co"
```

Usar el mismo correo de la cuenta de GitHub, para que los commits queden
asociados al perfil.

---

## 3. Subir el proyecto

Abrir una terminal **dentro de la carpeta `ProyectoBilletes`**:

```bash
cd ruta/hasta/ProyectoBilletes

git init
git branch -M main
git add .
git commit -m "Entrega 2: metodología, librería propia de PDI y resultados parciales"
git remote add origin https://github.com/rammiuwu/deteccion-billetes-colombianos.git
git push -u origin main
```

Antes de `git add .`, conviene verificar qué se va a subir:

```bash
git status --short
```

No debe aparecer nada dentro de `datos/simulado/` ni archivos `.aux`,
`.log` o `__pycache__`. Si aparecen, es que `.gitignore` no se copió.

---

## 4. Autenticación

GitHub no acepta la contraseña de la cuenta desde la terminal. Al hacer
`push` pedirá usuario y contraseña; en **contraseña** hay que pegar un
*token de acceso personal*:

1. <https://github.com/settings/tokens> → **Generate new token (classic)**
2. **Note:** `proyecto-pdi`
3. **Expiration:** 90 días
4. Marcar únicamente el alcance **`repo`**
5. **Generate token** y copiarlo de inmediato (solo se muestra una vez)

Guardarlo en un gestor de contraseñas. Si se pierde, se genera otro.

En Windows, el Administrador de credenciales lo recuerda tras el primer
uso. En Linux o macOS, para no repetirlo cada vez:

```bash
git config --global credential.helper store
```

---

## 5. Trabajo posterior

Cada vez que se avance:

```bash
git add .
git commit -m "Descripción corta de lo que se hizo"
git push
```

Para que los tres puedan trabajar, en **Settings → Collaborators** del
repositorio se agregan los usuarios de GitHub de Nicolás y Andrés. Cada
uno clona con:

```bash
git clone https://github.com/rammiuwu/deteccion-billetes-colombianos.git
cd deteccion-billetes-colombianos
pip install -r requirements.txt
python scripts/simular_reproduccion.py datos/autentico datos/simulado 6.0
python src/entrenar.py
```

Antes de empezar a trabajar cada día, conviene traer los cambios de los
demás:

```bash
git pull
```

---

## 6. Agregar las reproducciones físicas

Cuando consigan los billetes didácticos impresos:

```bash
# copiar las fotos a las carpetas correspondientes
datos/reproduccion/50mil/
datos/reproduccion/100mil/

# nombrar cada archivo identificando el ejemplar FÍSICO:
#   50mil_R01_S24FE_natural_claro_anverso.jpg

python src/entrenar.py     # reentrena incluyendo la clase real
git add datos/reproduccion resultados/
git commit -m "Reproducciones físicas: N ejemplares, M capturas"
git push
```

---

## 7. Conectar el notebook de Colab al repositorio

Una vez subido, la primera celda del cuaderno puede simplificarse a:

```python
!git clone https://github.com/rammiuwu/deteccion-billetes-colombianos.git
%cd deteccion-billetes-colombianos
!pip install -q pillow-heif
```

y ya no hace falta subir el ZIP a mano en cada sesión.

---

## Nota sobre público o privado

El repositorio contiene fotografías de billetes auténticos, lo cual es
habitual en proyectos académicos de esta área. Lo que **no** conviene
publicar son las reproducciones, ni las simuladas ni las físicas, y por
eso `datos/simulado/` está en `.gitignore`. Cuando agreguen las
reproducciones físicas, decidan conscientemente si las versionan; una
opción razonable es mantener el repositorio **privado** mientras dure el
proyecto y darle acceso al docente como colaborador, y publicarlo al
final solo con el código.

Para agregar al docente: **Settings → Collaborators → Add people**.
