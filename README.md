# 🦜🔗 Mini proyecto LangChain: Explicador de conceptos

Un programa muy pequeño en Python que usa **LangChain** para:

1. Explicar cualquier tema con palabras sencillas.
2. Generar una pregunta de repaso **estructurada** (pregunta, respuesta y dificultad), validada con **Pydantic**.

```
$ python main.py fotosíntesis

📘 Explicación:
La fotosíntesis es el proceso con el que las plantas fabrican su propia comida...

❓ Pregunta de repaso (fácil):
¿Qué necesitan las plantas para hacer la fotosíntesis?

(Pulsa Enter para ver la respuesta)
✅ Luz del sol, agua y dióxido de carbono.
```

---

## 📂 Archivos del proyecto

| Archivo | Para qué sirve |
|---|---|
| `main.py` | Todo el código (≈70 líneas, comentado paso a paso). |
| `requirements.txt` | Librerías que hay que instalar. |
| `.env.example` | Plantilla para poner tu API key. |
| `.gitignore` | Evita subir a GitHub tu `.env` (tu clave secreta). |

---

## 🚀 Cómo ejecutarlo

```bash
# 1. Crear un entorno virtual (opcional, pero recomendado)
python -m venv venv
source venv/bin/activate          # En Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar la API key
cp .env.example .env              # En Windows: copy .env.example .env
#    ...y edita .env poniendo tu clave de https://console.anthropic.com

# 4. Ejecutar
python main.py agujeros negros
#    o sin argumentos y te pregunta el tema:
python main.py
```

---

## 🧠 ¿Qué es LangChain?

LangChain es una librería que facilita construir programas que usan **modelos de lenguaje (LLMs)**
como Claude o GPT. En vez de escribir a mano todas las llamadas a la API, te da **piezas
reutilizables** que se conectan entre sí como bloques de LEGO.

En este proyecto usamos estas piezas:

| Pieza | Clase en el código | Qué hace |
|---|---|---|
| **Prompt Template** | `ChatPromptTemplate` | Una plantilla de mensaje con huecos (`{tema}`) que se rellenan después. |
| **Modelo (LLM)** | `ChatAnthropic` | El "cerebro": envía el mensaje a Claude y recibe la respuesta. |
| **Output Parser** | `StrOutputParser` | Convierte la respuesta del modelo en texto normal (`str`). |
| **Salida estructurada** | `with_structured_output` + Pydantic | Obliga al modelo a responder con un JSON que cumple un esquema y lo convierte en un objeto Python validado. |

### La "cadena" (chain)

La idea clave de LangChain es **encadenar** piezas con el operador `|` (como una tubería):

```python
cadena_explicar = prompt_explicar | modelo | parser
```

Se lee así: *"la salida del prompt entra al modelo, y la salida del modelo entra al parser"*.

```
 {"tema": "fotosíntesis"}
          │
          ▼
 ┌──────────────────┐     ┌──────────────┐     ┌─────────────────┐
 │  prompt_explicar │ ──▶ │    modelo    │ ──▶ │     parser      │ ──▶ "La fotosíntesis es..."
 │ (rellena {tema}) │     │   (Claude)   │     │ (saca el texto) │
 └──────────────────┘     └──────────────┘     └─────────────────┘
```

Para ejecutar una cadena se usa `.invoke()` pasando un diccionario con los huecos:

```python
explicacion = cadena_explicar.invoke({"tema": "fotosíntesis"})
```

Después, el resultado de la primera cadena se usa como entrada de la segunda:

```python
pregunta = cadena_pregunta.invoke({"explicacion": explicacion})
```

¡Eso es todo! Dos cadenas, una detrás de otra.

### Salida estructurada con Pydantic

La segunda cadena no devuelve texto libre, sino un **objeto con campos fijos**. El "molde" se define
con una clase de **Pydantic**:

```python
class PreguntaRepaso(BaseModel):
    pregunta: str
    respuesta_correcta: str
    dificultad: Literal["fácil", "media", "difícil"]

cadena_pregunta = prompt_pregunta | modelo.with_structured_output(PreguntaRepaso, method="json_schema")
```

- `with_structured_output(...)` convierte la clase en un **JSON Schema** y se lo envía a Claude.
- `method="json_schema"` usa la **salida estructurada nativa** de Claude: el modelo está obligado a
  responder con un JSON que cumple ese esquema.
- Pydantic **valida** la respuesta: si `dificultad` fuera `"imposible"` o faltara un campo, lanzaría un
  error en vez de dejar pasar datos incorrectos.
- El resultado es un objeto Python normal: `repaso.pregunta`, `repaso.dificultad`...

¿Por qué es útil? Porque con texto libre no sabes qué formato vas a recibir; con un esquema,
tu programa puede usar la respuesta con seguridad (guardarla, mostrarla por partes, etc.).

---

## 🔍 Recorrido por `main.py`

1. **`load_dotenv()`** – lee el archivo `.env` y carga `ANTHROPIC_API_KEY` como variable de entorno.
2. **`ChatAnthropic(...)`** – crea el modelo. Por defecto usa `claude-opus-5-5`; puedes cambiarlo
   con `MODELO=...` en el `.env` (por ejemplo `claude-haiku-5-5`, más barato y rápido).
3. **`ChatPromptTemplate.from_messages([...])`** – define los mensajes:
   - `"system"`: instrucciones de comportamiento ("eres un profesor paciente").
   - `"human"`: lo que pregunta el usuario, con el hueco `{tema}`.
4. **`class PreguntaRepaso(BaseModel)`** – el esquema Pydantic de la pregunta de repaso.
5. **`StrOutputParser()`** – para quedarnos solo con el texto de la explicación.
6. **`prompt | modelo | ...`** – se arman las dos cadenas (la segunda con `with_structured_output`).
7. **`main()`** – lee el tema, ejecuta las dos cadenas, muestra la pregunta y, al pulsar Enter, la respuesta.
