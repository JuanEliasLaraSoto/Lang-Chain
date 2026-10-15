"""
Experimentos: cosas que probé a romper para entender LangChain.

No necesita API key ni gasta dinero: usa un modelo falso (FakeListChatModel) que
devuelve respuestas escritas a mano, y para Claude solo mira la petición que
se enviaría, sin enviarla.

    python experimentos.py
"""

import os
import time
import warnings
from typing import Literal, Optional

from langchain_core.language_models import FakeListChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import BaseModel, ValidationError

os.environ.setdefault("ANTHROPIC_API_KEY", "clave-falsa")  # solo para crear ChatAnthropic


def titulo(t: str) -> None:
    print(f"\n=== {t} ===")


def error(e: Exception) -> str:
    msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
    return f"{type(e).__name__}: {str(msg).strip().splitlines()[0][:110]}"


plantilla = ChatPromptTemplate.from_messages([
    ("system", "Eres un profesor."),
    ("human", "Explícame {tema}"),
])


class PreguntaRepaso(BaseModel):
    pregunta: str
    respuesta_correcta: str
    dificultad: Literal["fácil", "media", "difícil"]


JSON_OK = '{"pregunta": "¿Qué usan las plantas?", "respuesta_correcta": "Luz, agua y CO2", "dificultad": "fácil"}'

# 1. Cambiar el orden de las piezas -------------------------------------------
titulo("1. Orden cambiado: plantilla | parser | modelo")
falso = FakeListChatModel(responses=["La fotosíntesis es..."])
try:
    (plantilla | StrOutputParser() | falso).invoke({"tema": "fotosíntesis"})
except Exception as e:
    print(error(e))
print("Pieza a pieza:",
      type(plantilla.invoke({"tema": "x"})).__name__, "->",
      type(falso.invoke("x")).__name__, "-> str")
print("El parser esperaba un AIMessage y le llegó un ChatPromptValue. El | no comprueba nada al montar.")

# 2. Un hueco mal escrito -----------------------------------------------------
titulo("2. Hueco mal escrito: {'tma': ...} en vez de {'tema': ...}")
try:
    plantilla.invoke({"tma": "fotosíntesis"})
except Exception as e:
    print(error(e))

# 3. StrOutputParser frente a Pydantic con la MISMA respuesta -----------------
titulo("3. Misma respuesta JSON: StrOutputParser vs PydanticOutputParser")
texto = (plantilla | FakeListChatModel(responses=[JSON_OK]) | StrOutputParser()).invoke({"tema": "x"})
try:
    texto["dificultad"]
except Exception as e:
    print("Str -> str | texto['dificultad'] ->", error(e))
parser = PydanticOutputParser(pydantic_object=PreguntaRepaso)
obj = (plantilla | FakeListChatModel(responses=[JSON_OK]) | parser).invoke({"tema": "x"})
print("Pydantic ->", type(obj).__name__, "| obj.dificultad ->", obj.dificultad)

# 4. El modelo responde en texto libre ----------------------------------------
titulo("4. El modelo responde en texto libre (sin JSON)")
libre = "Pregunta: ¿Qué usan las plantas? Respuesta: luz. Dificultad: fácil"
print("Str no se queja ->", (plantilla | FakeListChatModel(responses=[libre]) | StrOutputParser()).invoke({"tema": "x"})[:35], "...")
try:
    (plantilla | FakeListChatModel(responses=[libre]) | parser).invoke({"tema": "x"})
except Exception as e:
    print("Pydantic ->", error(e))
print("Mejor que falle aquí que tres funciones más abajo con un dato mal leído.")

# 5. str frente a Literal -----------------------------------------------------
titulo("5. dificultad = 'imposible': campo str vs Literal")


class ConStr(BaseModel):
    pregunta: str
    respuesta_correcta: str
    dificultad: str


malo = JSON_OK.replace("fácil", "imposible")
print("str ->", PydanticOutputParser(pydantic_object=ConStr).invoke(malo).dificultad, "(pasa sin avisar)")
try:
    parser.invoke(malo)
except Exception as e:
    print("Literal ->", error(e))

# 6. bool obligatorio frente a Optional ---------------------------------------
titulo("6. El anuncio no dice si el precio es negociable")


class AnuncioObligatorio(BaseModel):
    precio: int
    negociable: bool


class AnuncioOpcional(BaseModel):
    precio: int
    negociable: Optional[bool] = None


try:
    AnuncioObligatorio(precio=7500)
except ValidationError as e:
    print("bool ->", str(e).splitlines()[2].strip().split(" [")[0], "(falta el campo)")
print("Optional ->", AnuncioOpcional(precio=7500))
print("Con bool obligatorio, un modelo real se vería forzado a inventarse true o false.")

# 7. Números escritos como texto ----------------------------------------------
titulo("7. precio = '7.500 €' frente a '7500'")
try:
    AnuncioOpcional(precio="7.500 €")
except ValidationError as e:
    print("'7.500 €' ->", str(e).splitlines()[2].strip()[:70])
print("'7500' ->", AnuncioOpcional(precio="7500").precio, "(Pydantic convierte el texto a int)")

# 8. batch frente a un for -----------------------------------------------------
titulo("8. batch vs for (5 entradas, 0,3 s cada una)")
lento_falso = plantilla | FakeListChatModel(responses=["ok"] * 10, sleep=0.3) | StrOutputParser()
entradas = [{"tema": t} for t in "abcde"]
t0 = time.time(); [lento_falso.invoke(x) for x in entradas]; a = time.time() - t0
t0 = time.time(); lento_falso.batch(entradas); b = time.time() - t0
print(f"Modelo falso -> for: {a:.2f} s | batch: {b:.2f} s  (su batch es un for por dentro)")
red = RunnableLambda(lambda x: (time.sleep(0.3), "ok")[1])  # simula la espera de red
t0 = time.time(); [red.invoke(x) for x in entradas]; a = time.time() - t0
t0 = time.time(); red.batch(entradas); b = time.time() - t0
print(f"Espera de red -> for: {a:.2f} s | batch: {b:.2f} s")

# 9. Una herramienta sin docstring ---------------------------------------------
titulo("9. @tool sin docstring")
try:
    @tool
    def coste(km: int) -> float:
        return km * 0.1
except Exception as e:
    print(error(e))
print("El docstring es la descripción que lee el modelo para decidir si usarla.")

# 10 y 11. Lo que se envía de verdad a Claude ---------------------------------
from langchain_anthropic import ChatAnthropic  # noqa: E402

titulo("10. temperature=0 con Claude Opus 5.5")
m = ChatAnthropic(model="claude-opus-5-5", temperature=0, max_tokens=100)
peticion = m._get_request_payload("hola")
print("Campos que se envían:", sorted(peticion))
print("temperature no está: LangChain la quita sin avisar porque este modelo no la admite.")

titulo("11. with_structured_output: method por defecto vs json_schema")
m = ChatAnthropic(model="claude-opus-5-5", max_tokens=100)
for metodo in ["function_calling", "json_schema"]:
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        cadena = m.with_structured_output(PreguntaRepaso, method=metodo)
    kwargs = cadena.first.kwargs
    p = m._get_request_payload("hola", **kwargs)
    aviso = "con aviso: no puede forzar la herramienta" if avisos else "sin avisos"
    print(f"{metodo:17} -> envía herramienta={'tools' in p} | la obliga={p.get('tool_choice') is not None} | "
          f"esquema nativo={'output_config' in p} | {aviso}")
