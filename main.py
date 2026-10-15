"""
Mini proyecto de LangChain: "Explicador de conceptos".

Le das un tema (por ejemplo "fotosíntesis") y el programa:
  1. Le pide al modelo una explicación sencilla del tema.
  2. Con esa explicación, genera una pregunta de repaso ESTRUCTURADA
     (pregunta + respuesta + dificultad), validada con Pydantic.

Así se ve la idea central de LangChain: unir piezas (prompt -> modelo -> parser)
en una "cadena" (chain) usando el operador |.
"""

import os
import sys
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 1) Cargar la API key desde el archivo .env (ANTHROPIC_API_KEY=...)
load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit("Falta ANTHROPIC_API_KEY. Copia .env.example a .env y pon tu clave.")

# 2) El modelo de lenguaje (LLM). Se puede cambiar con la variable MODELO en .env
modelo = ChatAnthropic(
    model=os.getenv("MODELO", "claude-opus-5-5"),
    max_tokens=4000,  # límite de tokens (palabras aprox.) de la respuesta
)


# 3) El esquema de salida con Pydantic: define QUÉ campos debe devolver el modelo
#    y de qué tipo. Si la respuesta no cumple el esquema, Pydantic lanza un error.
class PreguntaRepaso(BaseModel):
    pregunta: str = Field(description="Pregunta corta de repaso sobre la explicación")
    respuesta_correcta: str = Field(description="Respuesta esperada, en una frase")
    dificultad: Literal["fácil", "media", "difícil"] = Field(description="Dificultad de la pregunta")


# 4) Las plantillas de prompt. Lo que va entre {llaves} se rellena después.
prompt_explicar = ChatPromptTemplate.from_messages([
    ("system", "Eres un profesor paciente. Explicas en español, con palabras sencillas."),
    ("human", "Explícame '{tema}' en máximo 3 frases, como si tuviera 12 años."),
])

prompt_pregunta = ChatPromptTemplate.from_messages([
    ("system", "Eres un profesor que hace preguntas de repaso en español."),
    ("human", "Basándote en esta explicación, crea UNA pregunta de repaso:\n\n{explicacion}"),
])

# 5) Las cadenas:
#    - La primera termina en StrOutputParser -> devuelve texto (str).
#    - La segunda usa with_structured_output -> devuelve un objeto PreguntaRepaso.
#      method="json_schema" usa la salida estructurada nativa de Claude: el modelo
#      está obligado a responder con un JSON que cumple el esquema.
cadena_explicar = prompt_explicar | modelo | StrOutputParser()
cadena_pregunta = prompt_pregunta | modelo.with_structured_output(PreguntaRepaso, method="json_schema")


def main() -> None:
    tema = " ".join(sys.argv[1:]) or input("¿Qué tema quieres entender? ")

    explicacion = cadena_explicar.invoke({"tema": tema})
    print("\n📘 Explicación:\n" + explicacion)

    repaso: PreguntaRepaso = cadena_pregunta.invoke({"explicacion": explicacion})
    print(f"\n❓ Pregunta de repaso ({repaso.dificultad}):\n{repaso.pregunta}")
    input("\n(Pulsa Enter para ver la respuesta) ")
    print(f"✅ {repaso.respuesta_correcta}")


if __name__ == "__main__":
    main()
