"""
Mini proyecto de LangChain: "Explicador de conceptos".

Le das un tema (por ejemplo "fotosíntesis") y el programa:
  1. Le pide al modelo una explicación sencilla del tema.
  2. Con esa explicación, genera una pregunta de repaso.

Así se ve la idea central de LangChain: unir piezas (prompt -> modelo -> parser)
en una "cadena" (chain) usando el operador |.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1) Cargar la API key desde el archivo .env (ANTHROPIC_API_KEY=...)
load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit("Falta ANTHROPIC_API_KEY. Copia .env.example a .env y pon tu clave.")

# 2) El modelo de lenguaje (LLM). Se puede cambiar con la variable MODELO en .env
modelo = ChatAnthropic(
    model=os.getenv("MODELO", "claude-opus-5-5"),
    max_tokens=4000,  # límite de tokens (palabras aprox.) de la respuesta
)

# 3) Las plantillas de prompt. Lo que va entre {llaves} se rellena después.
prompt_explicar = ChatPromptTemplate.from_messages([
    ("system", "Eres un profesor paciente. Explicas en español, con palabras sencillas."),
    ("human", "Explícame '{tema}' en máximo 3 frases, como si tuviera 12 años."),
])

prompt_pregunta = ChatPromptTemplate.from_messages([
    ("system", "Eres un profesor que hace preguntas de repaso."),
    ("human", "Basándote en esta explicación, escribe UNA pregunta corta de repaso:\n\n{explicacion}"),
])

# 4) El parser convierte la respuesta del modelo en texto plano (str).
parser = StrOutputParser()

# 5) Las cadenas: prompt -> modelo -> parser
cadena_explicar = prompt_explicar | modelo | parser
cadena_pregunta = prompt_pregunta | modelo | parser


def main() -> None:
    tema = " ".join(sys.argv[1:]) or input("¿Qué tema quieres entender? ")

    explicacion = cadena_explicar.invoke({"tema": tema})
    print("\n📘 Explicación:\n" + explicacion)

    pregunta = cadena_pregunta.invoke({"explicacion": explicacion})
    print("\n❓ Pregunta de repaso:\n" + pregunta)


if __name__ == "__main__":
    main()
