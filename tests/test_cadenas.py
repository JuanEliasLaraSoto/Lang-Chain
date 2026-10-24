"""
Tests del explicador de conceptos con un modelo FALSO: no necesitan API key ni gastan dinero.

FakeListChatModel devuelve respuestas escritas a mano, así se comprueba que las
piezas encajan (plantilla -> modelo -> parser) sin depender de lo que conteste Claude.

    pytest -v
"""

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import FakeListChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import ValidationError

from main import PreguntaRepaso, crear_cadenas, prompt_explicar, prompt_pregunta

JSON_OK = '{"pregunta": "¿Qué usan las plantas?", "respuesta_correcta": "Luz, agua y CO2", "dificultad": "fácil"}'


class ModeloFalso(FakeListChatModel):
    """Modelo falso que también sabe hacer with_structured_output.

    Imita a Claude: devuelve el JSON de la respuesta y Pydantic lo valida.
    """

    def with_structured_output(self, schema, **kwargs):
        return self | PydanticOutputParser(pydantic_object=schema)


# --- Plantillas ---------------------------------------------------------------

def test_plantilla_rellena_el_tema():
    mensajes = prompt_explicar.invoke({"tema": "fotosíntesis"}).to_messages()
    assert mensajes[0].type == "system"
    assert "fotosíntesis" in mensajes[1].content


def test_plantilla_falla_si_falta_un_hueco():
    with pytest.raises(KeyError):
        prompt_explicar.invoke({"tma": "fotosíntesis"})


# --- Cadenas LCEL -------------------------------------------------------------

def test_cadena_explicar_devuelve_texto():
    explicar, _ = crear_cadenas(ModeloFalso(responses=["La fotosíntesis es..."]))
    assert explicar.invoke({"tema": "fotosíntesis"}) == "La fotosíntesis es..."


def test_cadena_pregunta_devuelve_objeto_pydantic():
    _, preguntar = crear_cadenas(ModeloFalso(responses=[JSON_OK]))
    repaso = preguntar.invoke({"explicacion": "La fotosíntesis es..."})
    assert isinstance(repaso, PreguntaRepaso)
    assert repaso.dificultad == "fácil"


def test_las_dos_cadenas_encadenadas():
    """La salida de la primera cadena es la entrada de la segunda, como en main()."""
    modelo = ModeloFalso(responses=["La fotosíntesis es...", JSON_OK])
    explicar, preguntar = crear_cadenas(modelo)
    explicacion = explicar.invoke({"tema": "fotosíntesis"})
    repaso = preguntar.invoke({"explicacion": explicacion})
    assert repaso.pregunta == "¿Qué usan las plantas?"


def test_orden_cambiado_falla():
    """Si el parser va antes del modelo, recibe un ChatPromptValue y falla al ejecutar."""
    cadena = prompt_explicar | StrOutputParser() | ModeloFalso(responses=["x"])
    with pytest.raises(ValidationError):
        cadena.invoke({"tema": "fotosíntesis"})


# --- Validación con Pydantic --------------------------------------------------

def test_texto_libre_se_rechaza():
    _, preguntar = crear_cadenas(ModeloFalso(responses=["Pregunta: ¿qué? Dificultad: fácil"]))
    with pytest.raises(OutputParserException):
        preguntar.invoke({"explicacion": "..."})


def test_dificultad_fuera_de_literal_se_rechaza():
    _, preguntar = crear_cadenas(ModeloFalso(responses=[JSON_OK.replace("fácil", "imposible")]))
    with pytest.raises(OutputParserException):
        preguntar.invoke({"explicacion": "..."})


def test_falta_un_campo_se_rechaza():
    with pytest.raises(ValidationError):
        PreguntaRepaso(pregunta="¿Qué?", dificultad="fácil")


def test_esquema_que_se_envia_al_modelo():
    """El modelo nunca ve la clase: ve este esquema JSON."""
    esquema = PreguntaRepaso.model_json_schema()
    assert set(esquema["required"]) == {"pregunta", "respuesta_correcta", "dificultad"}
    assert esquema["properties"]["dificultad"]["enum"] == ["fácil", "media", "difícil"]


def test_prompt_pregunta_recibe_la_explicacion():
    mensajes = prompt_pregunta.invoke({"explicacion": "La fotosíntesis es..."}).to_messages()
    assert "La fotosíntesis es..." in mensajes[1].content
