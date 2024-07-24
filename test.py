from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import re

app = FastAPI()

# Configuración del modelo
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

sentiment_analysis = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

groserias = [
    "puto", "puta", "pendejo", "pendeja", "idiota", "imbecil", "estupido", "estupida",
    "mierda", "cabron", "cabrona", "verga", "vergota", "vergudo", "culero", "culera",
    "pendejada", "joto", "mampo", "mampa", "perra", "chinga", "maricon",
    "pendejote", "putazo", "mamon", "mamona"
]


class TextInput(BaseModel):
    text: str


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def contiene_groserias(text):
    for palabra in groserias:
        if palabra in text:
            return True
    return False


def convertir_calificacion(label):
    calificacion = (int(label) - 1) * 2.5
    return calificacion


def asignar_etiqueta(score):
    if score <= 4:
        return "Negativo"
    elif 4 < score <= 6:
        return "Neutro"
    else:
        return "Positivo"


@app.post("/analyze-sentiment/")
async def analyze_sentiment(input: TextInput):
    texto_procesado = preprocess_text(input.text)
    if contiene_groserias(texto_procesado):
        raise HTTPException(status_code=400, detail="El texto contiene groserías.")
    else:
        resultado = sentiment_analysis([texto_procesado])[0]
        calificacion = convertir_calificacion(resultado['label'][0])
        etiqueta = asignar_etiqueta(calificacion)
        return {
            "sentimiento": etiqueta,
            "calificación": f"{calificacion:.1f}",
        }
