from fastapi import FastAPI, HTTPException, Depends, Request
from datetime import datetime, timedelta
from pydantic import BaseModel
import secrets
from vllm import LLM, SamplingParams

# Initialiser le modèle LLM
llm = LLM(model="DevSelego/Mistral7b_summarizer_v2")
sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1000)

app = FastAPI()

# Stockage des clés temporaires
temp_keys = {}

# Classe pour les requêtes de génération
class PromptRequest(BaseModel):
    prompt: str  # Modifier ici pour accepter un seul prompt

# Classe pour les requêtes de token
class TokenRequest(BaseModel):
    token: str

# Fonction pour vérifier la clé dans l'en-tête
def verify_key(request: Request):
    token = request.headers.get('x-access-token')
    if token in temp_keys and temp_keys[token] > datetime.utcnow():
        return True
    raise HTTPException(status_code=403, detail="Clé invalide ou expirée")

# Route pour obtenir une clé temporaire
@app.post("/get_key")
def get_key(token_request: TokenRequest):
    if token_request.token == "votre_token_secret":
        temp_key = secrets.token_urlsafe()
        temp_keys[temp_key] = datetime.utcnow() + timedelta(days=1)
        return {"key": temp_key}
    raise HTTPException(status_code=403, detail="Token invalide")

# Route pour générer une réponse
@app.post("/generate")
def generate(prompt_request: PromptRequest, request: Request = Depends(verify_key)):
    modified_prompt = f"### OBJECTIF\nSynthétiser un texte en moins de 400 caractères.\n### INSTRUCTUTION\n- Résumer le texte en moins de 400 caractères.\n- Garde les éléments essentiels à la compréhension et au contexte.\n- Rédige le résumé à la première personne en te plaçant du point de vu de la personne ayant porté le projet.\n- Si le Texte est en anglais, le Résumé doit être écrit en anglais.\n- Si le Texte est en français, le Résumé doit être écrit en français.\n###Texte\n{prompt_request.prompt}\n###Résumé\n"
    output = llm.generate([modified_prompt], sampling_params)
    return {"response": {"prompt": prompt_request.prompt, "response": output[0].outputs[0].text}}
