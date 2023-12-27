from fastapi import FastAPI, HTTPException, Depends
from datetime import datetime, timedelta
from pydantic import BaseModel
import secrets
from vllm import LLM, SamplingParams

# Initialiser le modèle LLM
llm = LLM(model="DevSelego/Mistral7b_summarizer_v2")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

app = FastAPI()

# Stockage des clés temporaires
temp_keys = {}

# Classe pour les requêtes de génération
class PromptRequest(BaseModel):
    prompts: list[str]

# Classe pour les requêtes de token
class TokenRequest(BaseModel):
    token: str

# Fonction pour vérifier la clé
def verify_key(key: str):
    if key in temp_keys and temp_keys[key] > datetime.utcnow():
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
def generate(prompt_request: PromptRequest, key: str = Depends(verify_key)):
    modified_prompts = [f"### OBJECTIF\nSynthétiser un texte en moins de 400 caractères.\n### INSTRUCTUTION\n- Résumer le texte en moins de 400 caractères.\n- Garde les éléments essentiels à la compréhension et au contexte.\n- Rédige le résumé à la première personne en te plaçant du point de vu de la personne ayant porté le projet.\n- Si le Texte est en anglais, le Résumé doit être écrit en anglais.\n- Si le Texte est en français, le Résumé doit être écrit en français.\n###Texte\n{prompt}\n###Résumé\n" for prompt in prompt_request.prompts]
    outputs = llm.generate(modified_prompts, sampling_params)
    return {"responses": [{"prompt": output.prompt, "response": output.outputs[0].text} for output in outputs]}
