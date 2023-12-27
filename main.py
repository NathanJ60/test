from fastapi import FastAPI, HTTPException, Depends
from transformers import PhiForCausalLM, AutoTokenizer
from datetime import datetime, timedelta
from pydantic import BaseModel
import secrets

# Modèle et Tokenizer
model = PhiForCausalLM.from_pretrained("DevSelego/Mistral7b_summarizer_v2")
tokenizer = AutoTokenizer.from_pretrained("DevSelego/Mistral7b_summarizer_v2")

app = FastAPI()

# Stockage des clés temporaires (pour simplifier, on utilise un dictionnaire)
temp_keys = {}

# Classe pour les requêtes de génération
class Prompt(BaseModel):
    text: str

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
    # Ici, vérifiez le token (dans un cas réel, vous devriez vérifier un token réel)
    if token_request.token == "votre_token_secret":
        temp_key = secrets.token_urlsafe()
        temp_keys[temp_key] = datetime.utcnow() + timedelta(days=1)
        return {"key": temp_key}
    raise HTTPException(status_code=403, detail="Token invalide")

# Route pour générer une réponse
@app.post("/generate")
def generate(prompt: Prompt, key: str = Depends(verify_key)):
    tokens = tokenizer(prompt.text, return_tensors="pt")
    generated_output = model.generate(**tokens, use_cache=True, max_new_tokens=10)
    return {"response": tokenizer.batch_decode(generated_output)[0]}
