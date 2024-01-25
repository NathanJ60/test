# Importation des bibliothèques nécessaires
from pyannote.audio import Pipeline
from pydub import AudioSegment
import json
import os
import whisper

# Charger le modèle Whisper une seule fois
model = whisper.load_model("base")

# Fonction de découpage audio
def split_audio(file_path, start_time, end_time, output_file):
    """Split the audio file from start_time to end_time and save it to output_file."""
    audio = AudioSegment.from_file(file_path)
    start_time_ms = start_time * 1000  # convert to milliseconds
    end_time_ms = end_time * 1000
    extract = audio[start_time_ms:end_time_ms]
    extract.export(output_file, format="wav")

# Fonction de transcription (à remplacer par votre modèle Whisper)
def transcribe_audio_local(audio_file_path):
    """
    Transcrit un fichier audio local en utilisant Whisper.
    """
    # Vérifie si le fichier existe
    if not os.path.exists(audio_file_path):
        return "Fichier audio non trouvé."

    # Transcription de l'audio
    result = model.transcribe(audio_file_path, language="fr")

    # Retourne le texte transcrit
    return result["text"]

# Initialisation du pipeline pyannote
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="votre_token")

# Chemin vers le fichier audio original
audio_file = "chemin_vers_le_fichier_audio.mp3"

# Application du pipeline de diarisation
diarization = pipeline(audio_file)

# Initialisation des variables
current_speaker = None
current_start = None
current_end = None
segment_count = 1
transcription_json = {}

# Traitement des segments audio et transcription
for turn, _, speaker in diarization.itertracks(yield_label=True):
    if current_speaker is None:
        current_speaker = speaker
        current_start = turn.start
        current_end = turn.end
    elif speaker == current_speaker:
        current_end = turn.end
    else:
        output_file = f"temp_speaker_{current_speaker}_segment_{segment_count}.wav"
        split_audio(audio_file, current_start, current_end, output_file)
        transcribed_text = transcribe_audio_local(output_file)
        transcription_json.setdefault(current_speaker, []).append(transcribed_text)
        os.remove(output_file)  # Suppression du fichier temporaire après la transcription
        segment_count += 1
        current_speaker = speaker
        current_start = turn.start
        current_end = turn.end

# Traitement du dernier segment
if current_speaker is not None:
    output_file = f"temp_speaker_{current_speaker}_segment_{segment_count}.wav"
    split_audio(audio_file, current_start, current_end, output_file)
    transcribed_text = transcribe_audio_local(output_file)
    transcription_json.setdefault(current_speaker, []).append(transcribed_text)
    os.remove(output_file)  # Suppression du fichier temporaire après la transcription

# Sauvegarde du fichier JSON
with open('transcription.json', 'w') as json_file:
    json.dump(transcription_json, json_file, ensure_ascii=False, indent=4)

# Le résultat est un fichier JSON avec la transcription associée à chaque locuteur.
