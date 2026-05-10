import os
import re
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def record_audio(duration=5, fs=44100) -> str:
    """
    Records audio from the default microphone and saves to a temporary WAV file.
    """
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        
        temp_dir = tempfile.gettempdir()
        wav_path = os.path.join(temp_dir, 'mtu_voice_query.wav')
        write(wav_path, fs, recording)
        return wav_path
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None

def transcribe_audio(file_path: str) -> str:
    """
    Uses Groq's fast Whisper API to transcribe the audio.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY is not set."
        
    client = Groq(api_key=api_key)
    
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
              file=(os.path.basename(file_path), file.read()),
              model="whisper-large-v3",
              response_format="text",
              temperature=0.0
            )
        return transcription.strip()
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return f"Transcription error: {e}"

def parse_intent(text: str) -> dict:
    """
    Parses the transcribed text to find the intent and target engine ID.
    """
    text_lower = text.lower()
    
    # Try to find a number after "engine"
    match = re.search(r'engine\s+(\d+)', text_lower)
    if not match:
        # Just look for any number
        match = re.search(r'\b(\d+)\b', text_lower)
        
    engine_id = int(match.group(1)) if match else None
    
    intent = "unknown"
    if "status" in text_lower or "how is" in text_lower or "check" in text_lower:
        intent = "status"
    elif "what if" in text_lower or "delay" in text_lower or "simulate" in text_lower:
        intent = "simulate"
    elif "work order" in text_lower or "generate" in text_lower or "report" in text_lower:
        intent = "work_order"
        
    return {
        "intent": intent,
        "engine_id": engine_id,
        "raw_text": text
    }

def handle_voice_query(duration=5):
    """
    Complete flow: Record -> Transcribe -> Parse.
    """
    wav_path = record_audio(duration)
    if not wav_path:
        return {"error": "Failed to record."}
        
    text = transcribe_audio(wav_path)
    if text.startswith("Error"):
        return {"error": text}
        
    return parse_intent(text)
