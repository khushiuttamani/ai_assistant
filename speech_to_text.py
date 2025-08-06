import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Function to record audio from the microphone and save it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            logging.info(f"Audio saved to {file_path}")
            return True

    except sr.WaitTimeoutError:
        logging.error("No speech detected within the timeout period.")
        return False
    except Exception as e:
        logging.error(f"An error occurred during recording: {e}")
        return False


def transcribe_with_groq(audio_filepath):
    load_dotenv()
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        logging.error("GROQ_API_KEY not found. Make sure it is set in your .env file.")
        return "Error: API Key not found."

    client = Groq(api_key=GROQ_API_KEY)
    stt_model = "whisper-large-v3"
    
    try:
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        return transcription.text
    except FileNotFoundError:
        logging.error(f"Audio file not found at: {audio_filepath}")
        return "Error: Audio file not found."
    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}")
        return f"Error: {e}"