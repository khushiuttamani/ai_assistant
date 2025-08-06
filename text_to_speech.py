from gtts import gTTS
from pydub import AudioSegment
import platform
import subprocess
import os

def text_to_speech_with_gtts(input_text, output_filepath_mp3, output_filepath_wav):
    language = "en"
    audioobj = gTTS(text=input_text, lang=language, slow=False)
    audioobj.save(output_filepath_mp3)

    sound = AudioSegment.from_mp3(output_filepath_mp3)
    sound.export(output_filepath_wav, format="wav")

    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(['afplay', output_filepath_wav])
        elif os_name == "Windows":
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath_wav}").PlaySync();'])
        elif os_name == "Linux":
            subprocess.run(['aplay', output_filepath_wav])
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")

# This is a placeholder function since ElevenLabs is mentioned in the original main.py
def text_to_speech_with_elevenlabs(input_text, output_filepath):
    print("ElevenLabs text-to-speech not implemented. Using gTTS as a fallback.")
    text_to_speech_with_gtts(input_text, output_filepath, "temp_elevenlabs_output.wav")