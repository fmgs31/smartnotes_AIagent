import json
import os
import threading
import time
import wave
from io import BytesIO
from pathlib import Path

import pyaudio
import requests
from dotenv import load_dotenv
from token_utils import get_valid_token

load_dotenv()

RUN_AUDIO_URL = "http://localhost:8001/run-audio"

TOKEN_FILE = Path("gcp_token.json")
APP_API_KEY = os.environ["APP_API_KEY"]



CHUNK_SIZE = 1024
CHANNELS = 1
SAMPLE_RATE = 16000
SAMPLE_FORMAT = pyaudio.paInt16

def record_until_enter() -> bytes:
    audio = pyaudio.PyAudio()
    frames: list[bytes] = []
    stop_event = threading.Event()

    stream = audio.open(
        format=SAMPLE_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    def _read_frames() -> None:
        while not stop_event.is_set():
            frames.append(stream.read(CHUNK_SIZE, exception_on_overflow=False))

    print("Press Enter to start recording")
    input()

    thread = threading.Thread(target=_read_frames, daemon=True)
    thread.start()

    print("Press Enter to stop recording")
    input()
    stop_event.set()
    print("Recorded completed")
    thread.join()

    stream.stop_stream()
    stream.close()
    sample_width = audio.get_sample_size(SAMPLE_FORMAT)
    audio.terminate()

    if not frames:
        raise ValueError("No audio was captured")

    wav_buffer = BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"".join(frames))
    wav_buffer.seek(0)
    return wav_buffer.read()


AUDIO_SAVE_DIR = Path("data/audios_sent")


def save_audio(audio_bytes: bytes) -> Path:
    AUDIO_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    filename = AUDIO_SAVE_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}.wav"
    filename.write_bytes(audio_bytes)
    print(f"Audio saved: {filename}")
    return filename


def main() -> None:
    token = get_valid_token(TOKEN_FILE)
    audio_bytes = record_until_enter()
    save_audio(audio_bytes)


    known_drive_items = [{"name":"Ideas sueltas","id":"1mERgByV9OKSEh5VMOihvFp2HSWy_UaI8",
                          "type":"folder","location":"/"}]
    user_guidelines = """
    Trigger mappings (check in this order):
    - 'idea para un cuento', 'nueva idea para un cuento': create a document in 'Cuentos/Ideas/'.
    - 'nueva idea', 'idea suelta', 'new idea', 'nueva idea suelta' (only if the phrase does NOT contain 'para un cuento'): create a document in 'Ideas Sueltas'. If a topic is mentioned (e.g. 'nueva idea suelta sobre la consciencia'), create instead inside a subfolder named after the topic (e.g. 'consciencia') within 'Ideas Sueltas', unless a similar subfolder already exists.
    - 'idea para el juego de mesa', 'idea suelta para el juego de mesa', 'idea nueva el juego de mesa': append a cell at the bottom of one of the tabs called "Ideas suelta [X]" in the spreadsheet 'PrototipoJuego/Prototipo Juego', where [X] is the highest number from the ones available. New ideas go into a new cell in the same column, just below the last one. Annotations to an idea go into the immediate column to the right. If there is no room for a new cell in that column, create a new tab called "Ideas suelta [X+1]".
    - 'adding annotation about the last idea for the boardgame', 'aclaración de la última idea sobre el juego de mesa': in the same spreadsheet and tab as above, write the content in the cell immediately to the right of the last filled cell in the ideas column.
    - 'Terapia', 'Idea sobre terapia', 'idea del psicólogo': append (2 empty lines + current date + 1 empty line + text dictated by the user + 1 empty line) to the bottom of the document 'Terapia-Autoanálisis/Terapia'.
    """


    files = {"audio": ("recording.wav", audio_bytes, "audio/wav")}
    data = {"mcp_token": token, "known_drive_items": json.dumps(known_drive_items), "user_guidelines": user_guidelines, "planner_model": None}
    headers = {"X-API-Key": APP_API_KEY}
    response = requests.post(RUN_AUDIO_URL, files=files, data=data, headers=headers, timeout=180, stream=True)
    print(f"Status: {response.status_code}")
    for line in response.iter_lines():
        if line:
            line = line.decode() if isinstance(line, bytes) else line
            if line.startswith("data: "):
                print(json.loads(line[6:]))


if __name__ == "__main__":
    main()
