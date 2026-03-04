import json
import threading
import time
import wave
from io import BytesIO
from pathlib import Path

import pyaudio
import requests
from token_utils import get_valid_token

RUN_AUDIO_URL = "http://localhost:8001/run-audio"

TOKEN_FILE = Path("gcp_token.json")
APP_API_KEY = ""



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
    user_guidelines = """Phrases like 'nueva idea', 'idea suelta', 'new idea', or 'nueva idea suelta' are trigger phrases — they are instructions, not content.
    When you see one: (1) the file content is ONLY the text that follows the trigger phrase and any topic qualifier — never include the trigger or topic in the file body; (2) if a topic is mentioned (e.g. 'nueva idea suelta sobre la consciencia'), create the file inside a subfolder named after the topic (e.g. 'consciencia') within 'Ideas Sueltas', creating that subfolder first if it does not exist; (3) if no topic is mentioned, create the file directly in 'Ideas Sueltas'; (4) always create the file directly in its final destination — never create it elsewhere and move it."""


    files = {"audio": ("recording.wav", audio_bytes, "audio/wav")}
    data = {"mcp_token": token, "known_drive_items": json.dumps(known_drive_items), "user_guidelines": user_guidelines, "planner_model": None}
    headers = {"X-API-Key": APP_API_KEY}
    start = time.time()
    response = requests.post(RUN_AUDIO_URL, files=files, data=data, headers=headers, timeout=180)
    print(f"Request took {time.time() - start:.1f}s")
    print(f"Status: {response.status_code}")
    print(response.text)


if __name__ == "__main__":
    main()
