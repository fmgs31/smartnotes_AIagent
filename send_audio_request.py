import threading
import time
import wave
from io import BytesIO
from pathlib import Path

import pyaudio
import requests
from token_utils import get_valid_token

RUN_AUDIO_URL = "http://localhost:8001/run"

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

    files = {"audio": ("recording.wav", audio_bytes, "audio/wav")}
    data = {"mcp_token": token}
    headers = {"X-API-Key": APP_API_KEY}
    start = time.time()
    response = requests.post(RUN_AUDIO_URL, files=files, data=data, headers=headers, timeout=180)
    print(f"Request took {time.time() - start:.1f}s")
    print(f"Status: {response.status_code}")
    print(response.text)


if __name__ == "__main__":
    main()
