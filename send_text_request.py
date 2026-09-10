import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from token_utils import get_valid_token

load_dotenv()

# RUN_AUDIO_URL = "https://smartnotes-gateway-131932405877.australia-southeast1.run.app/run-audio"
RUN_URL = "http://localhost:8001/run"
TOKEN_FILE = Path("gcp_token.json")
APP_API_KEY = os.environ["APP_API_KEY"]

# Edit this variable with the text you want to send.
TEXT = "¿Qué archivos hay en la carpeta a la pingüina?"
TEXT = "Lista las primeras cuatro carpetas." 
# TEXT = "Cuando fue modificado el documento la humanidad final feli?" 
# TEXT = "Lista los primeros 5 items. Además, si hay un archivo con el nombre 'qwerty123456789' en esa lista, devuelve ID del item. En caso contrario, devuelve además la lista de los 5 items SIGUIENTES (es decir, del 6 al 10)" 
# TEXT = "Nueva idea suelta sobre la consciencia. La consciencia percibe cosas de en medio de la red, no sólo la salida final"
# TEXT = "create a document called my conversations with Richard and inside put that they were nice using red font"

# TEXT = "in the document called my conversations with Richard, append the conversation was a video call"

# TEXT = "in the document called my conversations with Richard, append the conversation was a video call"
TEXT = "que hay en el documento abriendo las alas"


known_drive_items = [{"name":"A La Pingïuina","id":"1gfXny7mSdqu--NbVYaNjCGeri9GKNDfi","type":"folder","location":"/"},
                     {"name":"Ideas sueltas","id":"1mERgByV9OKSEh5VMOihvFp2HSWy_UaI8","type":"folder","location":"/"}]
user_guidelines = """Trigger mappings (check in this order):
    - 'idea para un cuento', 'nueva idea para un cuento': create a document in 'Cuentos/Ideas/'.
    - 'nueva idea', 'idea suelta', 'new idea', 'nueva idea suelta' (only if the phrase does NOT contain 'para un cuento'): create a document in 'Ideas Sueltas'. If a topic is mentioned (e.g. 'nueva idea suelta sobre la consciencia'), create instead inside a subfolder named after the topic (e.g. 'consciencia') within 'Ideas Sueltas', unless a similar subfolder already exists.
    - 'idea para el juego de mesa', 'idea suelta para el juego de mesa', 'idea nueva el juego de mesa': append a cell at the bottom of one of the tabs called "Ideas suelta [X]" in the spreadsheet 'PrototipoJuego/Prototipo Juego', where [X] is the highest number from the ones available. New ideas go into a new cell in the same column, just below the last one. Annotations to an idea go into the immediate column to the right. If there is no room for a new cell in that column, create a new tab called "Ideas suelta [X+1]".
    - 'adding annotation about the last idea for the boardgame', 'aclaración de la última idea sobre el juego de mesa': in the same spreadsheet and tab as above, write the content in the cell immediately to the right of the last filled cell in the ideas column.
    - 'Terapia', 'Idea sobre terapia', 'idea de autoanálisis': append (2 empty lines + current date + 1 empty line + text dictated by the user + 1 empty line) to the bottom of the document 'Terapia-Autoanálisis/Terapia'.
   """
def main() -> None:
    token = get_valid_token(TOKEN_FILE)
    body = {
        "text": TEXT,
        "mcp_token": token,
        "known_drive_items": known_drive_items,
        "user_guidelines": user_guidelines,
        "planner_model": None
    }
    headers = {"X-API-Key": APP_API_KEY}

    # ----- POST WITHOUT STREAM ------
    # response = requests.post(RUN_URL, json=body, headers=headers, timeout=120)
    # print(f"Status: {response.status_code}")
    # print(response.text)

    # ------ POST WIT STREAM ------
    response = requests.post(RUN_URL, json=body, headers=headers, timeout=120, stream=True)
    print(f"Status: {response.status_code}")
    for line in response.iter_lines():
        if line:
            line = line.decode() if isinstance(line, bytes) else line
            if line.startswith("data: "):
                print(json.loads(line[6:]))


if __name__ == "__main__":
    main()
