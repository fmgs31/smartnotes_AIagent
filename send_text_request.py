import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from token_utils import get_valid_token

load_dotenv()

# RUN_AUDIO_URL = "https://smarnotes-gateway-131932405877.australia-southeast1.run.app/run-audio"
RUN_URL = "http://localhost:8001/run"
TOKEN_FILE = Path("gcp_token.json")
APP_API_KEY = os.environ["APP_API_KEY"]

# Edit this variable with the text you want to send.
TEXT = "¿Qué archivos hay en la carpeta a la pingüina?"
# TEXT = "Lista las primeras cuatro presentaciones." 
# TEXT = "Cuando fue modificado el documento la humanidad final feli?" 
# TEXT = "Lista los primeros 5 items. Además, si hay un archivo con el nombre 'qwerty123456789' en esa lista, devuelve ID del item. En caso contrario, devuelve además la lista de los 5 items SIGUIENTES (es decir, del 6 al 10)" 
TEXT = "Nueva idea suelta sobre la consciencia. La consciencia percibe cosas de en medio de la red, no sólo la salida final"


known_drive_items = [{"name":"A La Pingïuina","id":"1gfXny7mSdqu--NbVYaNjCGeri9GKNDfi","type":"folder","location":"/"},
                     {"name":"Ideas sueltas","id":"1mERgByV9OKSEh5VMOihvFp2HSWy_UaI8","type":"folder","location":"/"}]
user_guidelines = """Some phrases are trigger phrases — they are instructions, not content. Rules that apply to all triggers:
    - The file content is ONLY the text that follows the trigger phrase and any topic qualifier — never include the trigger or topic in the file body.
    - Always create the file directly in its final destination — never create it elsewhere and move it.
    - Folder names in the mappings below refer to existing folders. Always look them up by name to get their ID before using them — never assume they don't exist and create new ones.
    - Only create a folder if the lookup confirms it does not exist. Create any missing intermediate folders before creating the file.

    Trigger mappings (check in this order):
    - 'idea para un cuento', 'nueva idea para un cuento': create in 'Cuentos/Ideas'.
    - 'nueva idea', 'idea suelta', 'new idea', 'nueva idea suelta' (only if the phrase does NOT contain 'para un cuento'): create in 'Ideas Sueltas'. If a topic is mentioned (e.g. 'nueva idea suelta sobre la consciencia'), create instead inside a subfolder named after the topic (e.g. 'consciencia') within 'Ideas Sueltas', unless a similar subfolder already exists."""
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

    response = requests.post(RUN_URL, json=body, headers=headers, timeout=120)
    print(f"Status: {response.status_code}")
    print(response.text)


if __name__ == "__main__":
    main()
