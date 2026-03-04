from pathlib import Path

import requests
from token_utils import get_valid_token

# RUN_AUDIO_URL = "https://smarnotes-gateway-131932405877.australia-southeast1.run.app/run-audio"
RUN_URL = "http://localhost:8001/run"
TOKEN_FILE = Path("gcp_token.json")
APP_API_KEY = ""

# Edit this variable with the text you want to send.
TEXT = "¿Qué archivos hay en la carpeta a la pingüina?"
# TEXT = "Lista las primeras cuatro presentaciones." 
# TEXT = "Cuando fue modificado el documento la humanidad final feli?" 
# TEXT = "Lista los primeros 5 items. Además, si hay un archivo con el nombre 'qwerty123456789' en esa lista, devuelve ID del item. En caso contrario, devuelve además la lista de los 5 items SIGUIENTES (es decir, del 6 al 10)" 
TEXT = "Nueva idea suelta sobre la consciencia. La consciencia percibe cosas de en medio de la red, no sólo la salida final"


known_drive_items = [{"name":"A La Pingïuina","id":"1gfXny7mSdqu--NbVYaNjCGeri9GKNDfi","type":"folder","location":"/"},
                     {"name":"Ideas sueltas","id":"1mERgByV9OKSEh5VMOihvFp2HSWy_UaI8","type":"folder","location":"/"}]
user_guidelines = """Phrases like 'nueva idea', 'idea suelta', 'new idea', or 'nueva idea suelta' are trigger phrases — they are instructions, not content.
When you see one: (1) the file content is ONLY the text that follows the trigger phrase and any topic qualifier — never include the trigger or topic in the file body; (2) if a topic is mentioned (e.g. 'nueva idea suelta sobre los perros'), create the file inside a subfolder named after the topic (e.g. 'perros') within 'Ideas Sueltas', creating that subfolder first if it does not exist; (3) if no topic is mentioned, create the file directly in 'Ideas Sueltas'; (4) always create the file directly in its final destination — never create it elsewhere and move it."""
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
