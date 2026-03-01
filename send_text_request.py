from pathlib import Path

import requests
from token_utils import get_valid_token

RUN_URL = "http://localhost:8001/run"
TOKEN_FILE = Path("gcp_token.json")
APP_API_KEY = ""

# Edit this variable with the text you want to send.
TEXT = "¿Qué archivos hay en la carpeta a la pingüina?"
# TEXT = "Lista las primeras cuatro presentaciones." 
# TEXT = "Cuando fue modificado el documento la humanidad final feli?" 
# TEXT = "Lista los primeros 5 items. Además, si hay un archivo con el nombre 'qwerty123456789' en esa lista, devuelve ID del item. En caso contrario, devuelve además la lista de los 5 items SIGUIENTES (es decir, del 6 al 10)" 


extra_info = "This is information that may or may not be useful: " + str([{"name":"A La Pingïuina","id":"1gfXny7mSdqu--NbVYaNjCGeri9GKNDfi","type":"folder","location":"/"},{"name":"asd","id":"1Dvrnf__EyxKeS0TCkwJtl3aS7xdPM9u574Lr0hZYAaE","type":"spreadsheet","location":"/A La Pingïuina/"}])

def main() -> None:
    token = get_valid_token(TOKEN_FILE)
    body = {
        "text": TEXT,
        "mcp_token": token,
        "user_extra_info": extra_info
    }
    headers = {"X-API-Key": APP_API_KEY}

    response = requests.post(RUN_URL, json=body, headers=headers, timeout=120)
    print(f"Status: {response.status_code}")
    print(response.text)


if __name__ == "__main__":
    main()
