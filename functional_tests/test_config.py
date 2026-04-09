"""
Static configuration and helpers shared across all functional tests.
"""
from functional_tests.fake_drive import FakeDrive, MIME_DOC

# ─── User guidelines (copied from send_text_request.py) ───────────────────────

USER_GUIDELINES = """\
Trigger mappings (check in this order):
    - 'idea para un cuento', 'nueva idea para un cuento': create a document in 'Cuentos/Ideas/'.
    - 'nueva idea', 'idea suelta', 'new idea', 'nueva idea suelta' (only if the phrase does NOT contain 'para un cuento'): create a document in 'Ideas Sueltas'. If a topic is mentioned (e.g. 'nueva idea suelta sobre la consciencia'), create instead inside a subfolder named after the topic (e.g. 'consciencia') within 'Ideas Sueltas', unless a similar subfolder already exists.
    - 'idea para el juego de mesa', 'idea suelta para el juego de mesa', 'idea nueva el juego de mesa': append a cell at the bottom of one of the tabs called "Ideas suelta [X]" in the spreadsheet 'PrototipoJuego/Prototipo Juego', where [X] is the highest number from the ones available. New ideas go into a new cell in the same column, just below the last one. Annotations to an idea go into the immediate column to the right. If there is no room for a new cell in that column, create a new tab called "Ideas suelta [X+1]".
    - 'adding annotation about the last idea for the boardgame', 'aclaración de la última idea sobre el juego de mesa': in the same spreadsheet and tab as above, write the content in the cell immediately to the right of the last filled cell in the ideas column.
    - 'Terapia', 'Idea sobre terapia', 'idea de autoanálisis': append (2 empty lines + current date + 1 empty line + text dictated by the user + 1 empty line) to the bottom of the document 'Terapia-Autoanálisis/Terapia'.
"""

# ─── Stable content for assertions ────────────────────────────────────────────

TERAPIA_INITIAL_CONTENT = "Sesión del 2025-01-15\n\nHoy hablamos sobre mi infancia.\n"

# ─── Known drive items passed to the agent (mirrors production usage) ─────────

def build_known_items(drive: FakeDrive) -> list[dict]:
    """
    Build the known_drive_items list from the live drive index.
    The IDs must match what FakeDrive assigned so the LLM can use them.
    """
    entries = [
        ("Ideas Sueltas",        "folder", "/"),
        ("Terapia-Autoanálisis", "folder", "/"),
        ("Cuentos",              "folder", "/"),
        ("PrototipoJuego",       "folder", "/"),
    ]
    result = []
    for name, type_, location in entries:
        item = drive.find_by_name(name)
        if item:
            result.append({"name": name, "id": item["id"], "type": type_, "location": location})
    return result
