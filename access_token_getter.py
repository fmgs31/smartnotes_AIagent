import os.path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import json
# The full list of scopes the MCP server typically uses
SCOPES = [
    # # 'https://www.googleapis.com/auth/gmail.modify',
    # # 'https://www.googleapis.com/auth/calendar',
    # 'https://www.googleapis.com/auth/drive',
    # # 'https://www.googleapis.com/auth/contacts',
    # # 'https://www.googleapis.com/auth/tasks',
    # 'https://www.googleapis.com/auth/spreadsheets',
    # 'https://www.googleapis.com/auth/documents',
    # # 'https://www.googleapis.com/auth/presentations'


    # 'https://www.googleapis.com/auth/userinfo.email'

    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/documents',                                 
    'https://www.googleapis.com/auth/documents.readonly',                         
    'https://www.googleapis.com/auth/spreadsheets',                               
    'https://www.googleapis.com/auth/spreadsheets.readonly',                      
    'https://www.googleapis.com/auth/userinfo.email',                             
    'https://www.googleapis.com/auth/userinfo.profile',                           
    'openid'

    # "openid",
    # "https://www.googleapis.com/auth/userinfo.email",
    # "https://www.googleapis.com/auth/drive",
    # "https://www.googleapis.com/auth/documents",
    # "https://www.googleapis.com/auth/spreadsheets",
    ]

def generate_token():
    creds = None
    # If you have a cached token, it loads it (useful for re-running)
    if os.path.exists('gcp_token.json'):
        creds = Credentials.from_authorized_user_file('gcp_token.json', SCOPES)
        if not creds.valid:
            creds = get_creds()
        elif creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("No need to refersh token")
            return
    else:
        creds = get_creds()
    save_token(creds)


def get_creds(): 
    # ---------------------------------------------------------
    # IMPORTANT: You need your 'client_secret.json' file here.
    # Download it from Google Cloud Console > Credentials
    # ---------------------------------------------------------
    if not os.path.exists('client_secret.json'):
        print("❌ Error: 'client_secret.json' not found!")
        print("Please download it from Google Cloud Console -> Credentials -> OAuth 2.0 Client IDs")
        return

    flow = InstalledAppFlow.from_client_secrets_file(
        'client_secret.json', SCOPES)
    
    # This opens your local browser to log in
    creds = flow.run_local_server(port=0)
    return creds

def save_token(creds):
    # Save for next time

    with open('gcp_token.json', 'w') as tokenfw:
        # json.dump(creds.to_json(), tokenfw)
        id_token = creds.id_token
        print("ID TOKEN:", id_token)
        tokenfw.write(creds.to_json())

    # ---------------------------------------------------------
    # PRINT THE TOKEN
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("✅ ACCESS TOKEN GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"\n{creds.token}\n")
    print("="*60)
    print("Copy the string above into your MCP Client configuration.")

if __name__ == '__main__':
    generate_token()