from fyers_apiv2 import accessToken
import os
import webbrowser

client_id = os.getenv("FYERS_CLIENT_ID")      # XXXXXXXX-100
secret_key = os.getenv("FYERS_SECRET_KEY")
redirect_uri = os.getenv("FYERS_REDIRECT_URI")

session = accessToken.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type="code",
    grant_type="authorization_code"
)

auth_url = session.generate_authcode()

print("OPEN THIS URL IN BROWSER:")
print(auth_url)

webbrowser.open(auth_url)
