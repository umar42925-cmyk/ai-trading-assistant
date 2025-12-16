from fyers_apiv3 import fyersModel

# ===== EDIT THESE FOUR LINES ONLY =====
APP_ID = "0K4RH3LJYJ-100"          # same as Step 3
SECRET_KEY = "L7CY5MBUL9"  # from FYERS dashboard
REDIRECT_URI = "http://127.0.0.1:5000/redirect"
AUTH_CODE = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBfaWQiOiIwSzRSSDNMSllKIiwidXVpZCI6ImE1NTZhYzM0ODk2ZjQ1YjRiM2IwNzE3ODYyMTg2YmQ5IiwiaXBBZGRyIjoiIiwibm9uY2UiOiIiLCJzY29wZSI6IiIsImRpc3BsYXlfbmFtZSI6IlhVMDExOTQiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI3N2RlNmM2YzRjZTc2ZWZiYzNlY2ZhNmQ1ODc3N2Q0ZDU0YTkzMmJjMmU2NDgxOGUxOWRhYzIxMyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImF1ZCI6IltcImQ6MVwiLFwiZDoyXCIsXCJ4OjBcIixcIng6MVwiLFwieDoyXCJdIiwiZXhwIjoxNzY1ODI3OTkzLCJpYXQiOjE3NjU3OTc5OTMsImlzcyI6ImFwaS5sb2dpbi5meWVycy5pbiIsIm5iZiI6MTc2NTc5Nzk5Mywic3ViIjoiYXV0aF9jb2RlIn0.l56zXSJQjx1FQSP0diacnjJhcb5o5F_jj01hwmaWV3M"
# ====================================

session = fyersModel.SessionModel(
    client_id=APP_ID,
    secret_key=SECRET_KEY,
    redirect_uri=REDIRECT_URI,
    response_type="code",
    grant_type="authorization_code"
)

session.set_token(AUTH_CODE)
response = session.generate_token()

print("\n=== FYERS TOKEN RESPONSE ===")
print(response)
