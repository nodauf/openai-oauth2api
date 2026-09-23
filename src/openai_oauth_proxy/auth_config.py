from __future__ import annotations

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
SCOPE = "openid profile email offline_access"
AUTHORIZE_EXTRA_PARAMS = {
    "codex_cli_simplified_flow": "true",
    "id_token_add_organizations": "true",
    "originator": "codex_cli_rs",
}
