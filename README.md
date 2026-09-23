# OpenAI OAuth Proxy

This project provides a Python server with zero external dependencies that:

1. Performs an OAuth Authorization Code + PKCE flow,
2. Stores user credentials locally (SQLite),
3. Exposes OpenAI-compatible endpoints,
4. Forwards requests to a Codex model via the `responses` API.


## Important

OAuth parameters are hardcoded in the codebase to match the official application:

- `CLIENT_ID = app_EMoamEEZ73f0CkXaXp7hrann`
- `AUTHORIZE_URL = https://auth.openai.com/oauth/authorize`
- `TOKEN_URL = https://auth.openai.com/oauth/token`
- `REDIRECT_URI = http://localhost:1455/auth/callback`
- `SCOPE = openid profile email offline_access`

The `client_secret` is not used (public PKCE flow).

## Getting Started

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
openai-oauth-proxy
```

By default, the proxy listens on `http://127.0.0.1:8000`. You can customize the host and port using `--host` and `--port`:

```bash
openai-oauth-proxy --host 0.0.0.0 --port 8000
```

Then open `http://127.0.0.1:8000/login` in your browser, log in, and use the returned session ID as a Bearer token on `/v1/*` endpoints.

### Usage on Remote Machines or Headless Environments (no GUI)

Because the OAuth callback redirect URI is strictly `http://localhost:1455/auth/callback`, the browser will always redirect to the local machine (`localhost`). If the proxy runs on a remote server, two methods are available:

#### Method 1: Session Export / Import (Recommended)

Log in on your local machine with a browser, then transfer the session to the remote server:

```bash
# On your local machine
openai-oauth-proxy export --output creds.json
scp creds.json user@remote-machine:/tmp/

# On the remote server
ssh user@remote-machine 'openai-oauth-proxy import --input /tmp/creds.json'
```

The exported `session_id` remains identical after import, so your Bearer token does not change.

#### Method 2: SSH Port Forwarding

You can forward ports from your local machine to the remote server:

```bash
ssh -L 8000:localhost:8000 -L 1455:localhost:1455 user@remote-machine
```

You can then open `http://localhost:8000/login` in your local browser: the redirect to `localhost:1455` will be forwarded to the remote proxy callback handler.

## Environment Variables (Optional)

All settings provide sensible defaults that work out-of-the-box without extra configuration. You can optionally set:

- `HOST`: Proxy server listen host (default: `127.0.0.1`).
- `PORT`: Proxy server listen port (default: `8000`).
- `DATA_DIR`: Directory where SQLite database is stored (default: `./data`).
- `OPENAI_API_BASE`: Upstream Codex API URL (default: `https://chatgpt.com/backend-api/codex`).
- `OPENAI_DEFAULT_MODEL`: Default model to use (default: `gpt-5.3-codex`).
- `OPENAI_UPSTREAM_AUTH_MODE`: Upstream authentication mode (`oauth_token` by default, or `api_key`).
- `OPENAI_API_KEY`: Upstream API key if using `OPENAI_UPSTREAM_AUTH_MODE=api_key`.
- `OPENAI_ORGANIZATION`: Optional OpenAI organization identifier.
- `OPENAI_PROJECT`: Optional OpenAI project identifier.

## Endpoints

- `GET /login`: Initiates OAuth authorization flow
- `GET /healthz`: Healthcheck endpoint
- `GET /v1/models`: Lists available models
- `POST /v1/responses`: Direct proxy to the Codex Responses API
- `POST /v1/chat/completions`: Seamless translation from Chat Completions to Codex Responses

`/v1/responses` is forwarded to the upstream Codex backend without transformation.  
`/v1/chat/completions` is converted into a `responses` payload, and its response is reconstructed into the standard chat completion format.  
Both endpoints support `"stream": true`. The `chat/completions` stream is translated into SSE `chat.completion.chunk` events ending with `data: [DONE]`.

The local `"verbosity": true` parameter is stripped before forwarding upstream. On `/v1/chat/completions`, it returns the full `responses` object instead of the reduced chat completion format. With `"stream": true`, the complete upstream `responses` SSE events are relayed.

Example request:

```bash
curl -N -H 'Authorization: Bearer <session_id>' \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.3-codex","stream":true,"messages":[{"role":"user","content":"Hello"}]}' \
  http://127.0.0.1:8000/v1/chat/completions
```

## Limitations

- Local persistence uses SQLite (`data/proxy.sqlite3`).
- The OAuth callback port is strictly fixed to port `1455` by the OpenAI application configuration.
