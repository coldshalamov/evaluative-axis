# Gemini Rerun Blocked: API Key Permission

Date: June 22, 2026

The provided Google API key was stored in `.env.local`, and `.gitignore`
excludes `.env`, `.env.*`, and `.tmp/`.

The key loads locally, but Gemini API requests are blocked. Sanitized checks:

- `models:list` on `generativelanguage.googleapis.com`: HTTP 403
- `generateContent` on `gemini-2.0-flash`: HTTP 403
- `embedContent` on `text-embedding-004`: HTTP 403

The error status was `PERMISSION_DENIED`, with Google reporting that requests
to the Generative Language API methods are blocked.

This means the key is present but is not allowed to call the Gemini API. The
fix is one of:

1. Create/import an API key from Google AI Studio for the Gemini API.
2. In Google Cloud Console, enable the Generative Language API for the key's
   project.
3. If the key has API restrictions, allow the Generative Language API.
4. If the key has application restrictions, use a server/local-compatible
   restriction for this run or remove application restrictions temporarily.

The rerun script is ready, but it cannot start Phase 1/2 until the key can call
`generativelanguage.googleapis.com`.
