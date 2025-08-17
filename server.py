# server.py
import os, tempfile, mimetypes
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, AnyUrl
import httpx

from decoder import read_single_barcode  # <- your robust function

API_KEY = os.getenv("API_KEY", "")  # optional shared secret

app = FastAPI()

class Payload(BaseModel):
    url: AnyUrl

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/decode")
async def decode(payload: Payload, request: Request):
    # Optional simple auth
    if API_KEY:
        hdr = request.headers.get("x-api-key")
        if hdr != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    url = str(payload.url)

    # Download to temp file (keep extension to help pillow/opencv)
    suffix = os.path.splitext(url.split("?")[0])[1] or ".jpg"
    # Fallback if URL has no extension but sends a content-type
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, follow_redirects=True)
        r.raise_for_status()
        if suffix == ".jpg":
            ct = r.headers.get("content-type", "")
            ext = mimetypes.guess_extension(ct.split(";")[0].strip()) or ".jpg"
            if ext:
                suffix = ext
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name

    try:
        text = read_single_barcode(tmp_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Decode failed: {e}")
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass
