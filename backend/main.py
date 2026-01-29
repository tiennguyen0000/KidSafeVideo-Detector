"""FastAPI entrypoint."""

import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# === PATH / BASE DIR ===
APP_BASE_DIR = os.environ.get("APP_BASE_DIR", "/opt/airflow")
for p in (APP_BASE_DIR, "/app"):
    if p not in sys.path:
        sys.path.insert(0, p)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import router sau khi đã chỉnh sys.path
from api.router import router  # noqa: E402

# === Tạo app ===
app = FastAPI(
    title="Video Classifier API",
    description="Multimodal harmful video classification API",
    version="1.0.0"
)

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên specify domains cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Include API router ===
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    # chạy: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
