# Demo image: API server + batch worker in one container (cloud with internet access, CPU only).
#
# Build and run from the project root:
#     docker compose up -d --build
# or
#     docker build -t mrx .
#     docker run -d -p 8000:8000 -v mrx-data:/data --init --stop-timeout 120 --restart unless-stopped mrx

FROM python:3.13-slim

# OpenCV (installed with ultralytics) needs these system libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# CPU-only PyTorch: far smaller than the default CUDA build, and the demo machine has no GPU.
# Installed first so that the packages below keep this build instead of pulling the CUDA one.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Dependencies before the code, so code changes do not reinstall them.
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt \
    # Fail the build here, not at the first job, if something the API or the worker needs is missing.
    && python -c "import fastapi, uvicorn, sqlalchemy, ultralytics, cv2, torch; print('torch', torch.__version__)"

# Demo model weights (YOLO11n, matches inference.yolo.DEFAULT_NECK_LAYERS).
# Files downloaded with ADD are readable only by root by default, hence --chmod.
ADD --chmod=644 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt /models/yolo11n.pt

# Demo images: the sample images bundled with the ultralytics package (bus.jpg, zidane.jpg).
RUN mkdir -p /demo_images \
    && python -c "import pathlib, shutil, ultralytics; \
[shutil.copy(p, '/demo_images') for p in (pathlib.Path(ultralytics.__file__).parent / 'assets').glob('*.jpg')]" \
    && ls /demo_images/*.jpg

COPY models/ /models/
ADD --chmod=644 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt /models/yolo11/yolo11m.pt
COPY dataset/coco128/images/val /demo_images
COPY src/ src/

# Swagger UI files. app/main.py serves them itself instead of loading them from a CDN.
ADD --chmod=644 https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js src/app/static/swagger/
ADD --chmod=644 https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css src/app/static/swagger/

COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Run as a non-root user. /data holds the SQLite DB and the result files (mount a volume there).
RUN useradd --uid 1000 --create-home app && mkdir -p /data && chown app:app /data
USER app

ENV DB_PATH=/data/local.db \
    BATCH_OUTPUT_DIR=/data/batch_outputs \
    INFERENCE_DEVICE=cpu \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/openapi.json')"

CMD ["entrypoint.sh"]