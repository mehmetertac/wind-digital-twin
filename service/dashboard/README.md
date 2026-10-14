# Twin dashboard

Streamlit UI for the Week 11 deployment — reads the FastAPI service only.

## Run locally

```powershell
pip install -r service/dashboard/requirements.txt
pip install -e ".[serve]"
python service/scripts/export_models.py
$env:TWIN_MODELS_DIR = "service/models"
uvicorn twin_service.api.app:app --port 8000
$env:API_URL = "http://127.0.0.1:8000"
streamlit run service/dashboard/app.py
```

## Compose

`docker compose -f service/docker-compose.yml up --build` — API **:8000**, dashboard **:8501**.
