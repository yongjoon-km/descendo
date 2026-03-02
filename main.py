from enum import StrEnum
from pathlib import Path
from typing import Annotated
import uuid
from fastapi import FastAPI, HTTPException, Query, UploadFile
from pydantic import BaseModel

app = FastAPI()

# Temporary model storage. In production, we might use S3.
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

# Temporary db. In production, we might use RDB.
models_db = {}

class ModelMetadata(BaseModel):
    name: str
    framework: str

class ModelStatus(StrEnum):
    PENDING_UPLOAD = 'PENDING_UPLOAD'
    UPLOADING = 'UPLOADING'
    UPLOADED = 'UPLOADED'

@app.post("/models")
def create_model(metadata: ModelMetadata):
    model_id = str(uuid.uuid4())

    models_db[model_id] = {
        "id": model_id,
        "name": metadata.name,
        "framework": metadata.framework,
        "status": ModelStatus.PENDING_UPLOAD,
        "file_path": None
    }
    return {
        "message": "Metadata created successfully.",
        "model_id": model_id,
        "upload_url": f"/models/{model_id}/upload",
    }

@app.post("/models/{model_id}/upload")
async def upload_model(model_id: str, file: UploadFile):
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model ID not found. Create metadata first")
    if models_db[model_id]["status"] == ModelStatus.UPLOADING:
        raise HTTPException(status_code=400, detail="A file is uploadeding")
    if models_db[model_id]["status"] == ModelStatus.UPLOADED:
        raise HTTPException(status_code=400, detail="A file has already been uploaded")

    file_extension = Path(file.filename if file.filename else "").suffix
    destination_path = MODEL_DIR / f"{model_id}{file_extension}"
    models_db[model_id]["status"] = ModelStatus.UPLOADING

    try:
        with destination_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                _ = buffer.write(chunk)

        models_db[model_id]["status"] = ModelStatus.UPLOADED
        models_db[model_id]["file_path"] = str(destination_path)

    except Exception as e:
        if destination_path.exists():
            destination_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    finally:
        await file.close()

    return {
        "message": "Model uploaded successfully",
    }

@app.get("/models")
def read_models(
    skip: Annotated[int, Query(description="How many records to skip (offset)", ge=0)] = 0,
    limit: Annotated[int, Query(description="How many records to return (Max 100)", ge=1, le=100)] = 10
):
    all_models = list(models_db.values())

    total_count = len(all_models)

    paginated_models = all_models[skip : skip + limit]

    return {
        "total": total_count,
        "skip": skip,
        "limit": limit,
        "data": paginated_models,
    }

@app.get("/models/{model_id}")
def read_model(model_id: str):
    if model_id not in models_db:
        raise HTTPException(status_code=400, detail=f"Model of {model_id} not found")
    model = models_db[model_id]

    return {
        "id": model_id,
        "name": model["name"],
        "framework": model["framework"],
        "status": model["status"].value,
        "file_path": model["file_path"]
    }

@app.post("/tasks")
def publish_task():
    return {}

@app.get("/tasks/{task_id}")
def read_task():
    return {}

@app.post("/tasks/{task_id}/cancel")
def cancel_task():
    return {}
