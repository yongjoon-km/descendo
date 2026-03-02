import asyncio
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Dict
import uuid
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

app = FastAPI()


# ===
# Model API
# ===

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

# ===
# Task APIs
# ===

tasks_db = {}

class TaskMode(StrEnum):
    SYNC = 'SYNC'
    ASYNC = 'ASYNC'


class TaskState(StrEnum):
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


class TaskRequest(BaseModel):
    mode: TaskMode = Field(default=TaskMode.ASYNC, description="async / sync mode of executing the given task")
    payload: dict[str, Any] = Field(..., description="data required for the given task")
    retry: int = Field(default=3, ge=0, description="retry count if given task failed")
    timeout_sec: int = Field(default=5, gt=0, description="timeout second for waiting the task is completed")

class TaskResponse(BaseModel):
    task_id: str
    status: TaskState
    result: Any | None = None

async def dummy_executor(task_id: str):
    if tasks_db[task_id]["status"] == TaskState.CANCELLED:
        return

    tasks_db[task_id]["status"] = TaskState.RUNNING

    await asyncio.sleep(3)

    if tasks_db[task_id]["status"] == TaskState.CANCELLED:
        return

    tasks_db[task_id]["status"] = TaskState.COMPLETED
    tasks_db[task_id]["result"] = {"message": "Task finished successfully!"}


@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())

    tasks_db[task_id] = {
        "task_id": task_id,
        "status": TaskState.PENDING,
        "payload": request.payload,
        "config": {"retry": request.retry, "timeout_sec": request.timeout_sec},
        "result": None
    }

    if  request.mode == TaskMode.SYNC:
        max_attempts = request.retry + 1
        
        for attempt in range(max_attempts):
            try:
                # We reset the status to RUNNING for each new attempt
                tasks_db[task_id]["status"] = TaskState.RUNNING
                
                # Execute with the timeout applied to *this specific attempt*
                await asyncio.wait_for(dummy_executor(task_id), timeout=request.timeout_sec)
                
                # If it finishes successfully without throwing an error, we break the loop
                if tasks_db[task_id]["status"] == TaskState.COMPLETED:
                    break
                    
            except asyncio.TimeoutError:
                # If it times out, and we are out of retries, mark as FAILED
                if attempt == request.retry:
                    tasks_db[task_id]["status"] = TaskState.FAILED
                    tasks_db[task_id]["result"] = {
                        "error": f"Task timed out after {request.timeout_sec}s on final attempt ({max_attempts}/{max_attempts})."
                    }
                    
            except Exception as e:
                # Catch any other execution errors (like DB disconnects, math errors)
                if attempt == request.retry:
                    tasks_db[task_id]["status"] = TaskState.FAILED
                    tasks_db[task_id]["result"] = {
                        "error": f"Task failed on final attempt: {str(e)}"
                    }
                else:
                    # Optional: Add a small delay between retries so you don't hammer the system
                    await asyncio.sleep(1)

        return tasks_db[task_id]

    else:
        background_tasks.add_task(dummy_executor, task_id)
        return tasks_db[task_id]

@app.get("/tasks/{task_id}", response_model=TaskResponse)
def read_task(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_db[task_id]

@app.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    current_status = tasks_db[task_id]["status"]

    if current_status in [TaskState.COMPLETED, TaskState.FAILED]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel a task in {current_status} state")

    tasks_db[task_id]["status"] = TaskState.CANCELLED
    return {"message": "Task cancellation requested", "task_id": task_id}
