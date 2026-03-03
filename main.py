import concurrent
import json
import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from sqlalchemy import func
from sqlmodel import SQLModel, Field as SQLField, Session, create_engine, select

# ===
# Database Models
# ===

class ModelStatus(StrEnum):
    PENDING_UPLOAD = 'PENDING_UPLOAD'
    UPLOADING = 'UPLOADING'
    UPLOADED = 'UPLOADED'



class Model(SQLModel, table=True):
    id: int | None = SQLField(primary_key=True)
    name: str = SQLField()
    framework: str = SQLField()
    status: ModelStatus = SQLField()
    file_path: str | None = SQLField()

class TaskMode(StrEnum):
    SYNC = 'SYNC'
    ASYNC = 'ASYNC'


class TaskStatus(StrEnum):
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


class TaskConfig(BaseModel):
    retry: int
    timeout_sec: int


class Task(SQLModel, table=True):
    id: int | None = SQLField(primary_key=True)
    mode: TaskMode = SQLField()
    status: TaskStatus = SQLField()
    model_id: int = SQLField()
    payload: str = SQLField()
    config: str = SQLField()
    result: str | None= SQLField()


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args = connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]


app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()


# ===
# Model API
# ===

# Temporary model storage. In production, we might use S3.
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

class CreateModelRequest(BaseModel):
    name: str
    framework: str

class CreateModelResponse(BaseModel):
    message: str
    model_id: int
    upload_url: str

class ReadModelsResponse(BaseModel):
    total_count: int
    skip: int
    limit: int
    data: list[Model]


@app.post("/models", response_model=CreateModelResponse)
def create_model(metadata: CreateModelRequest, session: SessionDep):
    model = Model(
        id=None,
        name=metadata.name,
        framework=metadata.framework,
        status=ModelStatus.PENDING_UPLOAD,
        file_path=None
    )
    session.add(model)
    session.commit()
    session.refresh(model)

    return {
        "message": "Metadata created successfully.",
        "model_id": model.id,
        "upload_url": f"/models/{model.id}/upload",
    }

@app.post("/models/{model_id}/upload")
async def upload_model(session: SessionDep, model_id: str, file: UploadFile):
    model = session.get(Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model ID not found. Create metadata first")
    if model.status == ModelStatus.UPLOADING:
        raise HTTPException(status_code=400, detail="A file is uploadeding")
    if model.status == ModelStatus.UPLOADED:
        raise HTTPException(status_code=400, detail="A file has already been uploaded")

    file_extension = Path(file.filename if file.filename else "").suffix
    destination_path = MODEL_DIR / f"{model_id}{file_extension}"
    model.status = ModelStatus.UPLOADING
    session.commit()

    try:
        with destination_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                _ = buffer.write(chunk)

        model.status = ModelStatus.UPLOADED
        model.file_path = str(destination_path)
        session.commit()

    except Exception as e:
        if destination_path.exists():
            destination_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    finally:
        await file.close()

    return {
        "message": "Model uploaded successfully",
    }

@app.get("/models", response_model=ReadModelsResponse)
def read_models(
    session: SessionDep,
    skip: Annotated[int, Query(description="How many records to skip (offset)", ge=0)] = 0,
    limit: Annotated[int, Query(description="How many records to return (Max 100)", ge=1, le=100)] = 10
):
    query = select(Model).offset(skip).limit(limit).order_by(Model.id)
    models = session.exec(query).all()
    total_count = session.scalar(select(func.count()).select_from(Model))


    return {
        "total_count": total_count,
        "skip": skip,
        "limit": limit,
        "data": list(models),
    }

@app.get("/models/{model_id}", response_model=Model)
def read_model(session: SessionDep, model_id: str):
    model = session.get(Model, model_id)
    if not model:
        raise HTTPException(status_code=400, detail=f"Model of {model_id} not found")

    return model

# ===
# Task APIs
# ===

class TaskRequest(BaseModel):
    mode: TaskMode = Field(default=TaskMode.ASYNC, description="async / sync mode of executing the given task")
    model_id: int = Field(..., description="model id for running the task")
    payload: dict[str, Any] = Field(..., description="data required for the given task")
    retry: int = Field(default=3, ge=0, description="retry count if given task failed")
    timeout_sec: int = Field(default=5, gt=0, description="timeout second for waiting the task is completed")

class TaskResponse(BaseModel):
    id: int
    status: TaskStatus
    result: Any | None = None


def dummy_executor(session: Session, task_id: int):
    task = session.get(Task, task_id)

    if task is None:
        raise Exception(f"task of {task_id} not found")

    if task.status == TaskStatus.CANCELLED:
        return

    task.status = TaskStatus.RUNNING
    session.commit()

    time.sleep(3)

    task.status = TaskStatus.COMPLETED
    task.result = json.dumps({"message": "Task finished successfully!"})
    session.commit()


@app.post("/tasks", response_model=TaskResponse)
def create_task(session: SessionDep, request: TaskRequest):
    task = Task(
        id=None,
        mode=request.mode,
        status=TaskStatus.PENDING,
        model_id=request.model_id,
        payload=json.dumps(request.payload),
        config=TaskConfig(retry=request.retry, timeout_sec=request.timeout_sec).model_dump_json(),
        result=None
    )
    
    session.add(task)
    session.commit()      # 1. Commit FIRST to save the row and generate the ID
    session.refresh(task) # 2. Refresh to load the new ID into the Python object
    assert task.id is not None
    
    config = TaskConfig.model_validate_json(task.config)

    if task.mode == TaskMode.SYNC:
        max_attempts = config.retry + 1
        
        for attempt in range(max_attempts):
            try:
                # Execute the sync function inside a thread to enforce a timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    # TODO: Change to real executor
                    future = executor.submit(dummy_executor, session, task.id)
                    
                    # This blocks until it finishes OR hits the timeout_sec
                    future.result(timeout=config.timeout_sec)
                    
                # If we reach here without exceptions, it succeeded. Break the loop.
                break 
                
            except concurrent.futures.TimeoutError:
                # If it times out, check if we are out of retries
                if attempt == config.retry:
                    task.status = TaskStatus.FAILED
                    task.result = json.dumps({
                        "error": f"Task timed out after {config.timeout_sec}s on final attempt ({max_attempts}/{max_attempts})."
                    })
                    session.commit()
                    
            except Exception as e:
                # Catch any other execution errors (like DB disconnects, math errors)
                if attempt == config.retry:
                    task.status = TaskStatus.FAILED
                    task.result = json.dumps({
                        "error": f"Task failed on final attempt: {str(e)}"
                    })
                    session.commit()
                else:
                    # Synchronous sleep before retrying
                    time.sleep(1)

        return task

    else:
        return task

@app.get("/tasks/{task_id}", response_model=TaskResponse)
def read_task(session: SessionDep, task_id: int):
    task = session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/tasks/{task_id}/cancel")
def cancel_task(session: SessionDep, task_id: int):
    task = session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    current_status = task.status

    if current_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel a task in {current_status} state")

    task.status = TaskStatus.CANCELLED
    session.commit()
    return {"message": "Task cancellation requested", "task_id": task_id}
