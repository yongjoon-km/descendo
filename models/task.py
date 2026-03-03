from enum import StrEnum
from pydantic import BaseModel
from sqlmodel import SQLModel, Field


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


class TaskPayload(BaseModel):
    input_shape: list[int]
    num_runs: int
    warmup_runs: int


class Task(SQLModel, table=True):
    id: int | None = Field(primary_key=True)
    mode: TaskMode = Field()
    status: TaskStatus = Field()
    model_id: int = Field()
    payload: str = Field()
    config: str = Field()
    result: str | None= Field()
