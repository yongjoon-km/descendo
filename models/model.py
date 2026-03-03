from enum import StrEnum
from sqlmodel import SQLModel, Field


class ModelStatus(StrEnum):
    PENDING_UPLOAD = 'PENDING_UPLOAD'
    UPLOADING = 'UPLOADING'
    UPLOADED = 'UPLOADED'


class Model(SQLModel, table=True):
    id: int | None = Field(primary_key=True)
    name: str = Field()
    framework: str = Field()
    status: ModelStatus = Field()
    file_path: str | None = Field()
