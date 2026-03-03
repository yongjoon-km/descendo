from pathlib import Path

import pytest
import torch
from sqlalchemy import pool
from sqlmodel import Session, SQLModel, create_engine
from torch import nn

from models.model import Model, ModelStatus
from models.task import Task, TaskMode, TaskStatus
from task.executor import execute_task

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=pool.StaticPool,
)

@pytest.fixture(name="session")
def session():
    SQLModel.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        yield session
    SQLModel.metadata.drop_all(test_engine)


class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


def test_execute_task(session: Session):
    model = SimpleTestModel()

    input_shape = [1, 10]
    sample_input = torch.randn(input_shape)

    save_dir = Path("task/tests")
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / "test.pt2"

    exported_program = torch.export.export(model, (sample_input,))
    torch.export.save(exported_program, file_path)

    assert file_path.exists()

    model = Model(
        id = 1,
        name = "test",
        framework = "test",
        status = ModelStatus.UPLOADED,
        file_path = str(file_path)
    )


    task = Task(
        id = 1,
        mode = TaskMode.ASYNC,
        status = TaskStatus.PENDING,
        model_id = 1,
        payload = '{"input_shape": [1, 10], "num_runs": 10, "warmup_runs": 1}',
        config = '{"retry": 1, "timeout_sec": 10}',
        result = None
    )

    session.add_all([model, task])

    execute_task(session, 1)

    updated_task = session.get(Task, 1)
    assert updated_task is not None
    assert updated_task.status == TaskStatus.COMPLETED

    print(updated_task.result)

