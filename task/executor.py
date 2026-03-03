import json
from sqlmodel import Session
import torch

from task.benchmark import benchmark_model
from models.model import Model
from models.task import Task, TaskPayload, TaskStatus


def execute_task(session: Session, task_id: int):
    task = session.get(Task, task_id)

    if not task:
        raise Exception("task not found")

    if task.status == TaskStatus.CANCELLED:
        return

    model_id = task.model_id

    model = session.get(Model, model_id)
    if not model:
        raise Exception("model not found")

    file_path = model.file_path
    if not file_path:
        raise Exception("model is not uploaded yet")

    model = torch.export.load(file_path)

    task_payload = TaskPayload.model_validate_json(task.payload)
    task.status = TaskStatus.RUNNING
    session.commit()

    benchmark = benchmark_model(
        model.module(),
        task_payload.input_shape,
        task_payload.num_runs,
        task_payload.warmup_runs
    )

    task.status = TaskStatus.COMPLETED
    task.result = json.dumps({"message": "Task finished successfully!", "result": benchmark})
    session.commit()

