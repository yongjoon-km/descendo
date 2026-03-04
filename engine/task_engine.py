import signal
import threading
from sqlmodel import Session, create_engine, select

from models.task import Task, TaskMode, TaskStatus
from task.executor import execute_task


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args = connect_args)

shutdown_event = threading.Event()


def handle_shutdown_signal(signum, frame):
    print("shutdown signal received")

    shutdown_event.set()


signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)


def main():

    print("Engine loop started.")
    while not shutdown_event.is_set():
        with Session(engine) as session:
            query = select(Task).where(Task.status == TaskStatus.PENDING, Task.mode == TaskMode.ASYNC).order_by(Task.id)
            task = session.exec(query).first()
            if not task:
                continue

            execute_task(session, task)

        shutdown_event.wait(timeout=5.0)

    print("Engine loop exited")

if __name__ == "__main__":
    main()
