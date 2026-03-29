"""
server/app.py — FastAPI server.
"""

from openenv.core.env_server import create_fastapi_app
from server.environment import TrafficEnvironment
from models import TrafficAction, TrafficObservation

app = create_fastapi_app(TrafficEnvironment, TrafficAction, TrafficObservation)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()