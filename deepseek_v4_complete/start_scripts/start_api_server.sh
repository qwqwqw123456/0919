#!/bin/bash
uvicorn backend_server.fastapi_main_server:app --host 0.0.0.0 --port 8000
