#!/bin/bash

docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml down
docker compose -f docker-compose.gpu.yml run --rm -it --entrypoint bash app
#docker compose -f docker-compose.gpu.yml run --rm -it --entrypoint "bash -c 'python3 gradio_app.py'" app