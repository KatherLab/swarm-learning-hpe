#!/bin/bash

# Get the container ID for the latest user-env container
container_id=$(docker ps -a --filter "name=us*" --format "{{.ID}}" | head -n 1)

# Get the latest log for the user-env container
docker logs $container_id -f
