#!/bin/sh

# Function to display help information
show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Fetch and follow the latest logs for a specified service."
    echo ""
    echo "Options:"
    echo "  --ml     Fetch logs for the latest 'user-env' container."
    echo "  --sn     Fetch logs for the latest container based on the 'jeffzhu69/swarm-learning:sn' image."
    echo "  --swop   Fetch logs for the latest container based on the 'jeffzhu69/swarm-learning:swop' image."
    echo "  --swci   Fetch logs for the latest container based on the 'jeffzhu69/swarm-learning:swci' image."
    echo "  --sl     Fetch logs for the latest container based on the 'jeffzhu69/sl:swop' image."
    echo ""
    echo "Example: $0 --ml"
    exit 1
}

# Ensure an option is provided
if [ $# -eq 0 ]; then
    echo "Error: An option is required."
    show_help
fi

# Helper function to get the latest container ID based on image name
get_container_id_by_image() {
    docker ps -a --filter "ancestor=$1" --format "{{.ID}}" | head -n 1
}

# Process the provided option
case "$1" in
    --ml)
        container_id=$(docker ps -a --filter "name=us*" --format "{{.ID}}" | head -n 1)
        ;;
    --sn)
        container_id=$(get_container_id_by_image "jeffzhu69/swarm-learning:sn")
        ;;
    --swop)
        container_id=$(get_container_id_by_image "jeffzhu69/swarm-learning:swop")
        ;;
    --swci)
        container_id=$(get_container_id_by_image "jeffzhu69/swarm-learning:swci")
        ;;
    --sl)
        container_id=$(get_container_id_by_image "jeffzhu69/sl:swop")
        ;;
    *)
        echo "Invalid option: $1"
        show_help
        ;;
esac

# Fetch and follow logs if container ID is found
if [ -n "$container_id" ]; then
    echo "Fetching logs for container $container_id..."
    docker logs "$container_id" -f
else
    echo "No matching container found."
fi
