#!/bin/sh

# Function to display help information
show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Stop swarm services based on the specified option."
    echo ""
    echo "Options:"
    echo "  --sn     Stop the SN (Swarm Network) service."
    echo "  --swop   Stop the SWOP (Swarm Worker OPerator) service."
    echo "  --swci   Stop the SWCI (Swarm CI) service."
    echo "  --sl   Stop the SL (SL node) service."
    echo "  --clean   Clean up staled containers"
    echo "  --all    Stop all swarm services."
    echo ""
    echo "Example: $0 --sn"
    exit 1
}

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Error: An option is required."
    show_help
fi

# Directory where the stop-swarm script is located
script_dir="./workspace/swarm_learning_scripts"

# Process the provided option
case "$1" in
    --sn)
        $script_dir/stop-swarm --sn
        ;;
    --swop)
        $script_dir/stop-swarm --swop
        ;;
    --swci)
        $script_dir/stop-swarm --swci
        ;;
    --sl)
        $script_dir/stop-swarm --sl
        ;;
    --clean)
        docker rm $(docker ps --filter status=exited -q)
        ;;
    --all)
        $script_dir/stop-swarm
        ;;
    *)
        echo "Invalid option: $1"
        show_help
        ;;
esac
