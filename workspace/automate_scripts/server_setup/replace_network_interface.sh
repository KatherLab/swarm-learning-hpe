#!/bin/bash

# Help function to display usage information
usage() {
    echo "Usage: $0 [--tailscale | --local | --goodaccess]"
    echo "Options:"
    echo "  --tailscale   Replace 'eno1' or 'tun0' with 'tailscale0' in all .sh files"
    echo "  --local       Replace 'tailscale0' or 'tun0' with 'eno1' in all .sh files"
    echo "  --goodaccess  Replace 'tailscale0' or 'eno1' with 'tun0' in all .sh files"
    exit 1
}

# Ensure the script is run with exactly one argument
if [ "$#" -ne 1 ]; then
    usage
fi

# Define the base directory
BASE_DIR="/opt/hpe/swarm-learning-hpe"

# Process the command line argument
case "$1" in
    --tailscale)
        SEARCH_PATTERN="\(eno1\|tun0\)"
        REPLACE_WITH="tailscale0"
        ;;
    --local)
        SEARCH_PATTERN="\(tailscale0\|tun0\)"
        REPLACE_WITH="eno1"
        ;;
    --goodaccess)
        SEARCH_PATTERN="\(tailscale0\|eno1\)"
        REPLACE_WITH="tun0"
        ;;
    *)
        # If the argument does not match any of the expected flags, display usage information
        usage
        ;;
esac

# Use find to locate all .sh files under the specified directory recursively
# and use sed to replace the search pattern with the desired string
find "$BASE_DIR" -type f -name "*.sh" -exec sed -i "s/$SEARCH_PATTERN/$REPLACE_WITH/g" {} +

echo "Replacement complete."
