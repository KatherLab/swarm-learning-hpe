#!/bin/sh
set -eu
# Define script name and directory
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Help function
help()
{
    echo "Usage: $script_name [-h]"
    echo ""
    echo "Options:"
    echo "  -h, --help   Show help message and exit"
    echo ""
    exit 1
}

# Process command line options
while [ $# -gt 0 ]
do
    key="$1"
    case $key in
        -h|--help)
            help
            shift
            ;;
        *)
            echo "Unknown option: $1"
            help
            shift
            ;;
    esac
done

# Check if APLS directory already exists
DIR="./apls-9.12/"
if [ -d "$DIR" ]
then
    echo "APLS is already installed in $DIR"
else
    echo "Downloading APLS installer..."
    gdown --folder https://drive.google.com/drive/folders/166t3FPPtI25CQdNXkYtfX7thmXJcnOuO?usp=share_link  # for folders
    echo "Download complete."
fi

# Change to the UNIX directory and run the setup script
cd "$script_dir/apls-9.12/UNIX"
chmod a+x setup.bin
echo "Starting APLS installation..."
sudo ./setup.bin

echo "APLS installation complete."
# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "APLS installed successfully."