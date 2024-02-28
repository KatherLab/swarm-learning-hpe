#!/bin/sh
host_index=""

# Help function
help() {
  echo ""
  echo "Generate alias for swarm learning scripts"
  echo "-d    The host index, choose from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich] for your site"
  echo "-h    Display this help message."
  exit 1
}

# Process command options
while getopts "d:h" opt; do
  case "$opt" in
    h ) help ;;
    d ) host_index="$OPTARG" ;;
    * ) help ;;
  esac
done

# Check required options are set
if [ -z "$host_index" ]; then
  echo "Error: missing host index"
  help
fi

# Path to your alias script
alias_script="/opt/hpe/swarm-learning-hpe/workspace/automate_scripts/sl_env_setup/add_alias.sh -d $host_index"

# Determine which shell the user is using and set the config file path accordingly
config_file=""
case "$SHELL" in
  */bash) config_file="$HOME/.bashrc" ;;
  */zsh) config_file="$HOME/.zshrc" ;;
  *)
    echo "Unsupported shell. This script supports bash and zsh only."
    exit 1
    ;;
esac

# Check if the alias script is already sourced in the config file
if grep -q "$alias_script" "$config_file"; then
  echo "Alias script already sourced in $config_file"
else
  # If not, append the source command to the config file
  echo "Sourcing alias script in $config_file..."
  echo "source $alias_script" >> "$config_file"
  echo "Alias script sourced successfully. Please restart your terminal or source the $config_file file."
fi
