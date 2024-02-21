#!/bin/bash

# Path to your alias script
alias_script="/opt/hpe/swarm-learning-hpe/workspace/automate_scripts/sl_env_setup/add_alias.sh"

# Determine which shell the user is using and set the config file path accordingly
if [[ $SHELL == */bash ]]; then
    config_file="$HOME/.bashrc"
elif [[ $SHELL == */zsh ]]; then
    config_file="$HOME/.zshrc"
else
    echo "Unsupported shell. This script supports bash and zsh only."
    exit 1
fi

# Check if the alias script is already sourced in the config file
if grep -q "$alias_script" "$config_file"; then
    echo "Alias script already sourced in $config_file"
else
    # If not, append the source command to the config file
    echo "Sourcing alias script in $config_file..."
    echo "source $alias_script" >> "$config_file"
    echo "Alias script sourced successfully. Please restart your terminal or source the $config_file file."
fi
