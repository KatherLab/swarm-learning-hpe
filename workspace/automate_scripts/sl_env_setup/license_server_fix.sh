#!/bin/sh
set -eux

# Help function
help() {
  echo "Usage: ./license_server_fix.sh"
  echo "Options:"
  echo "  -h, --help     Show help"
  exit 1
}

# Process command options
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) help ;;
    *) echo "Unknown option: $1"; help ;;
  esac
done

# Update server.xml to use port 5000
sudo sed -i "s+5814+5000+g" "/opt/HP/HP AutoPass License Server/HP AutoPass License Server/HP AutoPass License Server/conf/server.xml"

# Create startup script for hpLicenseServer
sudo cp "/opt/HP/HP AutoPass License Server/HP AutoPass License Server/HP AutoPass License Server/bin/hpLicenseServer" "/etc/init.d/hpLicenseServer"
sudo chmod 755 "/etc/init.d/hpLicenseServer"

# Configure hpLicenseServer to start at boot
sudo update-rc.d "hpLicenseServer" defaults 97 03

# Start hpLicenseServer and check status
sudo service "hpLicenseServer" start
sudo service "hpLicenseServer" status
