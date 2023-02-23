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
sed -i "s+5814+5000+g" "/opt/HP/HP AutoPass License Server/HP AutoPass License Server/HP AutoPass License Server/conf/server.xml"

# Create startup script for hpLicenseServer
cp "/opt/HP/HP AutoPass License Server/HP AutoPass License Server/HP AutoPass License Server/bin/hpLicenseServer" "/etc/init.d/hpLicenseServer"
chmod 755 "/etc/init.d/hpLicenseServer"

# Configure hpLicenseServer to start at boot
update-rc.d "hpLicenseServer" defaults 97 03

# Start hpLicenseServer and check status
service "hpLicenseServer" start
service "hpLicenseServer" status
