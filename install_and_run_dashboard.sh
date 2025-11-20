#!/bin/bash
# install_and_run_dashboard.sh

set -e

echo "Installing GNOSIS Web Dashboard dependencies..."

# Install Python packages required for the Flask dashboard
pip install -q flask flask-cors flask-socketio python-socketio alpaca-py

# Ensure templates directory exists (holds dashboard.html)
mkdir -p templates

# Run the dashboard
echo "Starting GNOSIS Web Dashboard..."
python gnosis_dashboard.py
