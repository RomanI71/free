#!/bin/bash

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "Starting server with Python..."
    python3 -m http.server 8080
elif command -v python &> /dev/null; then
    echo "Starting server with Python..."
    python -m http.server 8080
else
    echo "Python not found. Please install Python to run this server."
    exit 1
fi