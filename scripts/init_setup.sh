#!/bin/bash
# HyperCrystal Setup Script
# Creates virtual environment, installs dependencies, and validates installation.
# >> chmod +x scripts/init_setup.sh

set -e  # exit on error

echo "========================================="
echo "HyperCrystal v2.0 Setup"
echo "========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi
echo "Python $PYTHON_VERSION found."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping."
fi

# Install the package in editable mode (if setup.py or pyproject.toml exists)
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing hypercrystal package in editable mode..."
    pip install -e .
else
    echo "No setup.py or pyproject.toml found. Skipping editable install."
fi

# Create necessary directories
echo "Creating directories: checkpoints, logs, data..."
mkdir -p checkpoints logs data

# Check for default config and users files
if [ ! -f "hypercrystal_config.json" ]; then
    echo "Creating default hypercrystal_config.json from template..."
    cat > hypercrystal_config.json << 'EOF'
{
  "seed": 42,
  "embedding_dim": 128,
  "memory_capacity": 10000,
  "verbose": true,
  "checkpoint_dir": "checkpoints"
}
EOF
fi

if [ ! -f "users.json" ]; then
    echo "Creating default users.json with admin/guest..."
    cat > users.json << 'EOF'
{
  "admin": {
    "password": "admin",
    "role": "admin",
    "credits": 1000
  },
  "guest": {
    "password": "guest",
    "role": "guest",
    "credits": 100
  }
}
EOF
fi

echo "========================================="
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the dashboard:"
echo "  python hypercrystal/dashboard/hypercrystal_dash.py"
echo ""
echo "To run the CLI:"
echo "  python run.py --help"
echo "========================================="
