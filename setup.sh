#!/bin/bash
set -e

# Activate Conda environment
source /opt/anaconda3/bin/activate multiaid_env

echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

# Install Node.js if not installed
if ! command -v node &> /dev/null
then
    echo "Node.js not found. Installing via Homebrew..."
    brew install node
fi

# Ensure correct ownership of ~/.zshrc
USER_NAME=$(whoami)
if [ -f "$HOME/.zshrc" ]; then
    sudo chown $USER_NAME:staff "$HOME/.zshrc"
fi

# Add Homebrew to PATH in .zshrc if not already added
if ! grep -q '/opt/homebrew/bin' "$HOME/.zshrc"; then
    echo 'export PATH="/opt/homebrew/bin:$PATH"' >> "$HOME/.zshrc"
    echo "Homebrew path added to .zshrc"
fi

# Source .zshrc to apply changes
source "$HOME/.zshrc"

echo "Setup completed successfully!"
echo "Node version: $(node -v)"
echo "NPM version: $(npm -v)"

npm start &

echo "Installing Redis via Homebrew..."
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

uvicorn main:app --reload &

if ! brew list redis &>/dev/null; then
    brew install redis
fi

echo "Starting Redis..."
brew services start redis

echo "Starting Celery..."
python -m celery -A tasks.executor worker --loglevel=info &

echo "Starting HTTP server..."
python -m http.server 8000
