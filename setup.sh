set -e

echo "Installing Python packages from requirements.txt..."
pip3 install --user -r requirements.txt

echo "Updating system packages..."
sudo apt update
sudo apt install redis-server -y
sudo service redis-server start

# Fix PATH for user-installed executables
export PATH=$PATH:~/.local/bin

# Start Celery via python module (more reliable)
python3 -m celery -A tasks.executor worker --loglevel=info &

# Start server
python3 -m http.server 8000