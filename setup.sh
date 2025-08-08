set -e

echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

echo "Updating system packages..."
sudo apt update
sudo apt install redis-server -y
sudo service redis-server start

celery -A tasks.executor worker --loglevel=info
