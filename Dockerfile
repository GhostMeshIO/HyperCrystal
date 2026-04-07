FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for scikit-learn and gudhi
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose dashboard/API port
EXPOSE 5000

# Default command: run the dashboard
CMD ["python", "hypercrystal/dashboard/hypercrystal_dash.py"]
