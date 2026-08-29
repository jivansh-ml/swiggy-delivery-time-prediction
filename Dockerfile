cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# LightGBM needs this system library to load at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 app:app
EOF
