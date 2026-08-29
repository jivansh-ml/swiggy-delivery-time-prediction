
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project 
COPY . .

# Pull the large model files tracked by DVC from  DagsHub remote.
ARG DAGSHUB_TOKEN
ENV DAGSHUB_TOKEN=${DAGSHUB_TOKEN}
RUN dvc remote modify origin --local auth basic && \
    dvc remote modify origin --local user jivanshs51 && \
    dvc remote modify origin --local password "${DAGSHUB_TOKEN}" && \
    dvc pull -r origin

EXPOSE 8080

CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 app:app