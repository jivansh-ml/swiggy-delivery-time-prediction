
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (app.py, templates/, dvc files, etc.)
COPY . .

# Pull the large model files tracked by DVC from your DagsHub remote.
# DAGSHUB_TOKEN is set as a Secret in the Space settings (never hardcoded here).
ARG DAGSHUB_TOKEN
ENV DAGSHUB_TOKEN=${DAGSHUB_TOKEN}
RUN dvc remote modify origin --local auth basic && \
    dvc remote modify origin --local user jivanshs51 && \
    dvc remote modify origin --local password "${DAGSHUB_TOKEN}" && \
    dvc pull -r origin

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]