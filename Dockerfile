# Phase 6 deliverable — kept minimal for now so the image builds once the app exists.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit UI arrives in increment 2.
EXPOSE 8501
CMD ["python", "-c", "print('conversation-memory: build the app first (increment 2)')"]
