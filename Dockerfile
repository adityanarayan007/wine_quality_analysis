FROM python:3.10-slim-buster

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY models/ /app/models/

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "src.api.app:app"]