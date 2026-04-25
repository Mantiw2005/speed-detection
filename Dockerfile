FROM ubuntu:22.04

WORKDIR /app

# Install Python and minimal dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120"]