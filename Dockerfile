# Usar imagen base de Python
FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema para OCR Ultra Pro
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app.py .

# Exponer puerto
EXPOSE 8080

# Variables de entorno
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Comando para ejecutar la aplicación con más workers y timeout mayor
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 app:app