# 🚀  OCR API

API REST para extracción de texto de documentos usando OCR (Tesseract + EasyOCR).

## 📋 Características

- ✅ **Imágenes**: JPG, PNG, BMP, TIFF, GIF, WEBP
- ✅ **PDFs**: Con texto nativo o escaneados
- ✅ **Word**: DOCX, DOC
- ✅ **PowerPoint**: PPTX, PPT
- ✅ **Multilenguaje**: Español e Inglés
- ✅ **Dual Engine**: Tesseract + EasyOCR para mejor precisión

## 🔧 Tecnologías

- Flask (API REST)
- Tesseract OCR
- EasyOCR
- OpenCV
- PyMuPDF (PDFs)
- python-docx (Word)
- python-pptx (PowerPoint)

## 🌐 Endpoints

### GET `/`
Información del servicio y formatos soportados

### GET `/health`
Verificar estado del servicio

### POST `/api/process`
Procesar documento y extraer texto

**Parámetros:**
- `file`: Archivo a procesar (multipart/form-data)

**Ejemplo con curl:**
```bash
curl -X POST -F "file=@documento.pdf" https://tu-servicio.run.app/api/process
```

**Ejemplo con Python:**
```python
import requests

url = "https://tu-servicio.run.app/api/process"
files = {'file': open('documento.pdf', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## 📦 Respuesta

```json
{
  "success": true,
  "filename": "documento.pdf",
  "file_type": "pdf",
  "extension": "pdf",
  "timestamp": "2025-09-29T10:30:00",
  "result": {
    "total_text": "Texto extraído...",
    "char_count": 1234,
    "confidence": 85.5
  }
}
```

## 🚀 Despliegue en Google Cloud Run

1. Conectar repositorio a Cloud Run
2. Configurar build automático desde GitHub
3. Desplegar

## 📝 Licencia

MIT