# 🚀 OCR Ultra Profesional API

Sistema híbrido inteligente de OCR con fallback a IA para máxima precisión y eficiencia.

## 🎯 Características Principales

### 🏆 Nivel 1: OCR Ultra Pro (100% GRATIS)
- **Tesseract**: 6 configuraciones optimizadas para diferentes tipos de documentos
- **EasyOCR**: 3 niveles de confianza para máxima cobertura
- **Mejoras de Imagen IA**:
  - Corrección automática de inclinación
  - Reducción de ruido multi-nivel
  - Mejora de contraste adaptativo (CLAHE)
  - Binarización inteligente (Otsu + Adaptativo)
  - Limpieza de artefactos

### 🤖 Nivel 2: Gemini Fallback (Solo cuando OCR falla)
- Activado automáticamente cuando la calidad del OCR es insuficiente
- Usa Gemini 1.5 Flash para máxima precisión
- **Ahorro inteligente**: Solo paga cuando es absolutamente necesario

## 📊 Ventajas del Sistema

| Característica | Valor |
|---------------|-------|
| 🆓 Procesamiento gratuito | 70-90% de documentos |
| ⚡ Velocidad promedio | 2-5 segundos |
| 🎯 Precisión | 85-95% |
| 💰 Ahorro estimado | 80% vs solo Gemini |
| 📄 Formatos soportados | 14+ tipos de archivo |

## 🛠️ Instalación y Despliegue

### Requisitos Previos
- Docker y Docker Compose
- Google Cloud CLI (para despliegue en Cloud Run)
- API Key de Google Gemini (opcional, para fallback)

### Configuración Local

1. **Clonar repositorio**
```bash
git clone <tu-repositorio>
cd OCR_IO
```

2. **Configurar variables de entorno**
```bash
cp .env.example .env
# Editar .env y añadir tu GEMINI_API_KEY (opcional)
```

3. **Construir y ejecutar con Docker**
```bash
docker build -t ocr-ultra-pro .
docker run -p 8080:8080 -e GEMINI_API_KEY=tu_api_key ocr-ultra-pro
```

4. **O ejecutar localmente sin Docker**
```bash
# Instalar dependencias del sistema (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng poppler-utils

# Instalar dependencias Python
pip install -r requirements.txt

# Ejecutar
export GEMINI_API_KEY=tu_api_key  # Opcional
python app.py
```

### Despliegue en Google Cloud Run

```bash
# Configurar proyecto
gcloud config set project TU_PROJECT_ID

# Construir y subir imagen
gcloud builds submit --tag gcr.io/TU_PROJECT_ID/ocr-ultra-pro

# Desplegar
gcloud run deploy ocr-ultra-pro \
  --image gcr.io/TU_PROJECT_ID/ocr-ultra-pro \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars GEMINI_API_KEY=tu_api_key
```

## 📡 API Reference

### 1. Información del Servicio
```bash
GET /
```

**Respuesta:**
```json
{
  "service": "OCR Ultra Profesional API",
  "version": "3.0.0",
  "status": "active",
  "features": {
    "ocr_ultra_pro": {
      "tesseract_configs": 6,
      "easyocr_levels": 3,
      "cost": "FREE"
    },
    "gemini_fallback": {
      "available": true,
      "cost": "Variable"
    }
  }
}
```

### 2. Procesar Documento
```bash
POST /api/process
Content-Type: multipart/form-data
```

**Parámetros:**
- `file`: Archivo a procesar (requerido)
- `use_gemini_fallback`: `true` o `false` (default: `true`)

**Ejemplo con curl:**
```bash
curl -X POST http://localhost:8080/api/process \
  -F "file=@documento.jpg" \
  -F "use_gemini_fallback=true"
```

**Respuesta Exitosa (OCR Ultra Pro):**
```json
{
  "success": true,
  "filename": "documento.jpg",
  "file_type": "image",
  "total_processing_time": 3.2,
  "result": {
    "success": true,
    "method": "ocr_ultra_pro",
    "engine": "tesseract",
    "text": "Texto extraído del documento...",
    "confidence": 87.5,
    "quality_score": 82.3,
    "structured_data": {
      "emails": ["contacto@ejemplo.com"],
      "phones": ["3001234567"],
      "dates": ["15/02/2026"]
    },
    "processing_time": 2.8,
    "cost": 0.0
  },
  "cost_info": {
    "method_used": "OCR Ultra Pro (GRATIS)",
    "cost": "$0.00",
    "savings": "100% - No se usó Gemini API"
  }
}
```

**Respuesta con Gemini Fallback:**
```json
{
  "success": true,
  "result": {
    "method": "gemini_fallback",
    "engine": "gemini",
    "text": "Texto extraído con Gemini...",
    "confidence": 85.0,
    "fallback_reason": "Calidad insuficiente (score: 32.5)",
    "cost": "variable"
  },
  "cost_info": {
    "method_used": "Gemini Fallback",
    "reason": "OCR insuficiente"
  }
}
```

### 3. Procesar a Markdown (con MarkItDown)
```bash
POST /api/process-markdown
Content-Type: multipart/form-data
```

**Parámetros:**
- `file`: Archivo a procesar

### 4. Health Check
```bash
GET /health
```

## 🎨 Formatos Soportados

### Imágenes
- JPG, JPEG, PNG, BMP, TIFF, TIF, GIF, WEBP

### Documentos
- **PDF**: Extracción directa + OCR para escaneados
- **Word**: DOC, DOCX
- **PowerPoint**: PPT, PPTX

## 📈 Análisis de Calidad

El sistema evalúa automáticamente la calidad del OCR con múltiples criterios:

1. ✅ **Longitud del texto**: Mínimo 3 palabras
2. ✅ **Confianza**: Mínimo 35%
3. ✅ **Detección de corrupción**: Símbolos raros, mayúsculas excesivas
4. ✅ **Coherencia lingüística**: Palabras comunes en español
5. ✅ **Datos estructurados**: Presencia de emails, teléfonos, fechas, etc.

**Score de Calidad**: 0-100
- 🟢 50+: Excelente (OCR suficiente)
- 🟡 35-49: Aceptable (OCR con texto largo)
- 🔴 <35: Insuficiente (activa Gemini fallback)

## 💡 Ejemplos de Uso

### Python
```python
import requests

url = "http://localhost:8080/api/process"
files = {'file': open('documento.jpg', 'rb')}
data = {'use_gemini_fallback': 'true'}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Método usado: {result['result']['method']}")
print(f"Texto: {result['result']['text']}")
print(f"Confianza: {result['result']['confidence']}%")
```

### JavaScript (Node.js)
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('documento.jpg'));
form.append('use_gemini_fallback', 'true');

axios.post('http://localhost:8080/api/process', form, {
  headers: form.getHeaders()
})
.then(response => {
  console.log('Método:', response.data.result.method);
  console.log('Texto:', response.data.result.text);
})
.catch(error => console.error(error));
```

## 🔧 Optimización de Costos

### Estrategias para Maximizar Ahorro

1. **Mejora la calidad de escaneo**
   - Usa resolución mínima de 300 DPI
   - Asegura buena iluminación
   - Evita sombras y reflejos

2. **Preprocesa imágenes**
   - Endereza documentos torcidos
   - Aumenta contraste
   - Recorta bordes innecesarios

3. **Usa el parámetro `use_gemini_fallback=false`**
   - Para pruebas o documentos de baja prioridad
   - Cuando el costo es crítico

### Monitoreo de Costos

Para PDFs, revisa las estadísticas en la respuesta:

```json
{
  "statistics": {
    "ocr_success": 18,
    "gemini_usage": 2,
    "cost_free_percentage": 90.0
  },
  "cost_info": {
    "pages_processed_free": 18,
    "pages_with_gemini": 2,
    "estimated_savings": "18 llamadas a Gemini evitadas"
  }
}
```

## 🐛 Solución de Problemas

### Error: "Gemini no disponible"
- Verifica que `GEMINI_API_KEY` esté configurada correctamente
- Asegúrate de tener acceso a la API de Gemini
- Revisa los límites de tu cuenta

### OCR devuelve texto vacío
- Verifica la calidad de la imagen (mínimo 300 DPI)
- Asegura que el texto sea legible
- Intenta con `use_gemini_fallback=true`

### Timeout en PDFs grandes
- El sistema procesa máximo 20 páginas por defecto
- Para PDFs más grandes, considera dividirlos
- Aumenta el timeout en Cloud Run si es necesario

## 📊 Benchmarks

### Rendimiento Promedio

| Tipo | Tiempo | Precisión | Costo |
|------|--------|-----------|-------|
| Imagen simple | 2-3s | 90-95% | $0.00 |
| Imagen compleja | 4-6s | 85-90% | $0.00* |
| PDF 10 páginas | 15-30s | 88-93% | $0.00-0.02 |

*Puede usar Gemini fallback en 10-20% de casos

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para detalles.

## 👥 Autor

Luis Carlos Gómez

## 🙏 Agradecimientos

- Tesseract OCR
- EasyOCR
- Google Gemini API
- OpenCV
- PyMuPDF

---

**⚡ Construido con Python, Flask y mucho ☕**
