"""
Super OCR API - Flask Application for Google Cloud Run
Soporta: Imágenes (JPG, PNG, etc), PDFs, Word, PowerPoint
"""

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import pytesseract
import easyocr
from PIL import Image
import numpy as np
import json
from datetime import datetime
import tempfile
import mimetypes

# Importar procesadores de documentos
try:
    import fitz  # PyMuPDF para PDFs
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document  # python-docx para Word
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from pptx import Presentation  # python-pptx para PowerPoint
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

# Configuración
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB máximo

# Configurar Tesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Inicializar EasyOCR globalmente
print("🚀 Inicializando EasyOCR...")
easyocr_reader = easyocr.Reader(['es', 'en'], gpu=False)
print("✅ EasyOCR listo")


class DocumentProcessor:
    """Procesador inteligente de documentos"""
    
    ALLOWED_EXTENSIONS = {
        'image': {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif', 'webp'},
        'pdf': {'pdf'},
        'word': {'docx', 'doc'},
        'powerpoint': {'pptx', 'ppt'}
    }
    
    @classmethod
    def detect_file_type(cls, filename):
        """Detectar tipo de archivo"""
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        for file_type, extensions in cls.ALLOWED_EXTENSIONS.items():
            if ext in extensions:
                return file_type, ext
        
        return None, ext
    
    @classmethod
    def is_allowed_file(cls, filename):
        """Verificar si el archivo es permitido"""
        file_type, ext = cls.detect_file_type(filename)
        return file_type is not None


class OCRProcessor:
    """Motor OCR con Tesseract y EasyOCR"""
    
    def __init__(self):
        self.easyocr_reader = easyocr_reader
        
    def enhance_image(self, image_path):
        """Mejorar calidad de imagen"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        enhanced = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return enhanced
    
    def process_with_tesseract(self, image):
        """OCR con Tesseract"""
        try:
            text = pytesseract.image_to_string(
                image,
                lang='spa+eng',
                config='--psm 6'
            ).strip()
            
            data = pytesseract.image_to_data(
                image, 
                lang='spa+eng', 
                output_type=pytesseract.Output.DICT
            )
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': text,
                'confidence': round(avg_confidence, 1),
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def process_with_easyocr(self, image_path):
        """OCR con EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(image_path)
            
            valid_results = [item for item in results if item[2] > 0.3]
            text = ' '.join([item[1] for item in valid_results])
            avg_confidence = sum([item[2] for item in valid_results]) / len(valid_results) if valid_results else 0
            
            return {
                'text': text,
                'confidence': round(avg_confidence * 100, 1),
                'word_count': len(text.split()),
                'char_count': len(text),
                'detections': len(valid_results)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def process_image(self, image_path):
        """Procesar imagen con ambos engines"""
        enhanced = self.enhance_image(image_path)
        
        results = {
            'tesseract': self.process_with_tesseract(enhanced),
            'easyocr': self.process_with_easyocr(image_path)
        }
        
        best = self._select_best(results)
        
        return {
            'engines': results,
            'best_result': best,
            'file_type': 'image'
        }
    
    def _select_best(self, engines):
        """Seleccionar mejor resultado"""
        best = {'engine': None, 'text': '', 'confidence': 0, 'score': 0}
        
        for engine, result in engines.items():
            if 'error' in result or not result.get('text', '').strip():
                continue
            
            text = result['text'].strip()
            confidence = result.get('confidence', 0)
            char_count = len(text)
            
            score = (confidence * 0.4) + min(char_count / 100, 10) * 6
            
            if score > best['score']:
                best = {
                    'engine': engine,
                    'text': text,
                    'confidence': confidence,
                    'score': round(score, 1)
                }
        
        return best


class PDFProcessor:
    """Procesador de archivos PDF"""
    
    def __init__(self, ocr_processor):
        self.ocr = ocr_processor
    
    def process(self, pdf_path):
        """Procesar PDF completo"""
        if not PDF_AVAILABLE:
            return {'error': 'PyMuPDF no disponible'}
        
        try:
            pdf_doc = fitz.open(pdf_path)
            pages = []
            total_text = ''
            
            for page_num in range(min(pdf_doc.page_count, 10)):
                page = pdf_doc[page_num]
                
                direct_text = page.get_text().strip()
                
                if direct_text and len(direct_text) > 30:
                    page_result = {
                        'page': page_num + 1,
                        'text': direct_text,
                        'method': 'direct',
                        'char_count': len(direct_text)
                    }
                else:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        pix.save(tmp.name)
                        ocr_result = self.ocr.process_image(tmp.name)
                        os.unlink(tmp.name)
                    
                    best = ocr_result.get('best_result', {})
                    page_result = {
                        'page': page_num + 1,
                        'text': best.get('text', ''),
                        'method': 'ocr',
                        'engine': best.get('engine', 'none'),
                        'confidence': best.get('confidence', 0)
                    }
                
                pages.append(page_result)
                total_text += page_result['text'] + '\n\n'
            
            pdf_doc.close()
            
            return {
                'file_type': 'pdf',
                'total_pages': len(pages),
                'pages_processed': len(pages),
                'pages': pages,
                'total_text': total_text.strip(),
                'char_count': len(total_text)
            }
            
        except Exception as e:
            return {'error': str(e)}


class WordProcessor:
    """Procesador de archivos Word"""
    
    def process(self, docx_path):
        """Procesar documento Word"""
        if not DOCX_AVAILABLE:
            return {'error': 'python-docx no disponible'}
        
        try:
            doc = Document(docx_path)
            
            full_text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)
            
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            full_text.append(cell.text)
            
            text = '\n'.join(full_text)
            
            return {
                'file_type': 'word',
                'text': text,
                'paragraphs': len(doc.paragraphs),
                'tables': len(doc.tables),
                'char_count': len(text),
                'word_count': len(text.split())
            }
            
        except Exception as e:
            return {'error': str(e)}


class PowerPointProcessor:
    """Procesador de archivos PowerPoint"""
    
    def process(self, pptx_path):
        """Procesar presentación PowerPoint"""
        if not PPTX_AVAILABLE:
            return {'error': 'python-pptx no disponible'}
        
        try:
            prs = Presentation(pptx_path)
            slides = []
            total_text = ''
            
            for slide_num, slide in enumerate(prs.slides, 1):
                slide_text = []
                
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text)
                
                text = '\n'.join(slide_text)
                slides.append({
                    'slide': slide_num,
                    'text': text,
                    'shapes_count': len(slide.shapes)
                })
                total_text += text + '\n\n'
            
            return {
                'file_type': 'powerpoint',
                'total_slides': len(slides),
                'slides': slides,
                'total_text': total_text.strip(),
                'char_count': len(total_text)
            }
            
        except Exception as e:
            return {'error': str(e)}


# Crear procesadores globales
ocr_processor = OCRProcessor()
pdf_processor = PDFProcessor(ocr_processor)
word_processor = WordProcessor()
ppt_processor = PowerPointProcessor()


@app.route('/', methods=['GET'])
def home():
    """Endpoint de información"""
    return jsonify({
        'service': 'Super OCR API',
        'version': '1.0.0',
        'status': 'active',
        'endpoints': {
            'process': {
                'url': '/api/process',
                'method': 'POST',
                'description': 'Procesar documento (imagen, PDF, Word, PowerPoint)'
            },
            'health': {
                'url': '/health',
                'method': 'GET',
                'description': 'Verificar estado del servicio'
            }
        },
        'supported_formats': {
            'images': list(DocumentProcessor.ALLOWED_EXTENSIONS['image']),
            'pdf': list(DocumentProcessor.ALLOWED_EXTENSIONS['pdf']),
            'word': list(DocumentProcessor.ALLOWED_EXTENSIONS['word']),
            'powerpoint': list(DocumentProcessor.ALLOWED_EXTENSIONS['powerpoint'])
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check para Cloud Run"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'engines': {
            'tesseract': True,
            'easyocr': easyocr_reader is not None,
            'pdf': PDF_AVAILABLE,
            'word': DOCX_AVAILABLE,
            'powerpoint': PPTX_AVAILABLE
        }
    }), 200


@app.route('/api/process', methods=['POST'])
def process_document():
    """Procesar cualquier documento"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró archivo en la solicitud'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400
        
        if not DocumentProcessor.is_allowed_file(file.filename):
            return jsonify({
                'error': 'Formato no soportado',
                'allowed_formats': {
                    'images': list(DocumentProcessor.ALLOWED_EXTENSIONS['image']),
                    'documents': list(DocumentProcessor.ALLOWED_EXTENSIONS['pdf'] | 
                                    DocumentProcessor.ALLOWED_EXTENSIONS['word'] | 
                                    DocumentProcessor.ALLOWED_EXTENSIONS['powerpoint'])
                }
            }), 400
        
        file_type, ext = DocumentProcessor.detect_file_type(file.filename)
        
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            if file_type == 'image':
                result = ocr_processor.process_image(temp_path)
            elif file_type == 'pdf':
                result = pdf_processor.process(temp_path)
            elif file_type == 'word':
                result = word_processor.process(temp_path)
            elif file_type == 'powerpoint':
                result = ppt_processor.process(temp_path)
            else:
                result = {'error': 'Tipo de archivo no reconocido'}
            
            os.unlink(temp_path)
            
            response = {
                'success': True,
                'filename': filename,
                'file_type': file_type,
                'extension': ext,
                'timestamp': datetime.now().isoformat(),
                'result': result
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)