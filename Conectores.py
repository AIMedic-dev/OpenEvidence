from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
import logging
from datetime import datetime
import traceback
from typing import Any
import secrets
from AgenteOpenEvidence import MedicalRAGSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de la aplicación Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)  # Genera clave secreta aleatoria
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

CORS(app, origins="*", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'csv', 'xlsx', 'xls'}

# Crear directorio de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inicializar el sistema RAG médico globalmente
try:
    medical_rag = MedicalRAGSystem()
    logger.info("✅ Sistema RAG médico inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error inicializando sistema RAG médico: {e}")
    medical_rag = None

def allowed_file(filename):
    """Verificar si el tipo de archivo está permitido"""
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_response(success: bool, message: str, data: Any = None, status_code: int = 200) -> tuple:
    """Crear respuesta estandarizada"""
    response = {
        'success': success,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    return jsonify(response), status_code

def handle_error(error: Exception, message: str = "Error interno del servidor") -> tuple:
    """Manejar errores de forma estandarizada"""
    logger.error(f"{message}: {str(error)}\n{traceback.format_exc()}")
    return create_response(
        success=False,
        message=f"{message}: {str(error)}",
        status_code=500
    )

# ========== ENDPOINTS DE SALUD Y ESTADO ==========

@app.route('/health', methods=['GET'])
def health_check():
    """Verificar el estado de salud de la API"""
    try:
        system_status = {
            'api_status': 'running',
            'rag_system': 'initialized' if medical_rag else 'error',
            'database_documents': len(medical_rag.documents_metadata) if medical_rag else 0,
            'faiss_index_size': medical_rag.index.ntotal if medical_rag and medical_rag.index else 0
        }
        
        return create_response(
            success=True,
            message="Sistema funcionando correctamente",
            data=system_status
        )
    except Exception as e:
        return handle_error(e, "Error verificando estado de salud")

@app.route('/status', methods=['GET'])
def get_status():
    """Obtener información detallada del sistema"""
    try:
        if not medical_rag:
            return create_response(
                success=False,
                message="Sistema RAG médico no disponible",
                status_code=503
            )
        
        # Estadísticas de fuentes
        sources = {}
        web_sources = 0
        local_sources = 0
        
        for doc in medical_rag.documents_metadata:
            source = doc['source']
            if source.startswith('WEB:'):  # Corregido: usar 'WEB:' no 'MED_WEB:'
                web_sources += 1
            else:
                local_sources += 1
                
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        status_info = {
            'total_documents': len(medical_rag.documents_metadata),
            'faiss_index_size': medical_rag.index.ntotal if medical_rag.index else 0,
            'local_sources': local_sources,
            'web_sources': web_sources,  # Corregido: cambiar nombre
            'sources_breakdown': sources,
            'storage_path': medical_rag.storage_path,
            'embedding_deployment': medical_rag.embedding_deployment,
            'chunk_size': medical_rag.chunk_size,
            'chunk_overlap': medical_rag.chunk_overlap
        }
        
        return create_response(
            success=True,
            message="Estado del sistema obtenido correctamente",
            data=status_info
        )
        
    except Exception as e:
        return handle_error(e, "Error obteniendo estado del sistema")

# ========== ENDPOINTS DE CONSULTAS MÉDICAS ==========

@app.route('/ask', methods=['POST'])
def ask_medical_question():
    """Realizar consulta médica al sistema RAG"""
    try:
        if not medical_rag:
            return create_response(
                success=False,
                message="Sistema RAG médico no disponible",
                status_code=503
            )
        
        data = request.get_json()
        
        if not data or 'question' not in data:
            return create_response(
                success=False,
                message="Se requiere el campo 'question' en el JSON",
                status_code=400
            )
        
        question = data['question'].strip()
        max_docs = data.get('max_docs', 10)
        
        if not question:
            return create_response(
                success=False,
                message="La pregunta no puede estar vacía",
                status_code=400
            )
        
        # Realizar consulta médica
        logger.info(f"🏥 Procesando consulta médica: {question[:100]}...")
        result = medical_rag.ask_medical_question(question, max_docs)
        
        # Preparar respuesta
        response_data = {
            'answer': result['answer'],
            'query': result['query'],
            'confidence': result['confidence'],
            'evidence_level': result['evidence_level'],
            'context_length': result['context_length'],
            'sources_summary': {
                'total_sources': len(result['sources']),
                'local_sources': result['local_sources'],
                'web_sources': result['web_sources']  # Corregido: cambiar nombre
            },
            'sources': [
                {
                    'source': doc['source'],
                    'section': doc.get('section', ''),
                    'similarity_score': doc['similarity_score'],
                    'tokens': doc.get('tokens', 0),
                    'doc_id': doc.get('doc_id', ''),
                    'chunk_index': doc.get('chunk_index', 0),
                    'excerpt': doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text']
                }
                for doc in result['sources']
            ]
        }
        
        return create_response(
            success=True,
            message=f"Consulta médica procesada correctamente (confianza: {result['confidence']:.2f})",
            data=response_data
        )
        
    except Exception as e:
        return handle_error(e, "Error procesando consulta médica")

@app.route('/search', methods=['POST'])
def search_documents():
    """Buscar documentos relevantes sin generar respuesta"""
    try:
        if not medical_rag:
            return create_response(
                success=False,
                message="Sistema RAG médico no disponible",
                status_code=503
            )
        
        data = request.get_json()
        
        if not data or 'query' not in data:
            return create_response(
                success=False,
                message="Se requiere el campo 'query' en el JSON",
                status_code=400
            )
        
        query = data['query'].strip()
        k = data.get('k', 10)
        
        if not query:
            return create_response(
                success=False,
                message="La consulta no puede estar vacía",
                status_code=400
            )
        
        # Realizar búsqueda
        logger.info(f"🔍 Buscando documentos para: {query[:100]}...")
        results = medical_rag.search_documents(query, k)
        
        # Preparar respuesta
        response_data = {
            'query': query,
            'total_results': len(results),
            'results': [
                {
                    'source': doc['source'],
                    'section': doc.get('section', ''),
                    'similarity_score': doc['similarity_score'],
                    'tokens': doc.get('tokens', 0),
                    'chunk_index': doc.get('chunk_index', 0),
                    'doc_id': doc.get('doc_id', ''),
                    'excerpt': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                }
                for doc in results
            ]
        }
        
        return create_response(
            success=True,
            message=f"Búsqueda completada: {len(results)} documentos encontrados",
            data=response_data
        )
        
    except Exception as e:
        return handle_error(e, "Error en búsqueda de documentos")

@app.route('/enhanced-search', methods=['POST'])
def enhanced_medical_search():
    """Realizar búsqueda mejorada con fuentes web cuando sea necesario"""
    try:
        if not medical_rag:
            return create_response(
                success=False,
                message="Sistema RAG médico no disponible",
                status_code=503
            )
        
        data = request.get_json()
        
        if not data or 'query' not in data:
            return create_response(
                success=False,
                message="Se requiere el campo 'query' en el JSON",
                status_code=400
            )
        
        query = data['query'].strip()
        k = data.get('k', 10)
        
        if not query:
            return create_response(
                success=False,
                message="La consulta no puede estar vacía",
                status_code=400
            )
        
        # Realizar búsqueda mejorada
        logger.info(f"🔍 Búsqueda mejorada para: {query[:100]}...")
        results = medical_rag.enhanced_medical_search(query, k)
        
        # Preparar respuesta
        response_data = {
            'query': query,
            'total_results': len(results),
            'results': [
                {
                    'source': doc['source'],
                    'section': doc.get('section', ''),
                    'similarity_score': doc['similarity_score'],
                    'tokens': doc.get('tokens', 0),
                    'chunk_index': doc.get('chunk_index', 0),
                    'doc_id': doc.get('doc_id', ''),
                    'is_web_source': doc['source'].startswith('WEB:'),
                    'excerpt': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                }
                for doc in results
            ]
        }
        
        return create_response(
            success=True,
            message=f"Búsqueda mejorada completada: {len(results)} documentos encontrados",
            data=response_data
        )
        
    except Exception as e:
        return handle_error(e, "Error en búsqueda mejorada")

@app.route('/web-search', methods=['POST'])
def medical_web_search():
    """Realizar búsqueda en fuentes médicas web"""
    try:
        if not medical_rag:
            return create_response(
                success=False,
                message="Sistema RAG médico no disponible",
                status_code=503
            )
        
        data = request.get_json()
        
        if not data or 'query' not in data:
            return create_response(
                success=False,
                message="Se requiere el campo 'query' en el JSON",
                status_code=400
            )
        
        query = data['query'].strip()
        num_results = data.get('num_results', 5)
        
        if not query:
            return create_response(
                success=False,
                message="La consulta no puede estar vacía",
                status_code=400
            )
        
        # Realizar búsqueda médica web
        logger.info(f"🌐 Búsqueda médica web para: {query[:100]}...")
        results = medical_rag.medical_search(query, num_results)
        
        # Preparar respuesta
        response_data = {
            'query': query,
            'total_results': len(results),
            'results': [
                {
                    'title': result['title'],
                    'url': result['link'],
                    'snippet': result['snippet'],
                    'source_type': result['source_type'],
                    'content_length': len(result['content'])
                }
                for result in results
            ]
        }
        
        return create_response(
            success=True,
            message=f"Búsqueda médica web completada: {len(results)} fuentes encontradas",
            data=response_data
        )
        
    except Exception as e:
        return handle_error(e, "Error en búsqueda médica web")

@app.route('/validate-source', methods=['POST'])
def validate_medical_source():
    """Validar fuente médica"""
    try:
        if not medical_rag:
            return create_response(
                success=False,
                message="Sistema RAG médico no disponible",
                status_code=503
            )
        
        data = request.get_json()
        
        if not data or 'source' not in data:
            return create_response(
                success=False,
                message="Se requiere el campo 'source' en el JSON",
                status_code=400
            )
        
        source = data['source'].strip()
        
        if not source:
            return create_response(
                success=False,
                message="La fuente no puede estar vacía",
                status_code=400
            )
        
        # Validar fuente
        validation_result = medical_rag.validate_medical_source(source)
        
        return create_response(
            success=True,
            message="Validación de fuente completada",
            data=validation_result
        )
        
    except Exception as e:
        return handle_error(e, "Error validando fuente")

# ========== ENDPOINTS DE GESTIÓN DE DOCUMENTOS ==========

@app.route('/upload', methods=['POST'])
def upload_file():
    """Subir y procesar archivo médico"""
    try:
        if not medical_rag:
            return create_response(
                success=False,
                message="Sistema RAG médico no disponible",
                status_code=503
            )
        
        if 'file' not in request.files:
            return create_response(
                success=False,
                message="No se encontró archivo en la solicitud",
                status_code=400
            )
        
        file = request.files['file']
        
        if file.filename == '':
            return create_response(
                success=False,
                message="No se seleccionó ningún archivo",
                status_code=400
            )
        
        if not allowed_file(file.filename):
            return create_response(
                success=False,
                message=f"Tipo de archivo no permitido. Extensiones permitidas: {ALLOWED_EXTENSIONS}",
                status_code=400
            )
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        file.save(file_path)
        
        try:
            # Procesar archivo
            logger.info(f"📁 Procesando archivo: {filename}")
            success = medical_rag.add_file(file_path, unique_filename)
            
            if success:
                return create_response(
                    success=True,
                    message=f"Archivo '{filename}' procesado y agregado exitosamente",
                    data={
                        'filename': filename,
                        'processed_filename': unique_filename,
                        'file_size': os.path.getsize(file_path),
                        'total_documents': len(medical_rag.documents_metadata)
                    }
                )
            else:
                return create_response(
                    success=False,
                    message=f"No se pudo procesar el archivo '{filename}'. Verifique el contenido del archivo.",
                    status_code=400
                )
        
        finally:
            # Limpiar archivo temporal
            if os.path.exists(file_path):
                os.remove(file_path)
        
    except RequestEntityTooLarge:
        return create_response(
            success=False,
            message="Archivo demasiado grande. Tamaño máximo permitido: 100MB",
            status_code=413
        )
    except Exception as e:
        return handle_error(e, "Error procesando archivo")

@app.route('/add-text', methods=['POST'])
def add_text_document():
    """Agregar documento de texto directamente"""
    try:
        if not medical_rag:
            return create_response(
                success=False,
                message="Sistema RAG médico no disponible",
                status_code=503
            )
        
        data = request.get_json()
        
        if not data or 'text' not in data or 'source' not in data:
            return create_response(
                success=False,
                message="Se requieren los campos 'text' y 'source' en el JSON",
                status_code=400
            )
        
        text = data['text'].strip()
        source = data['source'].strip()
        
        if not text or not source:
            return create_response(
                success=False,
                message="Los campos 'text' y 'source' no pueden estar vacíos",
                status_code=400
            )
        
        # Agregar documento
        logger.info(f"📄 Agregando documento de texto: {source}")
        success = medical_rag.add_document(text, source)
        
        if success:
            return create_response(
                success=True,
                message=f"Documento '{source}' agregado exitosamente",
                data={
                    'source': source,
                    'text_length': len(text),
                    'total_documents': len(medical_rag.documents_metadata)
                }
            )
        else:
            return create_response(
                success=False,
                message=f"No se pudo agregar el documento '{source}'",
                status_code=400
            )
        
    except Exception as e:
        return handle_error(e, "Error agregando documento de texto")

@app.route('/documents', methods=['GET'])
def list_documents():
    """Listar todos los documentos en la base de datos"""
    try:
        if not medical_rag:
            return create_response(
                success=False,
                message="Sistema RAG médico no disponible",
                status_code=503
            )
        
        # Agrupar por fuente
        sources = {}
        web_sources_count = 0
        local_sources_count = 0
        
        for doc in medical_rag.documents_metadata:
            source = doc['source']
            
            # Contadores por tipo de fuente
            if source.startswith('WEB:'):
                web_sources_count += 1
            else:
                local_sources_count += 1
            
            if source not in sources:
                sources[source] = {
                    'source': source,
                    'chunk_count': 0,
                    'total_tokens': 0,
                    'sections': set(),
                    'is_web_source': source.startswith('WEB:'),
                    'doc_ids': []
                }
            
            sources[source]['chunk_count'] += 1
            sources[source]['total_tokens'] += doc.get('tokens', 0)
            if doc.get('section'):
                sources[source]['sections'].add(doc.get('section'))
            if doc.get('doc_id'):
                sources[source]['doc_ids'].append(doc.get('doc_id'))
        
        # Convertir sets a listas y limpiar doc_ids duplicados
        for source_info in sources.values():
            source_info['sections'] = list(source_info['sections'])
            source_info['doc_ids'] = list(set(source_info['doc_ids']))
        
        response_data = {
            'total_sources': len(sources),
            'total_chunks': len(medical_rag.documents_metadata),
            'local_sources_count': local_sources_count,
            'web_sources_count': web_sources_count,
            'sources': list(sources.values())
        }
        
        return create_response(
            success=True,
            message=f"Lista de documentos obtenida: {len(sources)} fuentes",
            data=response_data
        )
        
    except Exception as e:
        return handle_error(e, "Error listando documentos")

@app.route('/documents/<path:source_name>', methods=['DELETE'])
def delete_document(source_name):
    """Eliminar documento específico por nombre de fuente"""
    try:
        if not medical_rag:
            return create_response(
                success=False,
                message="Sistema RAG médico no disponible",
                status_code=503
            )
        
        # Verificar si el documento existe
        source_exists = any(doc['source'] == source_name for doc in medical_rag.documents_metadata)
        
        if not source_exists:
            return create_response(
                success=False,
                message=f"No se encontró documento con fuente '{source_name}'",
                status_code=404
            )
        
        # Eliminar documento
        logger.info(f"🗑️ Eliminando documento: {source_name}")
        success = medical_rag.delete_medical_documents([source_name])
        
        if success:
            return create_response(
                success=True,
                message=f"Documento '{source_name}' eliminado exitosamente",
                data={
                    'deleted_source': source_name,
                    'remaining_documents': len(medical_rag.documents_metadata)
                }
            )
        else:
            return create_response(
                success=False,
                message=f"Error eliminando documento '{source_name}'",
                status_code=500
            )
        
    except Exception as e:
        return handle_error(e, f"Error eliminando documento '{source_name}'")

@app.route('/documents', methods=['DELETE'])
def clear_all_documents():
    """Limpiar toda la base de datos médica"""
    try:
        if not medical_rag:
            return create_response(
                success=False,
                message="Sistema RAG médico no disponible",
                status_code=503
            )
        
        # Obtener confirmación del query parameter
        confirm = request.args.get('confirm', '').lower()
        
        if confirm != 'true':
            return create_response(
                success=False,
                message="Se requiere parámetro 'confirm=true' para limpiar toda la base de datos",
                status_code=400
            )
        
        # Limpiar base de datos
        logger.info("🗑️ Limpiando toda la base de datos médica")
        success = medical_rag.clear_medical_database()
        
        if success:
            return create_response(
                success=True,
                message="Base de datos médica limpiada exitosamente",
                data={'total_documents': 0}
            )
        else:
            return create_response(
                success=False,
                message="Error limpiando la base de datos médica",
                status_code=500
            )
        
    except Exception as e:
        return handle_error(e, "Error limpiando base de datos médica")

# ========== MANEJO DE ERRORES ==========

@app.errorhandler(404)
def not_found(error):
    return create_response(
        success=False,
        message="Endpoint no encontrado",
        status_code=404
    )

@app.errorhandler(405)
def method_not_allowed(error):
    return create_response(
        success=False,
        message="Método HTTP no permitido para este endpoint",
        status_code=405
    )

@app.errorhandler(413)
def request_entity_too_large(error):
    return create_response(
        success=False,
        message="Archivo demasiado grande. Tamaño máximo permitido: 100MB",
        status_code=413
    )

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Error interno del servidor: {error}")
    return create_response(
        success=False,
        message="Error interno del servidor",
        status_code=500
    )

# ========== ENDPOINT DE INFORMACIÓN DE LA API ==========

@app.route('/', methods=['GET'])
def home():
    """Página principal con interfaz web (si existe el template)"""
    try:
        return render_template('InterfazAgente.html')
    except:
        return create_response(
            success=True,
            message="Medical RAG System API - Sistema funcionando correctamente",
            data={
                'api_version': '2.0',
                'system_name': 'Medical RAG System',
                'endpoints': '/api-info'
            }
        )[0]  # Solo devolver el JSON, no la tupla

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Iniciando Medical RAG System API en {host}:{port}")
    logger.info(f"🏥 Debug mode: {debug}")
    logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"📄 Supported file types: {ALLOWED_EXTENSIONS}")
    
    app.run(host=host, port=port, debug=debug)