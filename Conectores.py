from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from AgenteOpenEvidence import ImprovedRAGSystem

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

rag_system = ImprovedRAGSystem()

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    with open('InterfazAgente.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/upload_files', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'})

        files = request.files.getlist('files')
        uploaded_count = 0

        for file in files:
            if file.filename == '':
                continue

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
                    file.save(temp_file.name)
                    
                    if rag_system.add_file(temp_file.name, filename):
                        uploaded_count += 1
                    
                    os.unlink(temp_file.name)

        return jsonify({
            'success': True,
            'uploaded': uploaded_count,
            'total': len(files)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_documents', methods=['GET'])
def get_documents():
    try:
        documents = rag_system.get_documents_list()
        return jsonify(documents)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete_documents', methods=['POST'])
def delete_documents():
    try:
        data = request.get_json()
        sources = data.get('sources', [])
        
        if rag_system.delete_documents(sources):
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Error deleting documents'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear_database', methods=['POST'])
def clear_database():
    try:
        if rag_system.clear_database():
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Error clearing database'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '')
        max_docs = data.get('max_docs', 8)
        
        if not question:
            return jsonify({'success': False, 'error': 'Question is required'})

        result = rag_system.ask_enhanced(question, max_docs=max_docs)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/system_status', methods=['GET'])
def system_status():
    try:
        azure_connection = True
        google_search = bool(rag_system.google_api_key and rag_system.google_cse_id)
        
        try:
            rag_system.get_azure_embedding("test")
        except:
            azure_connection = False
        
        return jsonify({
            'azure_connection': azure_connection,
            'google_search': google_search,
            'embedding_model': rag_system.embedding_deployment
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)