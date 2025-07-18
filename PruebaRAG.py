import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Librerías para embeddings y vectores
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Para procesamiento de documentos
import PyPDF2
import docx
from bs4 import BeautifulSoup
import requests

# Para el chunking inteligente
import re
import tiktoken

class DocumentProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", storage_path: str = "knowledge_base"):
        """
        Inicializa el procesador de documentos con embeddings
        
        Args:
            model_name: Modelo de embeddings a usar
            storage_path: Carpeta donde guardar la base de datos
        """
        self.model_name = model_name
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Cargar modelo de embeddings
        print(f"Cargando modelo de embeddings: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Inicializar base vectorial FAISS
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product para cosine similarity
        
        # Metadatos de documentos
        self.documents = []
        self.document_chunks = []
        
        # Cargar datos existentes si existen
        self._load_existing_data()
        
        # Tokenizer para contar tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _load_existing_data(self):
        """Carga datos existentes desde disco"""
        index_path = self.storage_path / "faiss_index.bin"
        metadata_path = self.storage_path / "metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            print("Cargando base de datos existente...")
            self.index = faiss.read_index(str(index_path))
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data.get('documents', [])
                self.document_chunks = data.get('document_chunks', [])
    
    def _save_data(self):
        """Guarda la base de datos en disco"""
        index_path = self.storage_path / "faiss_index.bin"
        metadata_path = self.storage_path / "metadata.json"
        
        faiss.write_index(self.index, str(index_path))
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'document_chunks': self.document_chunks
            }, f, ensure_ascii=False, indent=2)
    
    def _count_tokens(self, text: str) -> int:
        """Cuenta tokens en un texto"""
        return len(self.tokenizer.encode(text))
    
    def _chunk_text(self, text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
        """
        Divide el texto en chunks semánticamente coherentes
        
        Args:
            text: Texto a dividir
            max_tokens: Máximo tokens por chunk
            overlap: Tokens de solapamiento entre chunks
        """
        # Primero dividir por párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Si el párrafo solo ya es muy largo, dividirlo por oraciones
            if self._count_tokens(paragraph) > max_tokens:
                sentences = re.split(r'[.!?]+', paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    test_chunk = current_chunk + "\n" + sentence if current_chunk else sentence
                    
                    if self._count_tokens(test_chunk) <= max_tokens:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
            else:
                test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                
                if self._count_tokens(test_chunk) <= max_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = paragraph
        
        # Agregar el último chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _extract_metadata(self, text: str, source: str) -> Dict[str, Any]:
        """Extrae metadatos básicos del documento"""
        metadata = {
            'source': source,
            'length': len(text),
            'tokens': self._count_tokens(text),
            'processed_at': datetime.now().isoformat(),
            'doc_id': hashlib.md5(text.encode()).hexdigest()[:16]
        }
        
        # Intentar extraer fecha si es posible
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                metadata['extracted_date'] = match.group()
                break
        
        return metadata
    
    def process_pdf(self, pdf_path: str) -> bool:
        """Procesa un archivo PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            return self._process_text(text, pdf_path)
        except Exception as e:
            print(f"Error procesando PDF {pdf_path}: {e}")
            return False
    
    def process_docx(self, docx_path: str) -> bool:
        """Procesa un archivo DOCX"""
        try:
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return self._process_text(text, docx_path)
        except Exception as e:
            print(f"Error procesando DOCX {docx_path}: {e}")
            return False
    
    def process_web_url(self, url: str) -> bool:
        """Procesa una URL web"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remover scripts y styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            # Limpiar texto
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            return self._process_text(text, url)
        except Exception as e:
            print(f"Error procesando URL {url}: {e}")
            return False
    
    def process_text(self, text: str, source: str = "manual_input") -> bool:
        """Procesa texto directo"""
        return self._process_text(text, source)
    
    def _process_text(self, text: str, source: str) -> bool:
        """Procesa el texto y lo añade a la base vectorial"""
        try:
            # Limpiar texto
            text = re.sub(r'\s+', ' ', text).strip()
            
            if len(text) < 50:  # Muy corto
                print(f"Texto muy corto, omitiendo: {source}")
                return False
            
            # Extraer metadatos
            metadata = self._extract_metadata(text, source)
            
            # Verificar si ya existe
            if any(doc['doc_id'] == metadata['doc_id'] for doc in self.documents):
                print(f"Documento ya existe: {source}")
                return False
            
            # Dividir en chunks
            chunks = self._chunk_text(text)
            
            print(f"Procesando {source}: {len(chunks)} chunks")
            
            # Crear embeddings para cada chunk
            chunk_embeddings = self.embedding_model.encode(chunks)
            
            # Normalizar para cosine similarity
            faiss.normalize_L2(chunk_embeddings)
            
            # Añadir al índice FAISS
            self.index.add(chunk_embeddings)
            
            # Guardar metadatos
            doc_index = len(self.documents)
            self.documents.append(metadata)
            
            # Guardar chunks con sus metadatos
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    'doc_index': doc_index,
                    'chunk_index': i,
                    'text': chunk,
                    'source': source,
                    'doc_id': metadata['doc_id']
                }
                self.document_chunks.append(chunk_metadata)
            
            # Guardar datos
            self._save_data()
            
            print(f"✅ Procesado exitosamente: {source}")
            return True
            
        except Exception as e:
            print(f"Error procesando texto de {source}: {e}")
            return False
    
    def search(self, query: str, k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Busca documentos similares a la consulta
        
        Args:
            query: Consulta de búsqueda
            k: Número de resultados
            threshold: Umbral mínimo de similitud
        """
        if self.index.ntotal == 0:
            return []
        
        # Crear embedding de la consulta
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Buscar en el índice
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.document_chunks):
                chunk = self.document_chunks[idx]
                result = {
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'score': float(score),
                    'doc_id': chunk['doc_id'],
                    'chunk_index': chunk['chunk_index']
                }
                results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la base de datos"""
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.document_chunks),
            'index_size': self.index.ntotal,
            'embedding_model': self.model_name,
            'storage_path': str(self.storage_path)
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """Lista todos los documentos procesados"""
        return self.documents


# Ejemplo de uso
if __name__ == "__main__":
    # Crear el procesador
    processor = DocumentProcessor()
    
    # Procesar algunos documentos de ejemplo
    processor.process_text(
        "La inteligencia artificial es una rama de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.",
        "definicion_ia.txt"
    )
    
    processor.process_text(
        "El aprendizaje automático es un subconjunto de la inteligencia artificial que permite a las máquinas aprender y mejorar automáticamente a partir de la experiencia.",
        "definicion_ml.txt"
    )
    
    # Buscar
    results = processor.search("¿Qué es la inteligencia artificial?")
    
    print("\n🔍 Resultados de búsqueda:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Fuente: {result['source']}")
        print(f"   Texto: {result['text'][:200]}...")
        print()