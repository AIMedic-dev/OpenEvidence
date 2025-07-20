import os
from typing import List, Dict, Any
from openai import AzureOpenAI
import chromadb
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import PyPDF2
import docx
import pandas as pd
from googleapiclient.discovery import build

load_dotenv()

class ImprovedRAGSystem:
    def __init__(self, 
                openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
                openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
                embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
                google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY"),
                google_cse_id = os.getenv("GOOGLE_CSE_ID"),
                storage_path: str = "knowledge_base",
                chunk_size: int = 1000,
                chunk_overlap: int = 200):
        
        self.openai_client = AzureOpenAI(
            api_key=openai_api_key,
            api_version=api_version,
            azure_endpoint=openai_endpoint
        )
        self.deployment_name = deployment_name
        self.embedding_deployment = embedding_deployment or "text-embedding-3-large"
        self.storage_path = storage_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configuración para Google Search
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        
        print(f"🔧 Usando Azure Embedding: {self.embedding_deployment}")
        
        # Inicializar base de datos vectorial
        self.client = chromadb.PersistentClient(path=storage_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prompt optimizado
        self.system_prompt = """Eres un asistente de investigación experto que analiza documentos para responder preguntas de manera precisa y completa.

INSTRUCCIONES:
1. Analiza TODA la información proporcionada en los documentos de manera exhaustiva
2. Proporciona respuestas completas y detalladas cuando tengas información suficiente
3. Si falta información específica, menciona qué información adicional sería útil
4. Siempre cita las fuentes específicas de donde obtienes la información
5. Si hay información contradictoria entre fuentes, analízala y menciónalo
6. Estructura tu respuesta de manera clara y organizada"""

    def get_azure_embedding(self, text: str) -> List[float]:
        """Obtener embedding usando Azure OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
                model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error obteniendo embedding de Azure: {e}")
            return None

    def process_file(self, file_path: str, filename: str) -> str:
        """Procesa diferentes tipos de archivos"""
        try:
            file_extension = filename.lower().split('.')[-1]
            content = ""
            
            if file_extension == 'pdf':
                content = self._process_pdf(file_path)
            elif file_extension in ['docx', 'doc']:
                content = self._process_docx(file_path)
            elif file_extension == 'txt':
                content = self._process_txt(file_path)
            elif file_extension in ['csv']:
                content = self._process_csv(file_path)
            elif file_extension in ['xlsx', 'xls']:
                content = self._process_excel(file_path)
            else:
                raise ValueError(f"Tipo de archivo no soportado: {file_extension}")
            
            return content
            
        except Exception as e:
            print(f"❌ Error procesando archivo {filename}: {e}")
            return ""

    def _process_pdf(self, file_path: str) -> str:
        """Procesa archivos PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            return content
        except Exception as e:
            print(f"❌ Error procesando PDF: {e}")
            return ""

    def _process_docx(self, file_path: str) -> str:
        """Procesa archivos DOCX"""
        try:
            doc = docx.Document(file_path)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content
        except Exception as e:
            print(f"❌ Error procesando DOCX: {e}")
            return ""

    def _process_txt(self, file_path: str) -> str:
        """Procesa archivos TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"❌ Error procesando TXT: {e}")
            return ""

    def _process_csv(self, file_path: str) -> str:
        """Procesa archivos CSV"""
        try:
            df = pd.read_csv(file_path)
            
            # Crear contenido descriptivo del CSV
            content = f"ARCHIVO CSV - Información estructurada:\n\n"
            content += f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas\n\n"
            content += f"Columnas: {', '.join(df.columns.tolist())}\n\n"
            
            # Añadir estadísticas básicas para columnas numéricas
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                content += "ESTADÍSTICAS NUMÉRICAS:\n"
                content += df[numeric_cols].describe().to_string() + "\n\n"
            
            # Añadir primeras filas como contexto
            content += "PRIMERAS 10 FILAS:\n"
            content += df.head(10).to_string() + "\n\n"
            
            # Añadir información de valores únicos para columnas categóricas
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                content += "VALORES ÚNICOS (COLUMNAS CATEGÓRICAS):\n"
                for col in categorical_cols:
                    unique_vals = df[col].unique()[:10]  # Solo primeros 10
                    content += f"{col}: {', '.join(str(v) for v in unique_vals)}\n"
            
            return content
            
        except Exception as e:
            print(f"❌ Error procesando CSV: {e}")
            return ""

    def _process_excel(self, file_path: str) -> str:
        """Procesa archivos Excel"""
        try:
            # Leer todas las hojas
            excel_file = pd.ExcelFile(file_path)
            content = f"ARCHIVO EXCEL - Información estructurada:\n\n"
            content += f"Hojas disponibles: {', '.join(excel_file.sheet_names)}\n\n"
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                content += f"\n--- HOJA: {sheet_name} ---\n"
                content += f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas\n"
                content += f"Columnas: {', '.join(df.columns.tolist())}\n\n"
                
                # Estadísticas para columnas numéricas
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    content += "ESTADÍSTICAS NUMÉRICAS:\n"
                    content += df[numeric_cols].describe().to_string() + "\n\n"
                
                # Primeras filas
                content += "PRIMERAS 5 FILAS:\n"
                content += df.head(5).to_string() + "\n\n"
            
            return content
            
        except Exception as e:
            print(f"❌ Error procesando Excel: {e}")
            return ""

    def google_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Busca información usando Google Custom Search API"""
        if not self.google_api_key or not self.google_cse_id:
            print("⚠️ Google Search API no configurada")
            return []
        
        try:
            service = build("customsearch", "v1", developerKey=self.google_api_key)
            
            # Ejecutar búsqueda
            result = service.cse().list(
                q=query,
                cx=self.google_cse_id,
                num=num_results
            ).execute()
            
            search_results = []
            
            if 'items' in result:
                for item in result['items']:
                    # Extraer contenido de la página
                    try:
                        page_content = self._extract_page_content(item['link'])
                        if page_content:
                            search_results.append({
                                'title': item['title'],
                                'link': item['link'],
                                'snippet': item['snippet'],
                                'content': page_content
                            })
                    except Exception as e:
                        print(f"❌ Error extrayendo contenido de {item['link']}: {e}")
                        continue
            
            print(f"🌐 Encontrados {len(search_results)} resultados web para: {query}")
            return search_results
            
        except Exception as e:
            print(f"❌ Error en búsqueda de Google: {e}")
            return []

    def _extract_page_content(self, url: str) -> str:
        """Extrae contenido limpio de una página web"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Eliminar scripts y estilos
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extraer texto
            text = soup.get_text()
            
            # Limpiar texto
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limitar tamaño
            return text[:5000]  # Primeros 5000 caracteres
            
        except Exception as e:
            print(f"❌ Error extrayendo contenido de {url}: {e}")
            return ""

    def improve_chunking(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Estrategia de chunking mejorada"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'text': chunk,
                'source': source,
                'chunk_index': i,
                'doc_id': f"{source}_{i}",
                'tokens': len(chunk.split())
            }
            
            if i > 0 and len(chunks[i-1]) > 0:
                overlap_text = chunks[i-1][-self.chunk_overlap:]
                chunk_data['text'] = overlap_text + " " + chunk
            
            overlapped_chunks.append(chunk_data)
        
        return overlapped_chunks

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Búsqueda en documentos locales"""
        try:
            # Generar embedding de la consulta usando Azure
            query_embedding = self.get_azure_embedding(query)
            
            if query_embedding is None:
                return []
            
            # Buscar en ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    score = 1 - distance
                    
                    formatted_results.append({
                        'text': doc,
                        'source': metadata.get('source', 'unknown'),
                        'score': score,
                        'chunk_index': metadata.get('chunk_index', 0),
                        'doc_id': metadata.get('doc_id', f'doc_{i}'),
                        'tokens': metadata.get('tokens', 0)
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error en búsqueda local: {e}")
            return []

    def enhanced_search(self, query: str, k: int = 8, min_score: float = 0.6) -> List[Dict[str, Any]]:
        """Búsqueda mejorada: primero local, luego web si es necesario"""
        
        # 1. Búsqueda en documentos locales
        local_results = self.search_documents(query, k=k)
        
        # 2. Filtrar por score mínimo
        good_results = [doc for doc in local_results if doc['score'] >= min_score]
        
        print(f"🔍 Resultados locales de calidad: {len(good_results)}/{len(local_results)}")
        
        # 3. Si no hay suficientes resultados de calidad, buscar en web
        if len(good_results) < 3:
            print("🌐 Buscando información adicional en web...")
            web_results = self.google_search(query, num_results=5)
            
            # 4. Procesar resultados web y agregarlos temporalmente
            for web_result in web_results:
                web_source = f"WEB: {web_result['title']}"
                self.add_text_improved(web_result['content'], web_source)
            
            # 5. Buscar de nuevo con la información nueva
            if web_results:
                local_results = self.search_documents(query, k=k)
                good_results = [doc for doc in local_results if doc['score'] >= min_score]
                print(f"🔍 Resultados después de web search: {len(good_results)}")
        
        return good_results or local_results[:5]  # Al menos devolver algo

    def add_text_improved(self, text: str, source: str) -> bool:
        """Versión mejorada para agregar texto con Azure embeddings"""
        try:
            chunks = self.improve_chunking(text, source)
            
            if not chunks:
                return False
            
            # Generar embeddings usando Azure
            embeddings = []
            texts = [chunk['text'] for chunk in chunks]
            
            for text_chunk in texts:
                embedding = self.get_azure_embedding(text_chunk)
                if embedding is None:
                    print(f"❌ Error generando embedding para chunk")
                    return False
                embeddings.append(embedding)
            
            # Preparar datos para ChromaDB
            ids = [chunk['doc_id'] for chunk in chunks]
            metadatas = [
                {
                    'source': chunk['source'],
                    'chunk_index': chunk['chunk_index'],
                    'doc_id': chunk['doc_id'],
                    'tokens': chunk['tokens']
                }
                for chunk in chunks
            ]
            
            # Añadir a la base de datos
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Texto procesado: {len(chunks)} chunks creados")
            return True
            
        except Exception as e:
            print(f"❌ Error procesando texto: {e}")
            return False

    def add_file(self, file_path: str, filename: str) -> bool:
        """Añade un archivo al sistema procesando su contenido"""
        try:
            content = self.process_file(file_path, filename)
            
            if not content:
                return False
            
            return self.add_text_improved(content, filename)
            
        except Exception as e:
            print(f"❌ Error añadiendo archivo {filename}: {e}")
            return False

    def ask_enhanced(self, question: str, max_docs: int = 8) -> Dict[str, Any]:
        """Versión mejorada de ask con búsqueda inteligente"""
        
        # Búsqueda mejorada (local + web si es necesario)
        relevant_docs = self.enhanced_search(question, k=max_docs)
        
        # Preparar contexto
        context = self._prepare_enhanced_context(relevant_docs)
        
        # Calcular confianza
        confidence = self._calculate_confidence(relevant_docs)
        
        # Prompt mejorado
        user_prompt = f"""PREGUNTA: {question}

CONTEXTO DE DOCUMENTOS:
{context}

Proporciona una respuesta completa y estructurada basada en la información disponible."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'sources': relevant_docs,
                'confidence': confidence,
                'query': question,
                'context_length': len(context),
                'local_sources': len([d for d in relevant_docs if not d['source'].startswith('WEB:')]),
                'web_sources': len([d for d in relevant_docs if d['source'].startswith('WEB:')])
            }
            
        except Exception as e:
            print(f"❌ Error generando respuesta: {e}")
            return {
                'answer': f"Error al generar respuesta: {str(e)}",
                'sources': relevant_docs,
                'confidence': 0.0,
                'query': question,
                'context_length': 0,
                'local_sources': 0,
                'web_sources': 0
            }

    def _prepare_enhanced_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Prepara contexto mejorado"""
        if not relevant_docs:
            return "No se encontraron documentos relevantes."
        
        context = "=== INFORMACIÓN RELEVANTE ===\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            source_type = "🌐 WEB" if doc['source'].startswith('WEB:') else "📄 LOCAL"
            context += f"FUENTE {i} ({source_type}): {doc['source']}\n"
            context += f"Relevancia: {doc['score']:.3f}\n"
            context += f"Contenido:\n{doc['text']}\n"
            context += "-" * 50 + "\n\n"
        
        return context

    def _calculate_confidence(self, relevant_docs: List[Dict[str, Any]]) -> float:
        """Calcula la confianza basada en la calidad de los documentos"""
        if not relevant_docs:
            return 0.0
        
        avg_score = sum(doc['score'] for doc in relevant_docs) / len(relevant_docs)
        doc_count_factor = min(len(relevant_docs) / 5, 1.0)
        
        return avg_score * doc_count_factor

    def get_documents_list(self) -> List[Dict[str, Any]]:
        """Obtiene lista de documentos únicos"""
        try:
            results = self.collection.get()
            
            if not results['metadatas']:
                return []
            
            # Agrupar por fuente
            sources = {}
            for metadata in results['metadatas']:
                source = metadata['source']
                if source not in sources:
                    sources[source] = {
                        'source': source,
                        'chunks': 0,
                        'tokens': 0,
                        'type': 'WEB' if source.startswith('WEB:') else 'LOCAL'
                    }
                sources[source]['chunks'] += 1
                sources[source]['tokens'] += metadata.get('tokens', 0)
            
            return list(sources.values())
            
        except Exception as e:
            print(f"❌ Error obteniendo lista de documentos: {e}")
            return []

    def delete_documents(self, sources: List[str]) -> bool:
        """Elimina documentos seleccionados"""
        try:
            for source in sources:
                # Buscar todos los chunks del documento
                results = self.collection.get(where={"source": source})
                
                if results['ids']:
                    # Eliminar los chunks
                    self.collection.delete(ids=results['ids'])
                    print(f"🗑️ Documento eliminado: {source}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error eliminando documentos: {e}")
            return False

    def clear_database(self) -> bool:
        """Limpia completamente la base de datos"""
        try:
            self.client.delete_collection("documents")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            print("🗑️ Base de datos limpiada exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error limpiando base de datos: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema"""
        try:
            collection_count = self.collection.count()
            docs_list = self.get_documents_list()
            
            local_docs = len([d for d in docs_list if d['type'] == 'LOCAL'])
            web_docs = len([d for d in docs_list if d['type'] == 'WEB'])
            
            return {
                'total_chunks': collection_count,
                'total_documents': len(docs_list),
                'local_documents': local_docs,
                'web_documents': web_docs,
                'embedding_model': self.embedding_deployment
            }
        except:
            return {
                'total_chunks': 0,
                'total_documents': 0,
                'local_documents': 0,
                'web_documents': 0,
                'embedding_model': self.embedding_deployment
            }