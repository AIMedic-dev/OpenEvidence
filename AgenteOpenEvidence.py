import os
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
import trafilatura
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
import PyPDF2
import docx
import pandas as pd
from googleapiclient.discovery import build
import pickle
import time
from urllib.parse import urlparse
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalRAGSystem:
    def __init__(self, 
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                embedding_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
                google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"),
                google_cse_id=os.getenv("GOOGLE_CSE_ID"),
                storage_path: str = "knowledge_base",
                chunk_size: int = 800,
                chunk_overlap: int = 150,
                embedding_dimension: int = 3072):
        
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
        self.embedding_dimension = embedding_dimension
        
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        
        self.index = None
        self.documents_metadata = []
        self._load_or_create_index()
        
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        
        self.system_prompt = """Eres un asistente especializado que analiza literatura y documentos para proporcionar información precisa y basada en evidencia.

        INSTRUCCIONES:
        1. Utiliza únicamente información de fuentes confiables
        2. Proporciona respuestas detalladas y técnicamente precisas cuando tengas evidencia suficiente
        3. Siempre cita las fuentes específicas cuando sea posible
        4. Si la información es contradictoria entre fuentes, analízala críticamente
        5. Menciona las limitaciones de los estudios o la evidencia disponible
        6. Estructura las respuestas de manera clara y profesional

        IMPORTANTE: Esta información es solo para fines educativos y de investigación."""

        logger.info(f"Sistema RAG inicializado con Azure Embedding: {self.embedding_deployment}")

    def _load_or_create_index(self):
        try:
            index_path = os.path.join(self.storage_path, 'faiss_index.bin')
            metadata_path = os.path.join(self.storage_path, 'metadata.pkl')
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.index = faiss.read_index(index_path)
                with open(metadata_path, 'rb') as f:
                    self.documents_metadata = pickle.load(f)
                logger.info(f"Índice FAISS cargado: {len(self.documents_metadata)} documentos")
            else:
                os.makedirs(self.storage_path, exist_ok=True)
                self.index = faiss.IndexFlatIP(self.embedding_dimension)
                self.documents_metadata = []
                logger.info("Nuevo índice FAISS creado")
                
        except Exception as e:
            logger.error(f"Error inicializando FAISS: {e}")
            os.makedirs(self.storage_path, exist_ok=True)
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.documents_metadata = []

    def _save_index(self):
        try:
            index_path = os.path.join(self.storage_path, 'faiss_index.bin')
            metadata_path = os.path.join(self.storage_path, 'metadata.pkl')
            
            faiss.write_index(self.index, index_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.documents_metadata, f)
            logger.info("Índice FAISS guardado exitosamente")
            
        except Exception as e:
            logger.error(f"Error guardando índice: {e}")

    def get_batch_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        embeddings = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"Procesando batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=self.embedding_deployment
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                if i + batch_size < len(texts):
                    time.sleep(0.1)
                    
            return embeddings
            
        except Exception as e:
            logger.error(f"Error obteniendo embeddings por lotes: {e}")
            return []

    def extract_content_trafilatura(self, url: str) -> Optional[str]:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                content = trafilatura.extract(downloaded, 
                                            include_comments=False,
                                            include_tables=True,
                                            include_formatting=True)
                
                if content:
                    logger.info(f"Contenido extraído con trafilatura: {url}")
                    return content
            return None
            
        except Exception as e:
            logger.error(f"Error con trafilatura en {url}: {e}")
            return None

    def extract_content_selenium(self, url: str) -> Optional[str]:
        driver = None
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.get(url)
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.main-content', '.article-content', '.abstract',
                '.full-text', '.article-body'
            ]
            
            content = ""
            for selector in content_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        content = elements[0].get_attribute('innerText')
                        break
                except:
                    continue
            
            if not content:
                content = driver.find_element(By.TAG_NAME, "body").get_attribute('innerText')
            
            if content:
                logger.info(f"Contenido extraído con Selenium: {url}")
                return content[:8000]
            return None
                
        except Exception as e:
            logger.error(f"Error con Selenium en {url}: {e}")
            return None
        finally:
            if driver:
                driver.quit()

    def medical_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        if not self.google_api_key or not self.google_cse_id:
            logger.warning("Google Search API no configurada")
            return []
        
        try:
            service = build("customsearch", "v1", developerKey=self.google_api_key)
            
            result = service.cse().list(
                q=query,
                cx=self.google_cse_id,
                num=num_results
            ).execute()
            
            search_results = []
            
            if 'items' in result:
                for item in result['items']:
                    url = item['link']
                    
                    content = self.extract_content_trafilatura(url)
                    
                    if not content:
                        content = self.extract_content_selenium(url)
                    
                    if content:
                        search_results.append({
                            'title': item['title'],
                            'link': url,
                            'snippet': item['snippet'],
                            'content': content,
                            'source_type': 'WEB'
                        })
            
            logger.info(f"Encontrados {len(search_results)} resultados para: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []

    def process_file(self, file_path: str, filename: str) -> str:
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
            logger.error(f"Error procesando archivo {filename}: {e}")
            return ""

    def _process_pdf(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        content += f"\n--- Página {page_num + 1} ---\n{page_text}\n"
            return content
        except Exception as e:
            logger.error(f"Error procesando PDF: {e}")
            return ""

    def _process_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            content = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content += paragraph.text + "\n"
            
            for table in doc.tables:
                for row in table.rows:
                    row_data = []
                    for cell in row.cells:
                        row_data.append(cell.text.strip())
                    content += " | ".join(row_data) + "\n"
                    
            return content
        except Exception as e:
            logger.error(f"Error procesando DOCX: {e}")
            return ""

    def _process_txt(self, file_path: str) -> str:
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            return ""
        except Exception as e:
            logger.error(f"Error procesando TXT: {e}")
            return ""

    def _process_csv(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
            
            content = f"DATASET - Información estructurada:\n\n"
            content += f"Dimensiones: {df.shape[0]} registros x {df.shape[1]} variables\n\n"
            content += f"Variables: {', '.join(df.columns.tolist())}\n\n"
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                content += "ESTADÍSTICAS DESCRIPTIVAS:\n"
                content += df[numeric_cols].describe().to_string() + "\n\n"
            
            content += "MUESTRA DE DATOS (primeras 10 filas):\n"
            content += df.head(10).to_string() + "\n\n"
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                content += "VARIABLES CATEGÓRICAS:\n"
                for col in categorical_cols:
                    unique_vals = df[col].value_counts().head(10)
                    content += f"{col}:\n{unique_vals.to_string()}\n\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Error procesando CSV: {e}")
            return ""

    def _process_excel(self, file_path: str) -> str:
        try:
            excel_file = pd.ExcelFile(file_path)
            content = f"ARCHIVO EXCEL:\n\n"
            content += f"Hojas disponibles: {', '.join(excel_file.sheet_names)}\n\n"
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                content += f"\n=== HOJA: {sheet_name} ===\n"
                content += f"Dimensiones: {df.shape[0]} registros x {df.shape[1]} variables\n"
                content += f"Variables: {', '.join(df.columns.tolist())}\n\n"
                
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    content += "ESTADÍSTICAS DESCRIPTIVAS:\n"
                    content += df[numeric_cols].describe().to_string() + "\n\n"
                
                content += "MUESTRA DE DATOS:\n"
                content += df.head(5).to_string() + "\n\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Error procesando Excel: {e}")
            return ""

    def intelligent_chunking(self, text: str, source: str) -> List[Dict[str, Any]]:
        separators = [
            '\n\nINTRODUCTION\n', '\n\nMETHODS\n', 
            '\n\nRESULTS\n', '\n\nDISCUSSION\n', '\n\nCONCLUSION\n',
            '\n\nBackground\n', '\n\nObjective\n', '\n\nStudy Design\n',
            '\n\nFindings\n', '\n\nInterpretation\n'
        ]
        
        chunks = []
        current_section = ""
        current_chunk = ""
        
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            is_section_header = any(sep.strip().upper() in paragraph.upper() 
                                    for sep in separators)
            
            if is_section_header:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'section': current_section,
                        'tokens': len(current_chunk.split())
                    })
                current_section = paragraph
                current_chunk = paragraph + "\n\n"
            else:
                if len(current_chunk) + len(paragraph) > self.chunk_size:
                    if current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'section': current_section,
                            'tokens': len(current_chunk.split())
                        })
                    current_chunk = paragraph + "\n\n"
                else:
                    current_chunk += paragraph + "\n\n"
        
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'section': current_section,
                'tokens': len(current_chunk.split())
            })
        
        final_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'text': chunk['text'],
                'source': source,
                'chunk_index': i,
                'section': chunk['section'],
                'tokens': chunk['tokens'],
                'doc_id': f"{source}_{i}"
            }
            
            if i > 0 and len(chunks[i-1]['text']) > 0:
                overlap_text = chunks[i-1]['text'][-self.chunk_overlap:]
                chunk_data['text'] = overlap_text + "\n\n" + chunk['text']
            
            final_chunks.append(chunk_data)
        
        return final_chunks

    def add_document(self, text: str, source: str) -> bool:
        try:
            chunks = self.intelligent_chunking(text, source)
            
            if not chunks:
                logger.warning(f"No se generaron chunks para: {source}")
                return False
            
            texts = [chunk['text'] for chunk in chunks]
            
            embeddings = self.get_batch_embeddings(texts)
            
            if not embeddings or len(embeddings) != len(texts):
                logger.error(f"Error generando embeddings para: {source}")
                return False
            
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            
            start_idx = len(self.documents_metadata)
            self.index.add(embeddings_array)
            
            for i, chunk in enumerate(chunks):
                self.documents_metadata.append({
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'chunk_index': chunk['chunk_index'],
                    'section': chunk['section'],
                    'tokens': chunk['tokens'],
                    'doc_id': chunk['doc_id'],
                    'faiss_idx': start_idx + i
                })
            
            self._save_index()
            
            logger.info(f"Documento procesado: {len(chunks)} chunks agregados - {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error agregando documento: {e}")
            return False

    def add_file(self, file_path: str, filename: str) -> bool:
        try:
            content = self.process_file(file_path, filename)
            
            if not content:
                logger.warning(f"No se pudo procesar: {filename}")
                return False
            
            return self.add_document(content, filename)
            
        except Exception as e:
            logger.error(f"Error añadiendo archivo {filename}: {e}")
            return False

    def search_documents(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        try:
            if self.index.ntotal == 0:
                logger.warning("Índice FAISS vacío")
                return []
            
            query_embeddings = self.get_batch_embeddings([query])
            
            if not query_embeddings:
                logger.error("Error generando embedding de consulta")
                return []
            
            query_vector = np.array([query_embeddings[0]]).astype('float32')
            faiss.normalize_L2(query_vector)
            
            scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents_metadata):
                    doc = self.documents_metadata[idx].copy()
                    doc['similarity_score'] = float(score)
                    results.append(doc)
            
            results = [r for r in results if r['similarity_score'] > 0.5]
            
            logger.info(f"Búsqueda local: {len(results)} documentos relevantes encontrados")
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda local: {e}")
            return []

    def enhanced_medical_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        local_results = self.search_documents(query, k=k)
        
        high_quality_results = [doc for doc in local_results if doc['similarity_score'] > 0.7]
        
        logger.info(f"Resultados locales de alta calidad: {len(high_quality_results)}/{len(local_results)}")
        
        if len(high_quality_results) < 3:
            logger.info("Buscando información adicional...")
            web_results = self.medical_search(query, num_results=5)
            
            for web_result in web_results:
                web_source = f"WEB: {web_result['title']}"
                self.add_document(web_result['content'], web_source)
            
            if web_results:
                local_results = self.search_documents(query, k=k)
                high_quality_results = [doc for doc in local_results if doc['similarity_score'] > 0.7]
                logger.info(f"Resultados después de búsqueda web: {len(high_quality_results)}")
        
        return high_quality_results or local_results[:5]

    def ask_medical_question(self, question: str, max_docs: int = 10) -> Dict[str, Any]:
        relevant_docs = self.enhanced_medical_search(question, k=max_docs)
        
        context = self._prepare_medical_context(relevant_docs)
        
        confidence = self._calculate_evidence_confidence(relevant_docs)
        
        user_prompt = f"""CONSULTA: {question}

        EVIDENCIA DISPONIBLE:
        {context}

        INSTRUCCIONES:
        - Proporciona una respuesta basada únicamente en la evidencia proporcionada
        - Menciona las limitaciones de los estudios
        - Si hay información contradictoria, analízala críticamente
        - Estructura la respuesta de manera clara y profesional

        Responde de manera estructurada y técnicamente precisa."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2500
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'sources': relevant_docs,
                'confidence': confidence,
                'query': question,
                'context_length': len(context),
                'local_sources': len([d for d in relevant_docs if not d['source'].startswith('WEB:')]),
                'web_sources': len([d for d in relevant_docs if d['source'].startswith('WEB:')]),
                'evidence_level': self._classify_evidence_level(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return {
                'answer': f"Error al generar respuesta: {str(e)}",
                'sources': relevant_docs,
                'confidence': 0.0,
                'query': question,
                'context_length': 0,
                'local_sources': 0,
                'web_sources': 0,
                'evidence_level': 'Unknown'
            }

    def _prepare_medical_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        if not relevant_docs:
            return "No se encontraron documentos relevantes."
        
        context = "=== EVIDENCIA ===\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            source_type = "🌐 FUENTE WEB" if doc['source'].startswith('WEB:') else "📄 DOCUMENTO LOCAL"
            context += f"EVIDENCIA {i} ({source_type}): {doc['source']}\n"
            context += f"Sección: {doc.get('section', 'General')}\n"
            context += f"Relevancia: {doc['similarity_score']:.3f}\n"
            context += f"Tokens: {doc.get('tokens', 0)}\n"
            context += f"Contenido:\n{doc['text']}\n"
            context += "=" * 80 + "\n\n"
        
        return context

    def _calculate_evidence_confidence(self, relevant_docs: List[Dict[str, Any]]) -> float:
        if not relevant_docs:
            return 0.0
        
        avg_similarity = sum(doc['similarity_score'] for doc in relevant_docs) / len(relevant_docs)
        
        quantity_factor = min(len(relevant_docs) / 8, 1.0)
        
        unique_sources = len(set(doc['source'] for doc in relevant_docs))
        diversity_factor = min(unique_sources / 5, 1.0)
        
        web_count = len([d for d in relevant_docs if d['source'].startswith('WEB:')])
        web_factor = min(web_count * 0.1, 0.3)
        
        confidence = (avg_similarity * 0.4 + quantity_factor * 0.3 + 
                     diversity_factor * 0.2 + web_factor * 0.1)
        
        return min(confidence, 1.0)

    def _classify_evidence_level(self, relevant_docs: List[Dict[str, Any]]) -> str:
        if not relevant_docs:
            return "No Evidence"
        
        source_count = len(relevant_docs)
        high_quality_count = len([d for d in relevant_docs if d['similarity_score'] > 0.8])
        
        if high_quality_count >= 3:
            return "Level I (High Quality Multiple Sources)"
        elif high_quality_count >= 2:
            return "Level II (Good Quality Sources)"
        elif source_count >= 3:
            return "Level III (Multiple Sources)"
        else:
            return "Level IV (Limited Evidence)"

    def delete_medical_documents(self, sources: List[str]) -> bool:
        try:
            indices_to_remove = []
            new_metadata = []
            
            for i, doc_meta in enumerate(self.documents_metadata):
                if doc_meta['source'] in sources:
                    indices_to_remove.append(doc_meta['faiss_idx'])
                else:
                    new_metadata.append(doc_meta)
            
            if not indices_to_remove:
                logger.warning("No se encontraron documentos para eliminar")
                return False
            
            if new_metadata:
                remaining_texts = [doc['text'] for doc in new_metadata]
                remaining_embeddings = self.get_batch_embeddings(remaining_texts)
                
                if remaining_embeddings:
                    new_index = faiss.IndexFlatIP(self.embedding_dimension)
                    embeddings_array = np.array(remaining_embeddings).astype('float32')
                    faiss.normalize_L2(embeddings_array)
                    new_index.add(embeddings_array)
                    
                    for i, doc_meta in enumerate(new_metadata):
                        doc_meta['faiss_idx'] = i
                    
                    self.index = new_index
                    self.documents_metadata = new_metadata
                else:
                    logger.error("Error regenerando embeddings")
                    return False
            else:
                self.index = faiss.IndexFlatIP(self.embedding_dimension)
                self.documents_metadata = []
            
            self._save_index()
            
            logger.info(f"Eliminados {len(sources)} documentos")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando documentos: {e}")
            return False

    def clear_medical_database(self) -> bool:
        try:
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.documents_metadata = []
            
            self._save_index()
            
            logger.info("Base de datos limpiada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error limpiando base de datos: {e}")
            return False

    def validate_medical_source(self, url_or_content: str) -> Dict[str, Any]:
        try:
            is_url = url_or_content.startswith('http')
            
            if is_url:
                domain = urlparse(url_or_content).netloc.lower()
                
                return {
                    'source_type': 'URL',
                    'domain': domain,
                    'recommendation': 'Approved'
                }
            else:
                return {
                    'source_type': 'CONTENT',
                    'recommendation': 'Approved'
                }
                
        except Exception as e:
            logger.error(f"Error validando fuente: {e}")
            return {
                'error': str(e),
                'recommendation': 'Error in Validation'
            }