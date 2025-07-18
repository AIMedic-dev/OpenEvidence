import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AzureOpenAI
from PruebaRAG import DocumentProcessor
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RAGResponse:
    """Respuesta del sistema RAG"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query: str

class RAGSystem:
    def __init__(self, 
                openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
                openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
                model_name: str = None,
                embedding_model: str = "all-MiniLM-L6-v2",
                storage_path: str = "knowledge_base"):
        """
        Inicializa el sistema RAG
        
        Args:
            openai_api_key: API key de OpenAI
            model_name: Modelo de OpenAI a usar
            embedding_model: Modelo de embeddings
            storage_path: Carpeta de almacenamiento
        """
        self.openai_client = AzureOpenAI(
            api_key=openai_api_key,
            api_version=api_version,
            azure_endpoint=openai_endpoint
        )
        self.deployment_name = deployment_name
        self.model_name = model_name or deployment_name
        
        # Inicializar procesador de documentos
        self.doc_processor = DocumentProcessor(
            model_name=embedding_model,
            storage_path=storage_path
        )
        
        # Configuración del sistema
        self.system_prompt = """Eres un asistente de investigación especializado en analizar documentos y responder preguntas basándote en evidencia textual.

INSTRUCCIONES:
1. Responde únicamente basándote en la información proporcionada en los documentos
2. Si no tienes información suficiente, di claramente que no puedes responder
3. Cita las fuentes cuando sea apropiado
4. Sé preciso y conciso
5. Si hay información contradictoria, menciónalo

FORMATO DE RESPUESTA:
- Respuesta clara y directa
- Menciona las fuentes relevantes
- Si es apropiado, incluye citas textuales"""
    
    def add_document(self, file_path: str) -> bool:
        """
        Añade un documento al sistema
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            bool: True si se procesó exitosamente
        """
        file_ext = file_path.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            return self.doc_processor.process_pdf(file_path)
        elif file_ext == 'docx':
            return self.doc_processor.process_docx(file_path)
        elif file_ext in ['txt', 'md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.doc_processor.process_text(content, file_path)
        else:
            print(f"Formato no soportado: {file_ext}")
            return False
    
    def add_web_content(self, url: str) -> bool:
        """
        Añade contenido web al sistema
        
        Args:
            url: URL a procesar
            
        Returns:
            bool: True si se procesó exitosamente
        """
        return self.doc_processor.process_web_url(url)
    
    def add_text(self, text: str, source: str = "manual_input") -> bool:
        """
        Añade texto directo al sistema
        
        Args:
            text: Texto a añadir
            source: Identificador de la fuente
            
        Returns:
            bool: True si se procesó exitosamente
        """
        return self.doc_processor.process_text(text, source)
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca documentos relevantes
        
        Args:
            query: Consulta de búsqueda
            k: Número de resultados
            
        Returns:
            List[Dict]: Lista de documentos relevantes
        """
        return self.doc_processor.search(query, k=k)
    
    def _prepare_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Prepara el contexto para el LLM
        
        Args:
            relevant_docs: Documentos relevantes
            
        Returns:
            str: Contexto formateado
        """
        if not relevant_docs:
            return "No se encontraron documentos relevantes."
        
        context = "DOCUMENTOS RELEVANTES:\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            context += f"DOCUMENTO {i}:\n"
            context += f"Fuente: {doc['source']}\n"
            context += f"Relevancia: {doc['score']:.3f}\n"
            context += f"Contenido: {doc['text']}\n\n"
        
        return context
    
    def _calculate_confidence(self, relevant_docs: List[Dict[str, Any]]) -> float:
        """
        Calcula un score de confianza basado en los documentos relevantes
        
        Args:
            relevant_docs: Documentos relevantes
            
        Returns:
            float: Score de confianza (0-1)
        """
        if not relevant_docs:
            return 0.0
        
        # Promedio de scores, con peso por posición
        weighted_score = 0.0
        total_weight = 0.0
        
        for i, doc in enumerate(relevant_docs):
            weight = 1.0 / (i + 1)  # Peso decreciente por posición
            weighted_score += doc['score'] * weight
            total_weight += weight
        
        confidence = weighted_score / total_weight if total_weight > 0 else 0.0
        return min(confidence, 1.0)
    
    def ask(self, question: str, max_docs: int = 5) -> RAGResponse:
        """
        Hace una pregunta al sistema RAG
        
        Args:
            question: Pregunta a responder
            max_docs: Máximo número de documentos a considerar
            
        Returns:
            RAGResponse: Respuesta del sistema
        """
        # 1. Buscar documentos relevantes
        relevant_docs = self.search_documents(question, k=max_docs)
        
        # 2. Preparar contexto
        context = self._prepare_context(relevant_docs)
        
        # 3. Calcular confianza
        confidence = self._calculate_confidence(relevant_docs)
        
        # 4. Preparar prompt para el LLM
        user_prompt = f"""PREGUNTA: {question}

{context}

Por favor, responde la pregunta basándote únicamente en la información proporcionada en los documentos."""
        
        try:
            # 5. Llamar al LLM
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            
            answer = response.choices[0].message.content
            
            # 6. Preparar respuesta
            return RAGResponse(
                answer=answer,
                sources=relevant_docs,
                confidence=confidence,
                query=question
            )
            
        except Exception as e:
            print(f"Error al generar respuesta: {e}")
            return RAGResponse(
                answer=f"Error al generar respuesta: {str(e)}",
                sources=relevant_docs,
                confidence=0.0,
                query=question
            )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema
        
        Returns:
            Dict: Estadísticas del sistema
        """
        return {
            **self.doc_processor.get_stats(),
            'llm_model': self.model_name
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Lista todos los documentos en el sistema
        
        Returns:
            List[Dict]: Lista de documentos
        """
        return self.doc_processor.list_documents()


# Clase para interfaz simple
class SimpleRAGInterface:
    def __init__(self, rag_system: RAGSystem):
        self.rag = rag_system
    
    def interactive_session(self):
        """Inicia una sesión interactiva"""
        print("🤖 Sistema RAG iniciado")
        print("Comandos disponibles:")
        print("  - Escribe una pregunta para buscar")
        print("  - '/add <archivo>' para añadir documento")
        print("  - '/web <url>' para añadir contenido web")
        print("  - '/stats' para ver estadísticas")
        print("  - '/docs' para listar documentos")
        print("  - '/quit' para salir")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n🔍 Pregunta: ").strip()
                
                if user_input.lower() == '/quit':
                    print("👋 ¡Hasta luego!")
                    break
                
                elif user_input.startswith('/add '):
                    file_path = user_input[5:].strip()
                    if self.rag.add_document(file_path):
                        print(f"✅ Documento añadido: {file_path}")
                    else:
                        print(f"❌ Error añadiendo: {file_path}")
                
                elif user_input.startswith('/web '):
                    url = user_input[5:].strip()
                    if self.rag.add_web_content(url):
                        print(f"✅ Contenido web añadido: {url}")
                    else:
                        print(f"❌ Error añadiendo: {url}")
                
                elif user_input == '/stats':
                    stats = self.rag.get_system_stats()
                    print("\n📊 Estadísticas del sistema:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                
                elif user_input == '/docs':
                    docs = self.rag.list_documents()
                    print(f"\n📚 Documentos ({len(docs)}):")
                    for i, doc in enumerate(docs, 1):
                        print(f"  {i}. {doc['source']} ({doc['tokens']} tokens)")
                
                elif user_input:
                    # Procesar pregunta
                    response = self.rag.ask(user_input)
                    
                    print(f"\n🤖 Respuesta (Confianza: {response.confidence:.2f}):")
                    print(response.answer)
                    
                    if response.sources:
                        print(f"\n📖 Fuentes ({len(response.sources)}):")
                        for i, source in enumerate(response.sources, 1):
                            print(f"  {i}. {source['source']} (Score: {source['score']:.3f})")
                    
                    print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")