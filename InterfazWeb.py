import streamlit as st
import os
from pathlib import Path
import tempfile
from AgenteOpenEvidence import RAGSystem, RAGResponse
import time
from dotenv import load_dotenv

load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="RAG System - OpenEvidence",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def initialize_rag_system():
    """Inicializa el sistema RAG con Azure OpenAI"""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not all([api_key, endpoint, deployment, api_version]):
        st.error("⚠️ Faltan variables de entorno para Azure OpenAI")
        st.stop()

    return RAGSystem(
        openai_api_key=api_key,
        openai_endpoint=endpoint,
        deployment_name=deployment,
        api_version=api_version,
        model_name=deployment,
        storage_path="streamlit_knowledge_base"
    )

# Función para mostrar estadísticas
def show_stats(rag_system):
    """Muestra estadísticas del sistema"""
    stats = rag_system.get_system_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Documentos", stats.get('total_documents', 0))
    
    with col2:
        st.metric("📄 Chunks", stats.get('total_chunks', 0))
    
    with col3:
        st.metric("🧠 Modelo LLM", stats.get('llm_model', 'N/A'))
    
    with col4:
        st.metric("🔤 Embeddings", stats.get('embedding_model', 'N/A'))

# Función para mostrar documentos
def show_documents(rag_system):
    """Muestra lista de documentos"""
    docs = rag_system.list_documents()
    
    if not docs:
        st.info("📭 No hay documentos en el sistema")
        return
    
    st.subheader("📚 Documentos en el sistema")
    
    for i, doc in enumerate(docs, 1):
        with st.expander(f"📄 {doc['source']} ({doc.get('tokens', 0)} tokens)"):
            st.write(f"**ID:** {doc.get('doc_id', 'N/A')}")
            st.write(f"**Procesado:** {doc.get('processed_at', 'N/A')}")
            st.write(f"**Tokens:** {doc.get('tokens', 0)}")
            if 'extracted_date' in doc:
                st.write(f"**Fecha extraída:** {doc['extracted_date']}")

# Función para mostrar respuesta RAG
def show_rag_response(response: RAGResponse):
    """Muestra la respuesta del sistema RAG"""
    # Respuesta principal
    st.markdown("### 🤖 Respuesta")
    st.write(response.answer)
    
    # Barra de confianza
    st.markdown("### 📊 Confianza")
    confidence_color = "green" if response.confidence > 0.7 else "orange" if response.confidence > 0.4 else "red"
    st.progress(response.confidence, text=f"Confianza: {response.confidence:.2f}")
    
    # Fuentes utilizadas
    if response.sources:
        st.markdown("### 📖 Fuentes utilizadas")
        
        for i, source in enumerate(response.sources, 1):
            with st.expander(f"📄 Fuente {i}: {source['source']} (Score: {source['score']:.3f})"):
                st.write(source['text'])
                st.caption(f"Chunk {source.get('chunk_index', 'N/A')} - ID: {source.get('doc_id', 'N/A')}")

# Función principal
def main():
    st.title("🔍 Sistema RAG - OpenEvidence Clone")
    st.markdown("---")
    
    # Inicializar sistema RAG
    rag_system = initialize_rag_system()
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Mostrar estadísticas
        show_stats(rag_system)
        
        st.markdown("---")
        
        # Sección de carga de documentos
        st.header("📁 Cargar Documentos")
        
        # Upload de archivos
        uploaded_files = st.file_uploader(
            "Selecciona archivos",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'md'],
            key="file_uploader"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Crear una clave única para cada botón
                button_key = f"process_{uploaded_file.name}_{uploaded_file.size}"
                
                if st.button(f"📤 Procesar {uploaded_file.name}", key=button_key):
                    with st.spinner(f"Procesando {uploaded_file.name}..."):
                        try:
                            # Guardar archivo temporalmente
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            # Procesar archivo
                            success = rag_system.add_document(tmp_path)
                            
                            # Limpiar archivo temporal
                            os.unlink(tmp_path)
                            
                            if success:
                                st.success(f"✅ {uploaded_file.name} procesado exitosamente")
                                time.sleep(1)  # Pequeña pausa para mostrar el mensaje
                                st.rerun()  # Reemplaza st.experimental_rerun()
                            else:
                                st.error(f"❌ Error procesando {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"❌ Error procesando {uploaded_file.name}: {str(e)}")
        
        st.markdown("---")
        
        # Añadir contenido web
        st.header("🌐 Contenido Web")
        url_input = st.text_input("URL a procesar", key="url_input")
        
        if st.button("🌐 Procesar URL", key="process_url_btn") and url_input:
            with st.spinner(f"Procesando {url_input}..."):
                try:
                    success = rag_system.add_web_content(url_input)
                    
                    if success:
                        st.success(f"✅ URL procesada exitosamente")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Error procesando URL")
                except Exception as e:
                    st.error(f"❌ Error procesando URL: {str(e)}")
        
        st.markdown("---")
        
        # Añadir texto manual
        st.header("✍️ Texto Manual")
        text_input = st.text_area("Texto a añadir", height=100, key="text_input")
        source_name = st.text_input("Nombre de la fuente", value="texto_manual", key="source_name")
        
        if st.button("➕ Añadir Texto", key="add_text_btn") and text_input:
            with st.spinner("Procesando texto..."):
                try:
                    success = rag_system.add_text(text_input, source_name)
                    
                    if success:
                        st.success("✅ Texto añadido exitosamente")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Error añadiendo texto")
                except Exception as e:
                    st.error(f"❌ Error añadiendo texto: {str(e)}")
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat con tus documentos")
        
        # Inicializar historial de chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Mostrar historial de chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "response" in message:
                    show_rag_response(message["response"])
                else:
                    st.write(message["content"])
        
        # Input de chat
        if prompt := st.chat_input("Haz una pregunta sobre tus documentos..."):
            # Añadir mensaje del usuario
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generar respuesta
            with st.chat_message("assistant"):
                with st.spinner("Buscando en documentos..."):
                    try:
                        response = rag_system.ask(prompt)
                        
                        show_rag_response(response)
                        
                        # Guardar respuesta en historial
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "response": response
                        })
                    except Exception as e:
                        st.error(f"❌ Error generando respuesta: {str(e)}")
    
    with col2:
        st.header("📚 Documentos")
        
        # Botón para limpiar historial
        if st.button("🗑️ Limpiar Chat", key="clear_chat_btn"):
            st.session_state.messages = []
            st.rerun()  # Reemplaza st.experimental_rerun()
        
        # Mostrar documentos
        show_documents(rag_system)

if __name__ == "__main__":
    main()