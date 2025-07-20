
# 🧠 AgenteOpenEvidence: Sistema Médico RAG con Azure y Web Intelligence

`AgenteOpenEvidence` es un sistema de Recuperación Aumentada por Generación (RAG) diseñado para entornos clínicos y científicos. Combina búsqueda local y web, procesamiento de documentos médicos, y generación de respuestas basadas en evidencia con Azure OpenAI.

---

## 🚀 Funcionalidades principales

- 🔎 **Búsqueda híbrida**: 
  - Consulta documentos cargados localmente (`.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx`) usando FAISS.
  - Si no encuentra suficiente evidencia local, realiza búsqueda web médica vía Google CSE.
  
- 📚 **Procesamiento de documentos**:
  - Extrae texto estructurado desde múltiples formatos clínicos.
  - Analiza y resume tablas, estadísticas, y categorías en CSV y Excel.

- 🧩 **Chunking inteligente**:
  - Segmentación semántica de documentos según secciones como `INTRODUCTION`, `METHODS`, `RESULTS`, etc., con solapamiento configurable.

- 🌐 **Extracción web**:
  - Usa `trafilatura` y `selenium` para extraer el contenido completo de URLs científicas o médicas.
  
- 🤖 **Embeddings y recuperación**:
  - Generación de embeddings por lotes usando Azure OpenAI (`text-embedding-3-large`).
  - Recuperación basada en similitud con `FAISS` normalizado (IP + L2).

- 🩺 **Respuesta médica estructurada**:
  - Usa instrucciones especializadas para dar respuestas clínicas claras, con citación de fuentes y clasificación del nivel de evidencia.

- 📤 **Gestión de la base de datos**:
  - Agregar, eliminar, o limpiar documentos procesados dinámicamente.

---

## 🛠️ Tecnologías y librerías utilizadas

| Propósito                         | Herramienta                                   |
|----------------------------------|-----------------------------------------------|
| Generación de embeddings         | `Azure OpenAI (text-embedding-3-large)`       |
| FAISS para búsqueda semántica    | `faiss`                                       |
| Extracción web                   | `trafilatura`, `selenium`, `googleapiclient`  |
| Extracción de documentos         | `PyPDF2`, `docx`, `pandas`                    |
| Lógica RAG y razonamiento        | `Azure OpenAI Chat Completion`                |
| Manejo de entornos y secretos    | `python-dotenv`                               |
| Logging y trazabilidad           | `logging`                                     |

---

## ⚡ Rendimiento y consideraciones

- **Embeddings**:
  - Se procesan en lotes (`batch_size=100`), con sleeps mínimos para evitar throttling.
  - Se usa `normalize_L2()` para mejorar la similitud coseno en FAISS.

- **Búsqueda Web**:
  - Activada solo si hay <3 documentos relevantes con `similarity_score > 0.7`.
  - Incluye mecanismos de fallback (selenium si trafilatura falla).

---

## 📄 Variables de entorno requeridas (`.env`)

```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://artifacts-neu.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large

GOOGLE_SEARCH_API_KEY=...
GOOGLE_CSE_ID=...
```

---

## ✅ Posibles mejoras futuras

- Agrega whitelist real con dominios médicos.
- Post-procesamiento con citación automática estilo APA
- Puedes truncar o resumir contextos largos antes de enviarlos al LLM
- Añadir soporte futuro para imágenes médicas o tablas estructuradas (via OCR o CSV embedding)
- Añadir filtros temáticos (oncología, neumología, etc.)
