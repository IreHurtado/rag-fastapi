📘 RAG con FastAPI y OpenAI

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) utilizando FAISS para indexar documentos, OpenAI para generar embeddings y respuestas, y FastAPI como framework backend.
La aplicación incluye un frontend básico en HTML + TailwindCSS para interactuar con la API.

🚀 Funcionalidades

Indexación de documentos en chunks y almacenamiento en FAISS.

Recuperación de los Top-K fragmentos más relevantes para cada consulta.

Generación de respuestas basadas exclusivamente en el contexto recuperado.

API REST con FastAPI.

Frontend sencillo en HTML/Tailwind para hacer consultas.

Guard-rails básicos:

Límite de peticiones por IP.

Umbral de similitud para evitar respuestas fuera de contexto.

🛠️ Requisitos

Python 3.10+

Clave de API de OpenAI (OPENAI_API_KEY)

📦 Instalación

Clona el repositorio y entra en la carpeta:

git clone https://github.com/TU-USUARIO/rag-fastapi.git
cd rag-fastapi


Crea un entorno virtual e instala dependencias:

python -m venv .venv
source .venv/bin/activate   # en Linux/Mac
.venv\Scripts\activate      # en Windows

pip install -r requirements.txt


Copia el archivo .env.example a .env y añade tu clave de OpenAI:

OPENAI_API_KEY=tu_api_key

📑 Indexación de documentos

Coloca tus documentos en la carpeta docs/ y ejecuta:

python index.py --docs docs/ --out .indexes


Esto generará los embeddings y guardará los índices en la carpeta .indexes/.

▶️ Ejecutar el servidor

Levanta el servidor con uvicorn:

uvicorn app:app --reload


Por defecto, estará en:
👉 http://127.0.0.1:8000

El frontend estará disponible en /, y la API en /ask.

📂 Estructura del proyecto
rag-fastapi/
│── app.py          # API principal con FastAPI
│── query.py        # Consulta y generación de respuestas
│── index.py        # Script de indexación de documentos
│── utils.py        # Utilidades (normalización, chunks, etc.)
│── requirements.txt
│── .env.example
│── .gitignore
│── index.html      # Frontend con Tailwind
│── docs/           # Carpeta de documentos fuente
│── .indexes/       # Carpeta de índices FAISS (ignorada en Git)

🌍 Despliegue

Puedes desplegar en Render o Railway.
Asegúrate de que en producción se genere .indexes/ al ejecutar index.py tras subir tus documentos.

📜 Licencia

MIT License.
