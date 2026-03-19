import os
import logging
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Flask App ----------
app = Flask(__name__)
CORS(app)

# ---------- Gemini API ----------
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# ✅ IMPORTANT: Lazy loading placeholders
embedding_model = None
reranker = None
vector_store = None
retriever = None

# ---------- Rerank ----------
def rerank(query: str, docs, top_k: int = 3):
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    filtered = [r for r in ranked if r[0] > 0.3]
    if not filtered:
        filtered = ranked[:2]

    return [doc for _, doc in filtered[:top_k]]

# ---------- Gemini LLM ----------
def generate_answer(query: str, context: str) -> str:
    try:
        prompt = f"""
You are a professional medical assistant.

Answer ONLY from the given context.
If the answer is not in the context, say:
"I couldn't find information regarding this in our medical database."

Context:
{context}

Question:
{query}

Answer in 1-3 sentences.
"""
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Error while processing request."

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        global embedding_model, reranker, vector_store, retriever

        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "No query provided"}), 400

        logger.info(f"Query: {query}")

        # ✅ Lazy load models
        if embedding_model is None:
            logger.info("Loading embedding model...")
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                encode_kwargs={"normalize_embeddings": True},
            )

        if reranker is None:
            logger.info("Loading reranker...")
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        if vector_store is None:
            logger.info("Loading documents...")
            FILE_PATH = "medical"

            if not os.path.exists(FILE_PATH):
                os.makedirs(FILE_PATH)

            loader = PyPDFDirectoryLoader(FILE_PATH)
            docs = loader.load()

            if not docs:
                return jsonify({
                    "query": query,
                    "answer": "No PDFs found in medical folder.",
                    "sources": []
                })

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(docs)

            vector_store = FAISS.from_documents(split_docs, embedding_model)
            retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # Retrieve
        initial_docs = retriever.invoke(query)

        # Rerank
        final_docs = rerank(query, initial_docs)

        if not final_docs:
            return jsonify({
                "query": query,
                "answer": "No relevant info found.",
                "sources": []
            })

        context_text = "\n\n".join(doc.page_content for doc in final_docs)
        answer = generate_answer(query, context_text)

        return jsonify({
            "query": query,
            "answer": answer
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)