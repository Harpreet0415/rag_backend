import os
import uuid
import shutil
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pdf_processor import extract_text_from_pdf, chunk_text
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

app = Flask(__name__)

# CORS Configuration for Production
# Allow your deployed frontend and local development
CORS(app, 
     resources={
         r"/api/*": {
             "origins": [
                 "https://rag-frontend.vercel.app",      # Your production frontend
                 "https://rag-frontend-git-main.vercel.app",  # Preview deployments
                 "http://localhost:5173",                 # Local Vite dev server
                 "http://localhost:3000",                 # Alternative local port
                 "https://rag-frontend-harpreet0415.vercel.app"  # Any other Vercel URLs
             ],
             "allow_headers": ["Content-Type", "X-Gemini-Key", "Authorization"],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "supports_credentials": True
         }
     })

# Alternative: More flexible configuration for testing
# Uncomment this if the above doesn't work
# CORS(app, resources={r"/api/*": {"origins": "*"}}, allow_headers=["Content-Type", "X-Gemini-Key"])

UPLOAD_FOLDER = "uploads"
INDEX_FOLDER = "indexes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# In-memory sessions: session_id -> { vector_store, rag_pipeline, doc_info }
sessions = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def get_api_key():
    """Read Gemini key from request header, then env var."""
    return request.headers.get("X-Gemini-Key", GEMINI_API_KEY).strip()


def get_or_create_session(session_id: str, api_key: str = ""):
    if session_id not in sessions:
        vs = VectorStore()
        rp = RAGPipeline(vs, api_key or GEMINI_API_KEY)
        sessions[session_id] = {
            "vector_store": vs,
            "rag_pipeline": rp,
            "doc_info": None
        }
    return sessions[session_id]


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "RAG Server is running"})


@app.route("/api/upload", methods=["POST", "OPTIONS"])
def upload_pdf():
    """Upload and process a PDF file."""
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return '', 200
    
    print(f"Received upload request. Files: {request.files}")
    if "file" not in request.files:
        print("Error: No file provided")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    print(f"Filename: {file.filename}")
    if not file.filename.endswith(".pdf"):
        print("Error: Invalid file type")
        return jsonify({"error": "Only PDF files are supported"}), 400

    api_key = get_api_key()
    if not api_key:
        print("Error: No API Key")
        return jsonify({"error": "Gemini API key not provided. Include it in the X-Gemini-Key header or set GEMINI_API_KEY env var."}), 400

    session_id = request.form.get("session_id", str(uuid.uuid4()))
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
    file.save(pdf_path)
    print(f"File saved to {pdf_path}")

    try:
        # Extract and chunk
        print("Starting PDF extraction...")
        doc_data = extract_text_from_pdf(pdf_path)
        print(f"Extraction complete. Pages: {doc_data['num_pages']}")
        chunks = chunk_text(doc_data["pages"])
        print(f"Chunking complete. Chunks: {len(chunks)}")
        
        if not chunks:
            print("Error: No chunks generated")
            return jsonify({"error": "Could not extract text from PDF"}), 422

        # Build vector index
        print("Building vector index...")
        sess = get_or_create_session(session_id, api_key)
        sess["vector_store"].clear()
        sess["rag_pipeline"].clear_history()
        sess["vector_store"].build_index(chunks)
        print("Vector index built.")
        sess["doc_info"] = {
            "filename": filename,
            "num_pages": doc_data["num_pages"],
            "num_chunks": len(chunks)
        }

        # Generate summary
        print("Generating summary...")
        summary = sess["rag_pipeline"].summarize()
        print("Summary generated.")

        return jsonify({
            "session_id": session_id,
            "filename": filename,
            "num_pages": doc_data["num_pages"],
            "num_chunks": len(chunks),
            "summary": summary
        })

    except Exception as e:
        print(f"EXCEPTION DURING UPLOAD: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print("Cleanup: PDF file removed")


@app.route("/api/ask", methods=["POST", "OPTIONS"])
def ask_question():
    """Answer a question using the RAG pipeline."""
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return '', 200
    
    data = request.get_json()
    session_id = data.get("session_id")
    question = data.get("question", "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid or expired session. Please upload a document first."}), 400
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    sess = sessions[session_id]
    try:
        result = sess["rag_pipeline"].answer(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET", "OPTIONS"])
def get_history():
    """Get chat history for a session."""
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return '', 200
    
    session_id = request.args.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"history": []})
    
    history = sessions[session_id]["rag_pipeline"].chat_history
    return jsonify({"history": history})


@app.route("/api/clear_history", methods=["POST", "OPTIONS"])
def clear_history():
    """Clear chat history for a session."""
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return '', 200
    
    data = request.get_json()
    session_id = data.get("session_id")
    if session_id and session_id in sessions:
        sessions[session_id]["rag_pipeline"].clear_history()
    return jsonify({"status": "cleared"})


@app.route("/api/session_info", methods=["GET", "OPTIONS"])
def session_info():
    """Get document info for a session."""
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return '', 200
    
    session_id = request.args.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"doc_info": None})
    return jsonify({"doc_info": sessions[session_id]["doc_info"]})


@app.route("/", methods=["GET"])
def root():
    """Root endpoint for health check."""
    return jsonify({
        "status": "online",
        "service": "RAG Backend",
        "version": "1.0.0",
        "endpoints": [
            "/api/health",
            "/api/upload",
            "/api/ask",
            "/api/history",
            "/api/clear_history",
            "/api/session_info"
        ]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)