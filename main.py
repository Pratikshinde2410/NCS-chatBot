# Level 2: AI Chat with Document Processing & Enhanced Memory
# Advanced implementation with document upload, processing, and vector search

import os
import sqlite3
import json
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import io
import base64

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import uvicorn

# Document processing imports
import PyPDF2
from docx import Document as DocxDocument
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI(title="Level 2 AI Chat", description="AI Chat with Document Processing & Memory")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
DB_PATH = "chat_memory.db"
UPLOADS_DIR = "uploads"
CHROMA_PATH = "document_vectors"

# Create necessary directories
Path(UPLOADS_DIR).mkdir(exist_ok=True)
Path(CHROMA_PATH).mkdir(exist_ok=True)

# Initialize embedding model and vector database
print("🔄 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
print("✅ Embedding model loaded successfully")

# Pydantic Models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    use_documents: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    session_id: str
    sources_used: Optional[List[str]] = []

class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    size: int
    chunks: int
    upload_date: str

class ConversationHistory(BaseModel):
    user_message: str
    ai_response: str
    timestamp: str
    sources_used: Optional[List[str]] = []

# Database Setup
def init_database():
    """Initialize SQLite database with document tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Conversations table (enhanced)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            sources_used TEXT,  -- JSON array of source documents
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            chunks_count INTEGER NOT NULL,
            upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            processed_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

# Document Processing Classes
class DocumentProcessor:
    def __init__(self):
        self.collection_name = "documents"
        try:
            self.collection = chroma_client.get_collection(self.collection_name)
            print("✅ Connected to existing document collection")
        except:
            self.collection = chroma_client.create_collection(self.collection_name)
            print("✅ Created new document collection")
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_file = io.BytesIO(file_content)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as e:
                    print(f"⚠️ Error extracting page {page_num + 1}: {e}")
                    continue
            
            return text.strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")
    
    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            docx_file = io.BytesIO(file_content)
            doc = DocxDocument(docx_file)
            
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            
            return "\n".join(text_parts)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"DOCX extraction failed: {str(e)}")
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extract text from TXT file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    return file_content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, use utf-8 with error handling
            return file_content.decode('utf-8', errors='replace')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"TXT extraction failed: {str(e)}")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        chunks = []
        text = text.strip()
        
        # Split by sentences first for better chunk boundaries
        sentences = text.replace('\n', ' ').split('. ')
        
        current_chunk = ""
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, start new chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk += sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_document(self, file_content: bytes, filename: str, original_filename: str) -> Dict[str, Any]:
        """Process document and store in vector database"""
        
        # Calculate file hash for deduplication
        file_hash = hashlib.md5(file_content).hexdigest()
        file_type = filename.split('.')[-1].lower()
        
        # Check if document already exists
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT filename, chunks_count FROM documents WHERE file_hash = ?", (file_hash,))
        existing = cursor.fetchone()
        conn.close()
        
        if existing:
            return {
                "message": f"Document already exists as '{existing[0]}' with {existing[1]} chunks",
                "filename": existing[0],
                "chunks": existing[1],
                "status": "duplicate"
            }
        
        # Extract text based on file type
        if file_type == "pdf":
            text = self.extract_text_from_pdf(file_content)
        elif file_type == "docx":
            text = self.extract_text_from_docx(file_content)
        elif file_type == "txt":
            text = self.extract_text_from_txt(file_content)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in document")
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Document could not be chunked properly")
        
        print(f"📄 Processing {filename}: {len(chunks)} chunks")
        
        # Generate embeddings and store in ChromaDB
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only process non-empty chunks
                try:
                    embedding = embedding_model.encode(chunk).tolist()
                    embeddings.append(embedding)
                    documents.append(chunk)
                    metadatas.append({
                        "filename": filename,
                        "original_filename": original_filename,
                        "chunk_id": i,
                        "file_type": file_type,
                        "total_chunks": len(chunks)
                    })
                    ids.append(f"{filename}_{i}")
                except Exception as e:
                    print(f"⚠️ Error processing chunk {i}: {e}")
                    continue
        
        if not embeddings:
            raise HTTPException(status_code=400, detail="Could not generate embeddings for document")
        
        # Store in ChromaDB
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vector database error: {str(e)}")
        
        # Store document info in SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents 
            (filename, original_filename, file_type, file_size, file_hash, chunks_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (filename, original_filename, file_type, len(file_content), file_hash, len(embeddings)))
        conn.commit()
        conn.close()
        
        print(f"✅ Document processed: {len(embeddings)} chunks stored")
        
        return {
            "message": f"Document processed successfully",
            "filename": filename,
            "chunks": len(embeddings),
            "status": "success"
        }
    
    def search_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for relevant document chunks"""
        try:
            # Generate query embedding
            query_embedding = embedding_model.encode(query).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {"chunks": [], "sources": []}
            
            # Process results
            chunks = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            # Get unique source filenames
            sources = list(set([meta['original_filename'] for meta in metadatas]))
            
            # Format results with relevance info
            formatted_chunks = []
            for chunk, meta, distance in zip(chunks, metadatas, distances):
                formatted_chunks.append({
                    "text": chunk,
                    "source": meta['original_filename'],
                    "chunk_id": meta['chunk_id'],
                    "relevance": 1 - distance  # Convert distance to relevance score
                })
            
            return {
                "chunks": formatted_chunks,
                "sources": sources
            }
            
        except Exception as e:
            print(f"❌ Document search error: {e}")
            return {"chunks": [], "sources": []}
    
    def get_all_documents(self) -> List[DocumentInfo]:
        """Get list of all processed documents"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT original_filename, file_type, file_size, chunks_count, upload_date
            FROM documents ORDER BY upload_date DESC
        """)
        
        documents = []
        for row in cursor.fetchall():
            documents.append(DocumentInfo(
                filename=row[0],
                file_type=row[1],
                size=row[2],
                chunks=row[3],
                upload_date=row[4]
            ))
        
        conn.close()
        return documents

# Enhanced Memory Manager
class EnhancedConversationMemory:
    def __init__(self):
        init_database()
    
    def create_session(self, session_id: str):
        """Create or update a session"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (session_id, last_activity) 
            VALUES (?, ?)
        """, (session_id, datetime.now()))
        conn.commit()
        conn.close()
    
    def store_conversation(self, session_id: str, user_message: str, ai_response: str, sources_used: List[str] = None):
        """Store conversation with document sources"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        sources_json = json.dumps(sources_used) if sources_used else None
        
        cursor.execute("""
            INSERT INTO conversations (session_id, user_message, ai_response, sources_used, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, user_message, ai_response, sources_json, datetime.now()))
        
        cursor.execute("""
            UPDATE sessions SET last_activity = ? WHERE session_id = ?
        """, (datetime.now(), session_id))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[ConversationHistory]:
        """Get enhanced conversation history with sources"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_message, ai_response, timestamp, sources_used
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in reversed(rows):
            sources = json.loads(row[3]) if row[3] else []
            history.append(ConversationHistory(
                user_message=row[0],
                ai_response=row[1],
                timestamp=row[2],
                sources_used=sources
            ))
        
        return history
    
    def get_recent_context(self, session_id: str, limit: int = 5) -> str:
        """Get recent conversation context"""
        history = self.get_conversation_history(session_id, limit)
        
        context_parts = []
        for conv in history[-limit:]:
            context_parts.append(f"Human: {conv.user_message}")
            context_parts.append(f"Assistant: {conv.ai_response}")
        
        return "\n".join(context_parts) if context_parts else ""

# Initialize components
doc_processor = DocumentProcessor()
memory = EnhancedConversationMemory()

# Enhanced AI Service
class EnhancedAIService:
    def __init__(self):
        self.model_name = "llama3.1:8b"
    
    def check_ollama_connection(self) -> bool:
        """Check Ollama connection"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response_with_documents(self, prompt: str, context: str = "", document_chunks: List[Dict] = None) -> str:
        """Generate response with document context"""
        
        # Build comprehensive prompt
        prompt_parts = ["You are a helpful AI assistant."]
        
        if document_chunks:
            prompt_parts.append("Here are some relevant document excerpts to help answer the question:")
            for i, chunk in enumerate(document_chunks, 1):
                relevance = chunk.get('relevance', 0)
                source = chunk.get('source', 'Unknown')
                text = chunk.get('text', '')
                prompt_parts.append(f"\n[Document {i}] (Source: {source}, Relevance: {relevance:.2f})")
                prompt_parts.append(text[:800] + "..." if len(text) > 800 else text)
        
        if context.strip():
            prompt_parts.append(f"\nRecent conversation context:\n{context}")
        
        prompt_parts.append(f"\nCurrent question: {prompt}")
        prompt_parts.append("\nPlease provide a helpful response based on the available information. If you reference document content, mention the source:")
        
        full_prompt = "\n".join(prompt_parts)
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 600  # Allow longer responses for document-based answers
            }
        }
        
        try:
            print(f"🤖 Generating response with document context...")
            response = requests.post(OLLAMA_URL, json=payload, timeout=90)
            response.raise_for_status()
            
            result = response.json()
            ai_response = result.get("response", "Sorry, I couldn't generate a response.")
            
            return ai_response.strip()
            
        except requests.exceptions.Timeout:
            return "⏰ Request timed out. The document context might be too large. Please try a more specific question."
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

# Initialize enhanced AI service
ai_service = EnhancedAIService()

# Enhanced Web Interface
@app.get("/")
async def root():
    """Serve enhanced chat interface with document upload"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Level 2 AI Chat with Documents</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
                margin: 0; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 1000px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
                display: grid;
                grid-template-columns: 1fr 300px;
                grid-template-rows: auto auto 1fr auto;
                height: 90vh;
            }
            .header {
                background: linear-gradient(135deg, #2c3e50, #3498db);
                color: white;
                padding: 20px;
                text-align: center;
                grid-column: 1 / -1;
            }
            .status {
                margin: 20px;
                padding: 10px;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                grid-column: 1 / -1;
            }
            .status.online { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.offline { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .main-chat {
                display: flex;
                flex-direction: column;
                border-right: 1px solid #eee;
            }
            .sidebar {
                background: #f8f9fa;
                padding: 20px;
                overflow-y: auto;
            }
            .chat-container { 
                flex: 1;
                overflow-y: auto; 
                padding: 20px;
                border-bottom: 1px solid #eee;
            }
            .message { 
                margin: 15px 0; 
                padding: 12px 16px; 
                border-radius: 20px; 
                max-width: 80%;
                word-wrap: break-word;
            }
            .user { 
                background: linear-gradient(135deg, #667eea, #764ba2); 
                color: white; 
                margin-left: auto; 
                text-align: right;
            }
            .ai { 
                background: #f1f3f4; 
                color: #333;
                border: 1px solid #e0e0e0;
            }
            .sources {
                font-size: 12px;
                color: #666;
                margin-top: 8px;
                border-top: 1px solid #ddd;
                padding-top: 8px;
            }
            .input-area { 
                padding: 20px; 
                display: flex; 
                gap: 10px;
                background: #f8f9fa;
            }
            #messageInput { 
                flex: 1; 
                padding: 12px 16px; 
                border: 2px solid #ddd; 
                border-radius: 25px; 
                font-size: 16px;
                outline: none;
                transition: border-color 0.3s;
            }
            #messageInput:focus {
                border-color: #667eea;
            }
            .send-btn { 
                padding: 12px 24px; 
                background: linear-gradient(135deg, #667eea, #764ba2); 
                color: white; 
                border: none; 
                border-radius: 25px; 
                cursor: pointer;
                font-weight: bold;
                transition: transform 0.2s;
            }
            .send-btn:hover { transform: translateY(-2px); }
            .send-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
            
            .upload-section {
                margin-bottom: 30px;
                padding: 15px;
                border: 2px dashed #ddd;
                border-radius: 10px;
                text-align: center;
                background: white;
            }
            .upload-section.dragover {
                border-color: #667eea;
                background: #f0f4ff;
            }
            .file-input {
                margin: 10px 0;
            }
            .upload-btn {
                background: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                cursor: pointer;
            }
            .documents-list {
                margin-top: 20px;
            }
            .document-item {
                background: white;
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
                font-size: 14px;
            }
            .loading { display: none; text-align: center; color: #666; font-style: italic; margin: 10px 0; }
            .timestamp { font-size: 11px; color: #999; margin-top: 5px; }
            
            .toggle-docs {
                margin-bottom: 10px;
            }
            .toggle-docs input[type="checkbox"] {
                margin-right: 8px;
            }
            
            @media (max-width: 768px) {
                .container {
                    grid-template-columns: 1fr;
                    grid-template-rows: auto auto 1fr auto auto;
                    height: 95vh;
                }
                .sidebar {
                    border-right: none;
                    border-top: 1px solid #eee;
                    max-height: 200px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖📚 Level 2 AI Chat</h1>
                <p>Chat with AI + Upload & Query Documents</p>
            </div>
            
            <div id="status" class="status">Checking system status...</div>
            
            <div class="main-chat">
                <div class="chat-container" id="chatContainer">
                    <div class="message ai">
                        <div>👋 Hello! I'm your enhanced AI assistant. I can now:</div>
                        <div>💬 Remember our conversations</div>
                        <div>📄 Read and analyze your documents (PDF, DOCX, TXT)</div>
                        <div>❓ Answer questions based on your uploaded files</div>
                        <div>🔍 Search through document content contextually</div>
                        <div class="timestamp" id="startTime"></div>
                    </div>
                </div>
                
                <div class="loading" id="loading">🤖 AI is thinking...</div>
                
                <div class="toggle-docs">
                    <label>
                        <input type="checkbox" id="useDocuments" checked>
                        Include documents in responses
                    </label>
                </div>
                
                <div class="input-area">
                    <input 
                        type="text" 
                        id="messageInput" 
                        placeholder="Ask me anything or questions about your documents..." 
                        disabled
                    >
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()" disabled>
                        Send
                    </button>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="upload-section" id="uploadSection">
                    <h3>📎 Upload Documents</h3>
                    <p>Drag & drop or click to upload<br>PDF, DOCX, TXT files</p>
                    <input type="file" id="fileInput" class="file-input" accept=".pdf,.docx,.txt" style="display:none">
                    <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                        Choose Files
                    </button>
                    <div id="uploadStatus"></div>
                </div>
                
                <div class="documents-list">
                    <h4>📚 Your Documents</h4>
                    <div id="documentsList">Loading...</div>
                </div>
            </div>
        </div>

        <script>
            let sessionId = 'session_' + Date.now();
            let isSystemOnline = false;

            // Set start timestamp
            document.getElementById('startTime').textContent = new Date().toLocaleString();

            // Check system status on load
            checkSystemStatus();
            loadDocuments();

            // Set up file upload
            setupFileUpload();

            // Enable Enter key
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey && isSystemOnline) {
                    sendMessage();
                }
            });

            async function checkSystemStatus() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    
                    const statusDiv = document.getElementById('status');
                    const messageInput = document.getElementById('messageInput');
                    const sendBtn = document.getElementById('sendBtn');
                    
                    if (data.ai_online) {
                        statusDiv.className = 'status online';
                        statusDiv.innerHTML = `✅ System Online | Model: ${data.model} | Conversations: ${data.total_conversations} | Documents: ${data.total_documents}`;
                        messageInput.disabled = false;
                        sendBtn.disabled = false;
                        messageInput.focus();
                        isSystemOnline = true;
                    } else {
                        statusDiv.className = 'status offline';
                        statusDiv.innerHTML = '❌ AI Offline - Please start Ollama: <code>ollama serve</code>';
                        isSystemOnline = false;
                    }
                } catch (error) {
                    console.error('Health check failed:', error);
                    document.getElementById('status').className = 'status offline';
                    document.getElementById('status').textContent = '❌ Cannot connect to server';
                }
            }

            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                const useDocuments = document.getElementById('useDocuments').checked;
                
                if (!message || !isSystemOnline) return;

                // Disable input during processing
                input.disabled = true;
                document.getElementById('sendBtn').disabled = true;
                document.getElementById('loading').style.display = 'block';

                // Add user message to chat
                addMessage(message, 'user');
                input.value = '';

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: message,
                            session_id: sessionId,
                            use_documents: useDocuments
                        })
                    });

                    const data = await response.json();
                    
                    if (response.ok) {
                        addMessage(data.response, 'ai', data.timestamp, data.sources_used);
                    } else {
                        addMessage(`Error: ${data.detail || 'Unknown error'}`, 'ai');
                    }
                } catch (error) {
                    console.error('Chat error:', error);
                    addMessage('Sorry, there was an error connecting to the AI service.', 'ai');
                }

                // Re-enable input
                input.disabled = false;
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('loading').style.display = 'none';
                input.focus();
            }

            function addMessage(text, sender, timestamp = null, sources = null) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                const messageText = document.createElement('div');
                messageText.textContent = text;
                messageDiv.appendChild(messageText);
                
                if (timestamp || sender === 'user') {
                    const timeDiv = document.createElement('div');
                    timeDiv.className = 'timestamp';
                    timeDiv.textContent = timestamp || new Date().toLocaleString();
                    messageDiv.appendChild(timeDiv);
                }
                
                if (sources && sources.length > 0) {
                    const sourcesDiv = document.createElement('div');
                    sourcesDiv.className = 'sources';
                    sourcesDiv.innerHTML = `<strong>Sources:</strong> ${sources.join(', ')}`;
                    messageDiv.appendChild(sourcesDiv);
                }
                
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            function setupFileUpload() {
                const uploadSection = document.getElementById('uploadSection');
                const fileInput = document.getElementById('fileInput');

                // Drag and drop handlers
                uploadSection.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    uploadSection.classList.add('dragover');
                });

                uploadSection.addEventListener('dragleave', function(e) {
                    e.preventDefault();
                    uploadSection.classList.remove('dragover');
                });

                uploadSection.addEventListener('drop', function(e) {
                    e.preventDefault();
                    uploadSection.classList.remove('dragover');
                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        uploadFiles(files);
                    }
                });

                // File input handler
                fileInput.addEventListener('change', function(e) {
                    if (e.target.files.length > 0) {
                        uploadFiles(e.target.files);
                    }
                });
            }

            async function uploadFiles(files) {
                const uploadStatus = document.getElementById('uploadStatus');
                
                for (let file of files) {
                    if (!['.pdf', '.docx', '.txt'].includes(file.name.toLowerCase().substring(file.name.lastIndexOf('.')))) {
                        uploadStatus.innerHTML = `<div style="color: red;">❌ ${file.name}: Unsupported file type</div>`;
                        continue;
                    }

                    uploadStatus.innerHTML = `<div style="color: blue;">📤 Uploading ${file.name}...</div>`;

                    const formData = new FormData();
                    formData.append('file', file);

                    try {
                        const response = await fetch('/upload', {
                            method: 'POST',
                            body: formData
                        });

                        const result = await response.json();
                        
                        if (response.ok) {
                            uploadStatus.innerHTML = `<div style="color: green;">✅ ${file.name}: ${result.message}</div>`;
                            loadDocuments(); // Refresh document list
                        } else {
                            uploadStatus.innerHTML = `<div style="color: red;">❌ ${file.name}: ${result.detail}</div>`;
                        }
                    } catch (error) {
                        uploadStatus.innerHTML = `<div style="color: red;">❌ ${file.name}: Upload failed</div>`;
                    }
                }
            }

            async function loadDocuments() {
                try {
                    const response = await fetch('/documents');
                    const data = await response.json();
                    
                    const documentsList = document.getElementById('documentsList');
                    
                    if (data.documents.length === 0) {
                        documentsList.innerHTML = '<div style="color: #666; font-style: italic;">No documents uploaded yet</div>';
                        return;
                    }
                    
                    documentsList.innerHTML = data.documents.map(doc => `
                        <div class="document-item">
                            <div style="font-weight: bold;">📄 ${doc.filename}</div>
                            <div style="font-size: 12px; color: #666;">
                                ${doc.file_type.toUpperCase()} • ${Math.round(doc.size/1024)}KB • ${doc.chunks} chunks
                            </div>
                            <div style="font-size: 11px; color: #999;">
                                ${new Date(doc.upload_date).toLocaleDateString()}
                            </div>
                        </div>
                    `).join('');
                } catch (error) {
                    console.error('Error loading documents:', error);
                    document.getElementById('documentsList').innerHTML = '<div style="color: red;">Error loading documents</div>';
                }
            }

            // Refresh status every 30 seconds
            setInterval(checkSystemStatus, 30000);
        </script>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Enhanced health check with document stats"""
    ai_online = ai_service.check_ollama_connection()
    
    # Get conversation and document counts
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversations")
    total_conversations = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM documents")
    total_documents = cursor.fetchone()[0]
    conn.close()
    
    return {
        "status": "healthy",
        "ai_online": ai_online,
        "model": ai_service.model_name,
        "total_conversations": total_conversations,
        "total_documents": total_documents,
        "available_models": ai_service.list_available_models() if ai_online else []
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    
    # Validate file type
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['pdf', 'docx', 'txt']:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF, DOCX, or TXT files.")
    
    # Read file content
    file_content = await file.read()
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    
    # Process document
    try:
        result = doc_processor.process_document(file_content, filename, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents():
    """Get list of all processed documents"""
    documents = doc_processor.get_all_documents()
    return {"documents": documents}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Enhanced chat endpoint with document search"""
    
    # Ensure session exists
    memory.create_session(message.session_id)
    
    # Get conversation context
    context = memory.get_recent_context(message.session_id, limit=5)
    
    print(f"💬 New message in session {message.session_id}: {message.message[:50]}...")
    
    # Search documents if enabled
    document_chunks = []
    sources_used = []
    
    if message.use_documents:
        doc_results = doc_processor.search_documents(message.message, n_results=5)
        document_chunks = doc_results.get('chunks', [])
        sources_used = doc_results.get('sources', [])
        
        if document_chunks:
            print(f"🔍 Found {len(document_chunks)} relevant document chunks from {len(sources_used)} sources")
    
    # Generate AI response with document context
    ai_response = ai_service.generate_response_with_documents(
        message.message, 
        context, 
        document_chunks
    )
    
    # Store conversation with sources
    memory.store_conversation(message.session_id, message.message, ai_response, sources_used)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"🤖 Response generated: {ai_response[:50]}...")
    
    return ChatResponse(
        response=ai_response,
        timestamp=timestamp,
        session_id=message.session_id,
        sources_used=sources_used
    )

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Get conversation history with sources"""
    history = memory.get_conversation_history(session_id)
    return {
        "session_id": session_id,
        "history": history,
        "total_messages": len(history)
    }

@app.get("/sessions")
async def get_sessions():
    """Get all chat sessions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.session_id, s.created_at, s.last_activity,
               COUNT(c.id) as message_count
        FROM sessions s
        LEFT JOIN conversations c ON s.session_id = c.session_id
        GROUP BY s.session_id, s.created_at, s.last_activity
        ORDER BY s.last_activity DESC
    """)
    
    sessions = []
    for row in cursor.fetchall():
        sessions.append({
            "session_id": row[0],
            "created_at": row[1],
            "last_activity": row[2],
            "message_count": row[3]
        })
    
    conn.close()
    return {"sessions": sessions}

@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    rows_deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    return {"message": f"Cleared {rows_deleted} messages from session {session_id}"}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and its vector embeddings"""
    # Remove from SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, chunks_count FROM documents WHERE filename = ?", (filename,))
    doc_info = cursor.fetchone()
    
    if not doc_info:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    
    cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()
    
    # Remove from ChromaDB (this is a simplified approach)
    # In a production system, you'd want to store the chunk IDs for proper deletion
    print(f"⚠️ Note: ChromaDB entries for {filename} may still exist. Full cleanup requires ChromaDB API.")
    
    return {"message": f"Document {filename} deleted from database"}

if __name__ == "__main__":
    print("🚀 Starting Level 2 AI Chat Application...")
    print("📊 Features: Enhanced Chat + Document Processing + Vector Search")
    print("🔗 URL: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info"
    )
