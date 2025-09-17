# 🤖📚 Level 2 AI Chat with Document Processing

An advanced AI chat application that combines conversational AI with document processing capabilities. Upload documents (PDF, DOCX, TXT) and have contextual conversations with AI that can reference and analyze your uploaded content.

## ✨ Features

### 🧠 Enhanced AI Chat
- **Conversation Memory**: AI remembers previous conversations in each session
- **Context-Aware Responses**: Uses recent conversation history for better responses
- **Session Management**: Multiple chat sessions with unique IDs
- **Real-time Status**: Shows AI connection status and system health

### 📄 Document Processing
- **Multi-format Support**: Upload PDF, DOCX, and TXT files
- **Drag & Drop Interface**: Modern file upload with visual feedback
- **Text Extraction**: Automatic text extraction from all supported formats
- **Smart Chunking**: Intelligent text segmentation with overlap for context preservation
- **Duplicate Detection**: Prevents processing duplicate files using content hashing

### 🔍 Vector Search & Semantic Understanding
- **ChromaDB Integration**: Persistent vector database for document storage
- **Semantic Search**: Find relevant content using meaning, not just keywords
- **Relevance Scoring**: Ranked search results with relevance scores
- **Contextual Responses**: AI responses include relevant document excerpts
- **Source Attribution**: Every response shows which documents were referenced

### 🎨 Modern User Interface
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Two-Panel Layout**: Chat interface + document management panel
- **Real-time Updates**: Live status updates and document list refresh
- **Toggle Controls**: Enable/disable document inclusion in responses
- **Beautiful Styling**: Modern gradient design with smooth animations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   FastAPI       │    │   AI Services   │
│                 │◄──►│   Backend       │◄──►│                 │
│ • Chat Interface│    │                 │    │ • Ollama LLM    │
│ • File Upload   │    │ • REST API      │    │ • Embeddings    │
│ • Document List │    │ • WebSocket     │    │ • Vector Search │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Data Storage  │
                       │                 │
                       │ • SQLite DB     │
                       │ • ChromaDB      │
                       │ • File Storage  │
                       └─────────────────┘
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended for document processing)
- **Storage**: At least 2GB free space for models and databases
- **Operating System**: Linux, macOS, or Windows

### Required Software
- **Ollama**: For running local LLM models
- **Git**: For cloning the repository (optional)

## 🚀 Installation Guide

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd level2-ai-chat

# Or simply download and extract the files to a directory
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Ollama

#### Linux/macOS:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from: https://ollama.ai/download
```

#### Windows:
1. Download Ollama from: https://ollama.ai/download
2. Run the installer
3. Add Ollama to your PATH

### Step 4: Download AI Model

```bash
# Start Ollama service
ollama serve

# In another terminal, pull a model (choose one):
ollama pull llama3.1:8b        # Recommended (8B parameters)
ollama pull llama3.1:70b       # Larger, more capable (70B parameters)
ollama pull llama2:7b          # Alternative option
ollama pull mistral:7b         # Fast alternative
```

### Step 5: Install Python Dependencies

```bash
# Make sure virtual environment is activated
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

### Step 6: Verify Installation

```bash
# Check if all packages installed correctly
python -c "import fastapi, chromadb, sentence_transformers, PyPDF2; print('✅ All packages installed successfully')"

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

## 🎯 Running the Application

### Start the Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Start the application
python main.py
```

### Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 Usage Guide

### 1. First Launch
1. Open your browser to `http://localhost:8000`
2. Wait for the system status to show "✅ System Online"
3. If status shows "❌ AI Offline", ensure Ollama is running: `ollama serve`

### 2. Uploading Documents
1. **Drag & Drop**: Drag files directly onto the upload area
2. **Click Upload**: Click "Choose Files" and select your documents
3. **Supported Formats**: PDF, DOCX, TXT files
4. **Processing**: Wait for "✅ Document processed successfully" message

### 3. Chatting with AI
1. **Basic Chat**: Type messages and press Enter or click Send
2. **Document-Enhanced**: Ensure "Include documents in responses" is checked
3. **Ask Questions**: Ask questions about your uploaded documents
4. **View Sources**: See which documents were referenced in AI responses

### 4. Document Management
- **View All Documents**: See list of uploaded files in the sidebar
- **Document Info**: View file type, size, and chunk count
- **Delete Documents**: Use the API endpoint `/documents/{filename}`

## 🔧 Configuration

### Environment Variables
You can customize the application by setting these environment variables:

```bash
export OLLAMA_URL="http://localhost:11434/api/generate"  # Ollama API URL
export DB_PATH="chat_memory.db"                          # SQLite database path
export UPLOADS_DIR="uploads"                             # Upload directory
export CHROMA_PATH="document_vectors"                    # Vector database path
export MODEL_NAME="llama3.1:8b"                         # Default AI model
```

### Model Configuration
Edit the model name in `main.py`:

```python
class EnhancedAIService:
    def __init__(self):
        self.model_name = "llama3.1:8b"  # Change this to your preferred model
```

## 📚 API Reference

### Chat Endpoints
- `POST /chat` - Send message and get AI response
- `GET /health` - System health and statistics
- `GET /history/{session_id}` - Get conversation history
- `GET /sessions` - List all chat sessions
- `DELETE /history/{session_id}` - Clear session history

### Document Endpoints
- `POST /upload` - Upload and process document
- `GET /documents` - List all processed documents
- `DELETE /documents/{filename}` - Delete specific document

### Example API Usage

```bash
# Send a chat message
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is in my uploaded documents?", "session_id": "test_session"}'

# Upload a document
curl -X POST "http://localhost:8000/upload" \
     -F "file=@document.pdf"

# Get system health
curl "http://localhost:8000/health"
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "AI Offline" Status
```bash
# Solution: Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

#### 2. Import Errors
```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt

# For specific packages
pip install --upgrade chromadb sentence-transformers
```

#### 3. Out of Memory Errors
- Use a smaller model: `ollama pull llama2:7b`
- Reduce chunk size in the code
- Close other applications to free memory

#### 4. Document Processing Fails
- Check file format is supported (PDF, DOCX, TXT)
- Ensure file is not corrupted
- Try with a smaller file first

#### 5. ChromaDB Errors
```bash
# Clear vector database (will delete all documents)
rm -rf document_vectors/
# Restart the application
```

### Performance Optimization

#### For Better Speed:
- Use smaller models (7B instead of 70B)
- Reduce document chunk size
- Limit number of search results

#### For Better Quality:
- Use larger models (70B parameters)
- Increase chunk overlap
- Upload more relevant documents

## 🔒 Security Considerations

### Data Privacy
- All data stays on your local machine
- No data is sent to external services (except Ollama API)
- Documents are stored locally in the `uploads/` directory

### Network Security
- Application runs on localhost by default
- Change host to `0.0.0.0` only in trusted networks
- Use HTTPS in production environments

### File Security
- Validate file types before processing
- Set file size limits to prevent DoS attacks
- Regularly clean up uploaded files

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings for functions and classes
- Keep functions small and focused

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** - For providing easy-to-use local LLM inference
- **FastAPI** - For the excellent web framework
- **ChromaDB** - For vector database capabilities
- **Sentence Transformers** - For high-quality text embeddings
- **Hugging Face** - For the embedding models

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Check the application logs for error messages
4. Open an issue with detailed error information

## 🔄 Version History

- **v2.0.0** - Level 2: Added document processing and vector search
- **v1.0.0** - Level 1: Basic AI chat with conversation memory

---

**Happy Chatting! 🚀**
