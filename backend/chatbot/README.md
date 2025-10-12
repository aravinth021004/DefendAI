# DefendAI Chatbot Setup

## Overview

The DefendAI chatbot has been optimized with a two-part architecture:

1. **Knowledge Base Manager** (`knowledge_base_manager.py`) - Creates and manages the vector store
2. **Optimized Chatbot Server** (`optimized_chatbot_server.py`) - Serves the chatbot API

## Quick Start

### 1. First Time Setup

```bash
cd backend/chatbot
python manage.py setup
```

### 2. Start Chatbot Server

```bash
cd backend/chatbot
python manage.py server
```

The chatbot server will run on `http://localhost:8000/chat`

### 3. Update Knowledge Base (only when training data changes)

```bash
cd backend/chatbot
python manage.py update
```

## Files Overview

### Required Files (Keep in Git)
- ✅ `chatbot_training_data.json` - Essential knowledge base content
- ✅ `knowledge_base_manager.py` - Vector store management
- ✅ `optimized_chatbot_server.py` - Chatbot server
- ✅ `manage.py` - Management script

### Generated Files (Excluded from Git)
- ❌ `defendai_vectorstore/` - Auto-generated vector database
- ❌ `data_hash.txt` - Hash tracking for data changes
- ❌ `__pycache__/` - Python cache files

## Configuration

### Environment Variables

Create a `.env` file in `backend/chatbot/`:

```env
# API Keys
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key  
TAVILY_API_KEY=your_tavily_api_key

# Server Configuration
DEFENDAI_HOST=localhost
DEFENDAI_PORT=8000
FLASK_DEBUG=false
```

### Frontend Configuration

In `frontend/.env`:

```env
REACT_APP_CHATBOT_URL=http://localhost:8000
```

## API Endpoints

The optimized chatbot server provides:

- `GET /` - Service information
- `GET /health` - Health check
- `POST /chat` - Main chatbot interaction
- `POST /search` - Knowledge base search
- `GET /config` - Server configuration
- `GET /vector-store-info` - Vector store status

## Usage Examples

### Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How does deepfake detection work?", "include_sources": true}'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Vector Store Not Found
```
⚠️ Vector store not found! Run 'python manage.py setup' first
```

**Solution:** Run the setup command to create the vector store.

### Embedding Model Issues
```
❌ HuggingFace embeddings failed
```

**Solution:** Install required packages:
```bash
pip install sentence-transformers torch
```

### API Connection Issues

1. Check if chatbot server is running on port 8000
2. Verify environment variables are set correctly
3. Check firewall/antivirus settings

## Performance Benefits

- ⚡ **Faster Startup**: ~2-3 seconds vs 30+ seconds
- 🧠 **Smart Updates**: Only re-embeds when data changes
- 💾 **Persistent Storage**: Vector store saved to disk
- 🔄 **Auto-Detection**: Hash-based change detection

## Development Workflow

1. **Make changes** to training data
2. **Update knowledge base**: `python manage.py update` 
3. **Restart chatbot server**: The server will use updated embeddings
4. **Test changes** in the frontend

## Architecture Diagram

```
Training Data (JSON) → Knowledge Base Manager → Vector Store (ChromaDB)
                                                      ↓
Frontend (React) → API Service → Optimized Server → RAG System → LLM Response
```

## Management Commands

```bash
# Check system status
python manage.py check

# Setup knowledge base (first time)
python manage.py setup

# Update knowledge base (when data changes)
python manage.py update [--force]

# Show vector store information  
python manage.py info

# Start optimized server
python manage.py server
```