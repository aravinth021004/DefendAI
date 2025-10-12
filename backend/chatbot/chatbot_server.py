"""
DefendAI Chatbot Server - Optimized Flask Version
A Flask server that provides REST API endpoints for the DefendAI RAG chatbot.
Uses pre-built vector store for optimal performance.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS

# Environment and configuration
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Configuration settings for the DefendAI chatbot server"""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Server settings
    HOST = os.getenv("DEFENDAI_HOST", "localhost")
    PORT = int(os.getenv("DEFENDAI_PORT", 8000))
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    # Model settings
    LLM_MODEL = "gemini-2.5-flash"  # More stable model for Flask
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector store settings
    CHROMA_PERSIST_DIR = "./defendai_vectorstore"
    COLLECTION_NAME = "deepfake_knowledge"
    
    # Search settings
    MAX_SEARCH_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.7

# Vector Store Loader
class VectorStoreLoader:
    """Loads existing vector store without recreating it"""
    
    def __init__(self):
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        self.retriever = None
        self._setup_complete = False
    
    def _initialize_embeddings(self):
        """Initialize embeddings with fallbacks"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            logger.info("✅ HuggingFace embeddings initialized")
            return embeddings
        except Exception as e:
            logger.error(f"❌ HuggingFace embeddings failed: {e}")
            # Fallback to Google embeddings if available
            if Config.GOOGLE_API_KEY:
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=Config.GOOGLE_API_KEY
                    )
                    logger.info("✅ Google Gemini embeddings initialized as fallback")
                    return embeddings
                except Exception as gemini_error:
                    logger.error(f"❌ Google embeddings also failed: {gemini_error}")
            
            raise ValueError("No embedding models available. Please install sentence-transformers or set GOOGLE_API_KEY")
    
    def load_vector_store(self):
        """Load existing vector store"""
        try:
            persist_dir = Path(Config.CHROMA_PERSIST_DIR)
            
            if not persist_dir.exists():
                raise FileNotFoundError(
                    f"Vector store not found at {persist_dir}. "
                    "Please run knowledge_base_manager.py first to create the vector store."
                )
            
            logger.info("📚 Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings,
                collection_name=Config.COLLECTION_NAME
            )
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            logger.info("🔍 Retriever created successfully!")
            self._setup_complete = True
            
        except Exception as e:
            logger.error(f"❌ Vector store loading failed: {e}")
            raise ValueError(f"Cannot load vector store: {e}")
    
    def search_knowledge_base(self, query: str, k: int = 3) -> List[Document]:
        """Search the knowledge base"""
        if not self._setup_complete or self.vector_store is None:
            logger.warning("⚠️ Vector store not available")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# Chatbot Tools
class ChatbotTools:
    """Manages LLM and search tools for the chatbot"""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.tavily_search = self._initialize_tavily_search()
    
    def _initialize_llm(self):
        """Initialize LLM with fallbacks"""
        if Config.GOOGLE_API_KEY:
            try:
                llm = ChatGoogleGenerativeAI(
                    google_api_key=Config.GOOGLE_API_KEY,
                    model=Config.LLM_MODEL,
                    temperature=0.1
                )
                logger.info(f"✅ Google {Config.LLM_MODEL} LLM initialized!")
                return llm
            except Exception as e:
                logger.error(f"❌ Google LLM failed: {e}")
                # Try fallback models
                try:
                    llm = ChatGoogleGenerativeAI(
                        google_api_key=Config.GOOGLE_API_KEY,
                        model="gemini-pro",
                        temperature=0.1
                    )
                    logger.info("✅ Google Gemini Pro LLM initialized as fallback!")
                    return llm
                except Exception as e2:
                    logger.error(f"❌ Fallback Google LLM also failed: {e2}")
        
        if Config.GROQ_API_KEY:
            try:
                llm = ChatGroq(
                    groq_api_key=Config.GROQ_API_KEY,
                    model_name="llama-3.1-8b-instant",
                    temperature=0.1
                )
                logger.info("✅ Groq LLM initialized as fallback!")
                return llm
            except Exception as e:
                logger.error(f"❌ Groq LLM also failed: {e}")
        
        raise ValueError("No LLM available. Please set GOOGLE_API_KEY or GROQ_API_KEY")
    
    def _initialize_tavily_search(self):
        """Initialize Tavily search tool"""
        if Config.TAVILY_API_KEY:
            try:
                return TavilySearchResults(
                    api_key=Config.TAVILY_API_KEY,
                    max_results=Config.MAX_SEARCH_RESULTS
                )
            except Exception as e:
                logger.warning(f"⚠️ Tavily search initialization failed: {e}")
        else:
            logger.warning("⚠️ Tavily API key not found. News search will be disabled.")
        return None
    
    def determine_search_intent(self, query: str) -> bool:
        """Determine if the query needs recent news/current events"""
        news_keywords = [
            "recent", "latest", "current", "news", "today", "this week", 
            "this month", "2024", "2025", "happening now", "breaking",
            "update", "development", "trend", "recently"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in news_keywords)

# RAG Chatbot
class RAGChatbot:
    """Main RAG chatbot implementation"""
    
    def __init__(self, vector_store_loader: VectorStoreLoader, tools: ChatbotTools):
        self.vs_loader = vector_store_loader
        self.tools = tools
        self.llm = tools.llm
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are DefendAI, an expert AI assistant specializing in deepfake detection and media authentication.

Your expertise includes:
- Deepfake detection techniques and algorithms
- Media forensics and authentication methods
- AI-generated content identification
- Digital media analysis tools
- Current trends in synthetic media

Guidelines:
1. Provide accurate, technical information about deepfake detection
2. Use the knowledge base to answer questions about techniques and methods
3. For recent news or current events, use the search results provided
4. Be helpful but acknowledge limitations
5. Recommend professional verification when stakes are high
6. Explain technical concepts clearly for different audience levels

Available context from knowledge base:
{context}

News context (if applicable):
{news_context}

Chat History:
{chat_history}

Current conversation:
{input}"""),
            ("human", "{input}")
        ])
        
        # Create retrieval chain
        def get_context(input_dict):
            query = input_dict.get("input", "") if isinstance(input_dict, dict) else str(input_dict)
            try:
                docs = self.vs_loader.retriever.invoke(query) if self.vs_loader.retriever else []
                return "\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                logger.error(f"⚠️ Retrieval error: {e}")
                return "No specific context available."
        
        self.retrieval_chain = (
            {
                "context": RunnableLambda(get_context),
                "news_context": lambda x: x.get("news_context", ""),
                "chat_history": lambda x: x.get("chat_history", ""),
                "input": lambda x: x.get("input", "") if isinstance(x, dict) else str(x)
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("🧠 DefendAI RAG Chatbot initialized!")
    
    def get_response(self, query: str, chat_history: str = "", include_sources: bool = False) -> tuple[str, str, List[Document]]:
        """Get response from RAG system with optional news search"""
        
        retrieved_docs = []
        news_context = ""
        
        # Get knowledge base context
        if self.vs_loader.retriever:
            try:
                retrieved_docs = self.vs_loader.search_knowledge_base(query, k=3)
            except Exception as e:
                logger.error(f"Knowledge base search error: {e}")
        
        # Check if query needs recent news
        if self.tools.determine_search_intent(query) and self.tools.tavily_search:
            try:
                search_results = self.tools.tavily_search.invoke(query)
                
                # Format search results
                news_context = "Recent News:\n"
                for i, result in enumerate(search_results[:3], 1):
                    news_context += f"{i}. {result['title']}\n"
                    news_context += f"   {result['content'][:200]}...\n"
                    news_context += f"   Source: {result['url']}\n\n"
                    
            except Exception as e:
                logger.error(f"⚠️ News search failed: {e}")
        
        # Generate response
        try:
            response = self.retrieval_chain.invoke({
                "input": query,
                "chat_history": chat_history,
                "news_context": news_context
            })
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            response = "I apologize, but I encountered an error processing your request. Please try again."
        
        return response, news_context, retrieved_docs

# Global instances
vs_loader = None
tools = None
rag_chatbot = None

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def initialize_chatbot():
    """Initialize the chatbot system"""
    global vs_loader, tools, rag_chatbot
    
    try:
        logger.info("🚀 Initializing DefendAI Chatbot...")
        
        # Initialize vector store loader
        vs_loader = VectorStoreLoader()
        vs_loader.load_vector_store()
        
        # Initialize tools
        tools = ChatbotTools()
        
        # Initialize RAG chatbot
        rag_chatbot = RAGChatbot(vs_loader, tools)
        
        logger.info("✅ DefendAI Chatbot initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {e}")
        return False

# Initialize on startup
chatbot_initialized = initialize_chatbot()

# Routes

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        "service": "DefendAI Chatbot API",
        "version": "2.0.0",
        "description": "REST API for DefendAI - Expert AI assistant for deepfake detection and media authentication",
        "endpoints": {
            "GET /": "This help message",
            "GET /health": "Health check",
            "POST /chat": "Main chatbot interaction",
            "POST /search": "Knowledge base search",
            "GET /config": "Server configuration",
            "GET /vector-store-info": "Vector store information"
        },
        "status": "ready" if chatbot_initialized else "initializing",
        "note": "This server uses pre-built vector store. Run knowledge_base_manager.py to update embeddings."
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    components = {
        "llm": "✅ Ready" if tools and tools.llm else "❌ Not available",
        "vector_store": "✅ Ready" if vs_loader and vs_loader._setup_complete else "❌ Not available",
        "news_search": "✅ Ready" if tools and tools.tavily_search else "⚠️ Disabled",
        "vector_store_path": Config.CHROMA_PERSIST_DIR
    }
    
    status = "healthy" if all("✅" in status for status in [components["llm"], components["vector_store"]]) else "degraded"
    
    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": components
    })

@app.route('/vector-store-info', methods=['GET'])
def vector_store_info():
    """Get information about the vector store"""
    try:
        persist_dir = Path(Config.CHROMA_PERSIST_DIR)
        
        info = {
            "exists": persist_dir.exists(),
            "path": str(persist_dir),
            "collection_name": Config.COLLECTION_NAME,
            "embedding_model": Config.EMBEDDING_MODEL,
            "ready": vs_loader._setup_complete if vs_loader else False
        }
        
        if persist_dir.exists():
            # Get modification time
            info["last_modified"] = datetime.fromtimestamp(
                persist_dir.stat().st_mtime
            ).isoformat()
            
            # Get size
            total_size = sum(f.stat().st_size for f in persist_dir.rglob('*') if f.is_file())
            info["size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return jsonify({
            "vector_store_info": info,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Vector store info error: {e}")
        return jsonify({
            "error": "Error getting vector store information",
            "detail": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    Main chat endpoint for DefendAI chatbot
    
    Expected JSON body:
    {
        "query": "Your question about deepfakes, AI security, or current events",
        "thread_id": "optional_conversation_thread_id",
        "include_sources": false
    }
    """
    try:
        if not chatbot_initialized or not rag_chatbot:
            return jsonify({
                "error": "Chatbot not initialized",
                "detail": "Please check server logs and ensure vector store exists. Run knowledge_base_manager.py first."
            }), 503
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        if len(query) > 1000:
            return jsonify({"error": "Query too long (max 1000 characters)"}), 400
        
        thread_id = data.get('thread_id', 'default')
        include_sources = data.get('include_sources', False)
        
        # Get response from chatbot
        response, news_context, retrieved_docs = rag_chatbot.get_response(
            query=query,
            include_sources=include_sources
        )
        
        # Format sources if requested
        sources = None
        if include_sources and retrieved_docs:
            sources = [
                {
                    "id": doc.metadata.get("id", "unknown"),
                    "title": doc.metadata.get("title", "Unknown"),
                    "type": doc.metadata.get("type", "content"),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                for doc in retrieved_docs
            ]
        
        result = {
            "response": response,
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        if sources:
            result["sources"] = sources
        if news_context:
            result["news_context"] = news_context
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({
            "error": "Error processing chat request",
            "detail": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/search', methods=['POST'])
def search_knowledge_base():
    """
    Search the DefendAI knowledge base directly
    
    Expected JSON body or URL params:
    {
        "query": "Search query",
        "limit": 5
    }
    """
    try:
        if not vs_loader or not vs_loader._setup_complete:
            return jsonify({
                "error": "Knowledge base not available",
                "detail": "Vector store not initialized. Run knowledge_base_manager.py first."
            }), 503
        
        # Get query from JSON body or URL params
        if request.is_json:
            data = request.get_json()
            query = data.get('query', '')
            limit = data.get('limit', 5)
        else:
            query = request.args.get('query', '')
            limit = int(request.args.get('limit', 5))
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        # Validate limit
        if not 1 <= limit <= 20:
            limit = 5
        
        results = vs_loader.search_knowledge_base(query, k=limit)
        
        formatted_results = [
            {
                "id": doc.metadata.get("id", "unknown"),
                "title": doc.metadata.get("title", "Unknown"),
                "type": doc.metadata.get("type", "content"),
                "content": doc.page_content,
                "relevance_score": "N/A"  # ChromaDB doesn't return scores by default
            }
            for doc in results
        ]
        
        return jsonify({
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        return jsonify({
            "error": "Error searching knowledge base",
            "detail": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/config', methods=['GET'])
def get_configuration():
    """Get current server configuration (without sensitive data)"""
    return jsonify({
        "model_config": {
            "llm_model": Config.LLM_MODEL,
            "embedding_model": Config.EMBEDDING_MODEL,
        },
        "features": {
            "google_llm_available": bool(Config.GOOGLE_API_KEY),
            "groq_llm_available": bool(Config.GROQ_API_KEY),
            "news_search_available": bool(Config.TAVILY_API_KEY),
            "vector_store_persistent": os.path.exists(Config.CHROMA_PERSIST_DIR)
        },
        "server": {
            "host": Config.HOST,
            "port": Config.PORT,
            "debug": Config.DEBUG
        },
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "detail": "The requested endpoint does not exist",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "detail": "The method is not allowed for this endpoint",
        "timestamp": datetime.now().isoformat()
    }), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "detail": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }), 500

# Main function for running the server
def run_server():
    """Run the DefendAI chatbot server"""
    logger.info(f"🚀 Starting DefendAI Chatbot Server (Optimized Flask) on {Config.HOST}:{Config.PORT}")
    
    if not chatbot_initialized:
        logger.warning("⚠️ Chatbot not fully initialized. Check logs for errors.")
        logger.info("💡 Make sure to run 'python knowledge_base_manager.py' first to create the vector store.")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True  # Enable threading for better performance
    )

if __name__ == "__main__":
    run_server()