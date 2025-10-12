"""
DefendAI Knowledge Base Manager
Handles vector store creation and management for the DefendAI RAG chatbot.
Only runs when training data is updated to avoid unnecessary re-embedding.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# LangChain imports
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Environment and configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class KnowledgeBaseConfig:
    """Configuration settings for the knowledge base manager"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector store settings
    CHROMA_PERSIST_DIR = "./defendai_vectorstore"
    COLLECTION_NAME = "deepfake_knowledge"
    TRAINING_DATA_PATH = "./chatbot_training_data.json"
    HASH_FILE_PATH = "./data_hash.txt"

# Knowledge Base Manager
class KnowledgeBaseManager:
    """Manages the vector store and knowledge base operations"""
    
    def __init__(self):
        self.embeddings = self._initialize_embeddings()
        self.config = KnowledgeBaseConfig()
    
    def _initialize_embeddings(self):
        """Initialize embeddings with fallbacks"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=KnowledgeBaseConfig.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            logger.info("✅ HuggingFace embeddings initialized")
            return embeddings
        except Exception as e:
            logger.error(f"❌ HuggingFace embeddings failed: {e}")
            # Fallback to Google embeddings if available
            if KnowledgeBaseConfig.GOOGLE_API_KEY:
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=KnowledgeBaseConfig.GOOGLE_API_KEY
                    )
                    logger.info("✅ Google Gemini embeddings initialized as fallback")
                    return embeddings
                except Exception as gemini_error:
                    logger.error(f"❌ Google embeddings also failed: {gemini_error}")
            
            raise ValueError("No embedding models available. Please install sentence-transformers or set GOOGLE_API_KEY")
    
    def _calculate_data_hash(self, json_path: str) -> str:
        """Calculate hash of training data file to detect changes"""
        try:
            with open(json_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def _get_stored_hash(self) -> str:
        """Get previously stored hash"""
        try:
            if os.path.exists(self.config.HASH_FILE_PATH):
                with open(self.config.HASH_FILE_PATH, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading stored hash: {e}")
        return ""
    
    def _save_hash(self, file_hash: str):
        """Save current hash to file"""
        try:
            with open(self.config.HASH_FILE_PATH, 'w') as f:
                f.write(file_hash)
        except Exception as e:
            logger.error(f"Error saving hash: {e}")
    
    def _has_data_changed(self, json_path: str) -> bool:
        """Check if training data has changed since last embedding"""
        current_hash = self._calculate_data_hash(json_path)
        stored_hash = self._get_stored_hash()
        
        if current_hash != stored_hash:
            logger.info(f"📊 Data change detected. Current: {current_hash[:8]}, Stored: {stored_hash[:8]}")
            return True
        
        logger.info("📊 No data changes detected. Using existing vector store.")
        return False
    
    def load_training_data(self, json_path: str) -> List[Document]:
        """Load and process the chatbot training data"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents = []
            for item in data['chatbot_training_data']:
                # Create main content document
                main_content = f"Title: {item['title']}\n\nContent: {item['content']}"
                doc = Document(
                    page_content=main_content,
                    metadata={
                        'id': item['id'],
                        'title': item['title'],
                        'type': 'main_content'
                    }
                )
                documents.append(doc)
                
                # Create step-by-step guide document if available
                if 'steps' in item:
                    steps_content = f"Steps for {item['title']}:\n" + "\n".join(
                        f"{i+1}. {step}" for i, step in enumerate(item['steps'])
                    )
                    step_doc = Document(
                        page_content=steps_content,
                        metadata={
                            'id': f"{item['id']}_steps",
                            'title': f"Steps: {item['title']}",
                            'type': 'steps'
                        }
                    )
                    documents.append(step_doc)
            
            logger.info(f"✅ Loaded {len(documents)} documents from training data")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Failed to load training data: {e}")
            raise ValueError(f"Cannot load training data from {json_path}: {e}")
    
    def create_vector_store(self, documents: List[Document], force_recreate: bool = False):
        """Create vector store from documents"""
        try:
            persist_dir = Path(self.config.CHROMA_PERSIST_DIR)
            
            # Remove existing vector store if forcing recreation
            if force_recreate and persist_dir.exists():
                import shutil
                shutil.rmtree(persist_dir)
                logger.info("🗑️ Removed existing vector store for recreation")
            
            logger.info("🔄 Creating new vector store...")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(persist_dir),
                collection_name=self.config.COLLECTION_NAME
            )
            logger.info("💾 Vector store created and persisted!")
            return vector_store
            
        except Exception as e:
            logger.error(f"❌ Vector store creation failed: {e}")
            raise ValueError(f"Cannot create vector store: {e}")
    
    def load_existing_vector_store(self):
        """Load existing vector store"""
        try:
            persist_dir = Path(self.config.CHROMA_PERSIST_DIR)
            
            if not persist_dir.exists():
                raise FileNotFoundError("Vector store directory does not exist")
            
            logger.info("📚 Loading existing vector store...")
            vector_store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings,
                collection_name=self.config.COLLECTION_NAME
            )
            logger.info("✅ Existing vector store loaded successfully!")
            return vector_store
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing vector store: {e}")
            raise ValueError(f"Cannot load vector store: {e}")
    
    def setup_knowledge_base(self, force_recreate: bool = False) -> bool:
        """
        Main method to setup knowledge base.
        Only recreates vector store if data has changed or force_recreate is True.
        
        Args:
            force_recreate: Force recreation of vector store regardless of data changes
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Check if training data exists
            if not os.path.exists(self.config.TRAINING_DATA_PATH):
                logger.error(f"❌ Training data not found at {self.config.TRAINING_DATA_PATH}")
                return False
            
            persist_dir = Path(self.config.CHROMA_PERSIST_DIR)
            
            # Check if we need to recreate the vector store
            needs_recreation = (
                force_recreate or 
                not persist_dir.exists() or 
                self._has_data_changed(self.config.TRAINING_DATA_PATH)
            )
            
            if needs_recreation:
                logger.info("🔄 Setting up knowledge base with new data...")
                
                # Load training data
                documents = self.load_training_data(self.config.TRAINING_DATA_PATH)
                
                # Create vector store
                vector_store = self.create_vector_store(documents, force_recreate)
                
                # Save current data hash
                current_hash = self._calculate_data_hash(self.config.TRAINING_DATA_PATH)
                self._save_hash(current_hash)
                
                logger.info("✅ Knowledge base setup completed!")
            else:
                logger.info("📚 Using existing vector store (no changes detected)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Knowledge base setup failed: {e}")
            return False
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the current vector store"""
        try:
            persist_dir = Path(self.config.CHROMA_PERSIST_DIR)
            
            info = {
                "exists": persist_dir.exists(),
                "path": str(persist_dir),
                "collection_name": self.config.COLLECTION_NAME,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "training_data_path": self.config.TRAINING_DATA_PATH,
                "training_data_exists": os.path.exists(self.config.TRAINING_DATA_PATH)
            }
            
            if persist_dir.exists():
                # Get modification time
                info["last_modified"] = datetime.fromtimestamp(
                    persist_dir.stat().st_mtime
                ).isoformat()
                
                # Get stored hash
                info["data_hash"] = self._get_stored_hash()
            
            if os.path.exists(self.config.TRAINING_DATA_PATH):
                info["current_data_hash"] = self._calculate_data_hash(self.config.TRAINING_DATA_PATH)
                info["data_changed"] = self._has_data_changed(self.config.TRAINING_DATA_PATH)
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting vector store info: {e}")
            return {"error": str(e)}


def main():
    """Main function to setup knowledge base"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DefendAI Knowledge Base Manager")
    parser.add_argument("--force", action="store_true", help="Force recreation of vector store")
    parser.add_argument("--info", action="store_true", help="Show vector store information")
    
    args = parser.parse_args()
    
    try:
        kb_manager = KnowledgeBaseManager()
        
        if args.info:
            # Show vector store information
            info = kb_manager.get_vector_store_info()
            print("\n📊 Vector Store Information:")
            for key, value in info.items():
                print(f"   {key}: {value}")
            return
        
        # Setup knowledge base
        logger.info("🚀 Starting DefendAI Knowledge Base Setup...")
        success = kb_manager.setup_knowledge_base(force_recreate=args.force)
        
        if success:
            logger.info("✅ Knowledge base setup completed successfully!")
            
            # Show final info
            info = kb_manager.get_vector_store_info()
            print(f"\n📊 Vector Store Status:")
            print(f"   Location: {info.get('path', 'N/A')}")
            print(f"   Exists: {info.get('exists', False)}")
            print(f"   Collection: {info.get('collection_name', 'N/A')}")
            if info.get('last_modified'):
                print(f"   Last Modified: {info.get('last_modified')}")
        else:
            logger.error("❌ Knowledge base setup failed!")
            exit(1)
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        exit(1)


if __name__ == "__main__":
    main()