"""
DefendAI Setup and Management Script
Helps manage the knowledge base and chatbot server
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def setup_knowledge_base(force=False):
    """Setup or update the knowledge base"""
    force_flag = "--force" if force else ""
    command = f"python knowledge_base_manager.py {force_flag}"
    return run_command(command, "Setting up knowledge base")

def show_kb_info():
    """Show knowledge base information"""
    command = "python knowledge_base_manager.py --info"
    return run_command(command, "Getting knowledge base information")

def start_server():
    """Start the optimized chatbot server"""
    command = "python optimized_chatbot_server.py"
    print(f"\n🚀 Starting DefendAI Chatbot Server...")
    print("Press Ctrl+C to stop the server")
    try:
        subprocess.run(command, shell=True)
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")

def check_requirements():
    """Check if required files exist"""
    required_files = [
        "knowledge_base_manager.py",
        "optimized_chatbot_server.py",
        "chatbot_training_data.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True

def main():
    parser = argparse.ArgumentParser(description="DefendAI Management Script")
    parser.add_argument("action", choices=["setup", "update", "info", "server", "check"], 
                      help="Action to perform")
    parser.add_argument("--force", action="store_true", 
                      help="Force recreation of knowledge base (use with setup/update)")
    
    args = parser.parse_args()
    
    print("🛡️ DefendAI Management Script")
    print("=" * 40)
    
    # Check requirements first
    if not check_requirements():
        sys.exit(1)
    
    if args.action == "check":
        print("✅ Requirements check passed!")
        
    elif args.action == "setup":
        print("📚 Setting up knowledge base for the first time...")
        success = setup_knowledge_base(force=True)  # Always force on first setup
        if success:
            print("\n💡 Knowledge base setup complete!")
            print("💡 You can now run: python manage.py server")
        
    elif args.action == "update":
        print("🔄 Updating knowledge base...")
        success = setup_knowledge_base(force=args.force)
        if success:
            print("\n💡 Knowledge base updated!")
            print("💡 Restart the server to use updated data")
        
    elif args.action == "info":
        print("📊 Getting knowledge base information...")
        show_kb_info()
        
    elif args.action == "server":
        # Check if vector store exists
        if not os.path.exists("./defendai_vectorstore"):
            print("⚠️ Vector store not found!")
            print("💡 Run 'python manage.py setup' first")
            sys.exit(1)
        
        start_server()

if __name__ == "__main__":
    main()