#!/bin/bash

# Enable strict error handling
set -e

# Function to cleanup on exit
cleanup() {
    echo
    echo "🛑 Shutting down DefendAI..."
    if [ ! -z "$BACKEND_PID" ] && kill -0 $BACKEND_PID 2>/dev/null; then
        echo "Stopping backend server..."
        kill $BACKEND_PID
    fi
    if [ ! -z "$FRONTEND_PID" ] && kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "Stopping frontend server..."
        kill $FRONTEND_PID
    fi
    echo "✅ DefendAI stopped successfully"
    exit 0
}

# Set up trap for cleanup
trap cleanup INT TERM

echo "🚀 Starting DefendAI Application..."
echo

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed or not in PATH"
    echo "Please install Node.js and try again"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed or not in PATH"
    echo "Please install npm and try again"
    exit 1
fi

echo "📁 Setting up backend environment..."
cd "$(dirname "$0")/backend"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "📦 Installing/Updating Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo "🌐 Starting Flask backend server..."
python app.py &
BACKEND_PID=$!

# Check if backend started successfully
sleep 3
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend server failed to start"
    exit 1
fi

echo "⏳ Waiting for backend to initialize..."
sleep 5

cd ../frontend

echo "📦 Installing/Updating Node.js dependencies..."
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js packages..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Node.js dependencies"
        cleanup
        exit 1
    fi
else
    echo "Checking for package updates..."
    npm update
fi

echo "🎨 Starting React frontend server..."
npm start &
FRONTEND_PID=$!

# Check if frontend started successfully
sleep 3
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Frontend server failed to start"
    cleanup
    exit 1
fi

echo
echo "✅ DefendAI is running successfully!"
echo
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:5000"
echo "📊 Health Check: http://localhost:5000/api/health"
echo
echo "📊 Server Status:"
echo "   Backend PID: $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo
echo "Press Ctrl+C to stop all servers"
echo

# Wait for user to stop or for processes to exit
while kill -0 $BACKEND_PID 2>/dev/null && kill -0 $FRONTEND_PID 2>/dev/null; do
    sleep 1
done

echo "⚠️  One or more servers have stopped unexpectedly"
cleanup
