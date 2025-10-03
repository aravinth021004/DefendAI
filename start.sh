#!/bin/bash

echo "🚀 Starting DefendAI Application..."
echo

echo "📁 Setting up backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "🌐 Starting Flask backend server..."
python app.py &
BACKEND_PID=$!

echo "⏳ Waiting for backend to start..."
sleep 5

cd ../frontend

echo "📦 Installing Node.js dependencies..."
if [ ! -d "node_modules" ]; then
    npm install
fi

echo "🎨 Starting React frontend server..."
npm start &
FRONTEND_PID=$!

echo
echo "✅ DefendAI is running!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend: http://localhost:5000"
echo
echo "Press Ctrl+C to stop all servers"

# Wait for user to stop
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
