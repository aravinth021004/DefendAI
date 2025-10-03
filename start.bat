@echo off
echo 🚀 Starting DefendAI Application...
echo.

echo 📁 Creating virtual environment...
cd backend
if not exist "venv" (
    python -m venv venv
)

echo 🔧 Activating virtual environment...
call venv\Scripts\activate

echo 📦 Installing Python dependencies...
pip install -r requirements.txt

echo 🌐 Starting Flask backend server...
start "DefendAI Backend" cmd /k "python app.py"

echo ⏳ Waiting for backend to start...
timeout /t 5 /nobreak > nul

cd ..\frontend

echo 📦 Installing Node.js dependencies...
if not exist "node_modules" (
    npm install
)

echo 🎨 Starting React frontend server...
start "DefendAI Frontend" cmd /k "npm start"

echo.
echo ✅ DefendAI is starting up!
echo 🌐 Frontend: http://localhost:3000
echo 🔧 Backend: http://localhost:5000
echo.
echo Press any key to exit...
pause > nul
