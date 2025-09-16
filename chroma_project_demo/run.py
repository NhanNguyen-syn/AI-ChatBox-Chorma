#!/usr/bin/env python3
"""
Chroma AI Chat System - Setup and Run Script
This script sets up and runs both backend and frontend
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_ollama():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            return True
        else:
            print("❌ Ollama is not installed")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False

def install_ollama():
    """Install Ollama if not present"""
    print("\n🔧 Installing Ollama...")
    print("Please visit: https://ollama.ai/download")
    print("Download and install Ollama for your operating system")
    print("After installation, run: ollama pull llama2")
    return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    # Check Python packages
    required_packages = [
        'fastapi', 'uvicorn', 'chromadb', 'sqlalchemy', 
        'python-jose', 'passlib', 'python-multipart', 
        'pydantic', 'python-dotenv', 'aiofiles', 
        'langchain', 'langchain-community', 'langchain-ollama',
        'pandas', 'matplotlib', 'seaborn', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All Python packages are installed")
    return True

def setup_backend():
    """Setup backend environment"""
    print("\n🔧 Setting up backend...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    # Create .env file if it doesn't exist
    env_file = backend_dir / ".env"
    if not env_file.exists():
        print("📝 Creating .env file...")
        env_content = """SECRET_KEY=your-secret-key-here-change-this-in-production
# No OpenAI API key needed - using Ollama instead
# OPENAI_API_KEY=your-openai-api-key
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
    
    # Create static directory
    static_dir = backend_dir / "static"
    static_dir.mkdir(exist_ok=True)
    
    # Create uploads directory
    uploads_dir = backend_dir / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    
    # Initialize admin user
    try:
        subprocess.run([sys.executable, "init_admin.py"], cwd=backend_dir, check=True)
        print("✅ Admin user initialized")
    except subprocess.CalledProcessError:
        print("⚠️  Could not initialize admin user")
    
    print("✅ Backend setup complete")
    return True

def setup_frontend():
    """Setup frontend environment"""
    print("\n🔧 Setting up frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Check if node_modules exists
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("📦 Installing frontend dependencies...")
        try:
            subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
            print("✅ Frontend dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install frontend dependencies")
            return False
    else:
        print("✅ Frontend dependencies already installed")
    
    return True

def start_backend():
    """Start the backend server"""
    print("\n🚀 Starting backend server...")
    backend_dir = Path("backend")
    
    try:
        # Run backend in background
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is running
        if process.poll() is None:
            print("✅ Backend server started on http://localhost:8000")
            return process
        else:
            print("❌ Failed to start backend server")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the frontend development server"""
    print("\n🚀 Starting frontend server...")
    frontend_dir = Path("frontend")
    
    try:
        # Run frontend in background
        process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if server is running
        if process.poll() is None:
            print("✅ Frontend server started on http://localhost:5173")
            return process
        else:
            print("❌ Failed to start frontend server")
            return None
            
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def open_browser():
    """Open browser to the application"""
    print("\n🌐 Opening browser...")
    try:
        webbrowser.open("http://localhost:3002")
        print("✅ Browser opened")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")

def main():
    """Main setup and run function"""
    print("🤖 Chroma AI Chat System - Setup and Run")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check Ollama
    if not check_ollama():
        install_ollama()
        print("\n⚠️  Please install Ollama and run 'ollama pull llama2' before continuing")
        input("Press Enter after installing Ollama...")
    
    # Check dependencies
    if not check_dependencies():
        print("\n📦 Installing missing dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return
    
    # Setup backend
    if not setup_backend():
        print("❌ Backend setup failed")
        return
    
    # Setup frontend
    if not setup_frontend():
        print("❌ Frontend setup failed")
        return
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Could not start backend")
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Could not start frontend")
        backend_process.terminate()
        return
    
    # Open browser
    open_browser()
    
    print("\n🎉 System is running!")
    print("📱 Frontend: http://localhost:3002")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n💡 Default admin credentials:")
    print("   Username: admin")
    print("   Password: admin123")
    print("\n⚠️  Remember to:")
    print("   1. Install Ollama: https://ollama.ai/download")
    print("   2. Run: ollama pull llama2")
    print("   3. Change default admin password")
    
    try:
        print("\n🛑 Press Ctrl+C to stop the servers...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print("✅ Servers stopped")

if __name__ == "__main__":
    main() 