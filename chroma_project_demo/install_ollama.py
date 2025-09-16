#!/usr/bin/env python3
"""
Ollama Installation Helper Script
This script helps users install Ollama and set up the AI model
"""

import os
import sys
import subprocess
import platform
import webbrowser

def detect_os():
    """Detect the operating system"""
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    else:
        return "unknown"

def get_ollama_download_url():
    """Get the appropriate Ollama download URL for the OS"""
    os_type = detect_os()
    
    if os_type == "windows":
        return "https://ollama.ai/download/windows"
    elif os_type == "macos":
        return "https://ollama.ai/download/macos"
    elif os_type == "linux":
        return "https://ollama.ai/download/linux"
    else:
        return "https://ollama.ai/download"

def check_ollama_installed():
    """Check if Ollama is already installed"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ollama is already installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama is installed but not working properly")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ Ollama is not installed")
        return False

def install_ollama_windows():
    """Install Ollama on Windows"""
    print("\n🔧 Installing Ollama on Windows...")
    print("1. Download Ollama from: https://ollama.ai/download/windows")
    print("2. Run the installer")
    print("3. Restart your terminal/command prompt")
    print("4. Run: ollama pull llama2")
    
    # Open download page
    try:
        webbrowser.open("https://ollama.ai/download/windows")
    except:
        pass
    
    input("\nPress Enter after installing Ollama...")

def install_ollama_macos():
    """Install Ollama on macOS"""
    print("\n🔧 Installing Ollama on macOS...")
    
    try:
        # Try to install using Homebrew
        print("Attempting to install via Homebrew...")
        subprocess.run(['brew', 'install', 'ollama'], check=True)
        print("✅ Ollama installed via Homebrew")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Homebrew not available or installation failed")
        print("Please visit: https://ollama.ai/download/macos")
        
        try:
            webbrowser.open("https://ollama.ai/download/macos")
        except:
            pass
        
        input("Press Enter after installing Ollama...")
        return False

def install_ollama_linux():
    """Install Ollama on Linux"""
    print("\n🔧 Installing Ollama on Linux...")
    
    try:
        # Try to install using curl
        print("Installing via curl...")
        subprocess.run([
            'curl', '-fsSL', 'https://ollama.ai/install.sh', '|', 'sh'
        ], shell=True, check=True)
        print("✅ Ollama installed via curl")
        return True
    except subprocess.CalledProcessError:
        print("Curl installation failed")
        print("Please visit: https://ollama.ai/download/linux")
        
        try:
            webbrowser.open("https://ollama.ai/download/linux")
        except:
            pass
        
        input("Press Enter after installing Ollama...")
        return False

def pull_llama_model():
    """Pull the Llama2 model"""
    print("\n📥 Pulling Llama2 model...")
    print("This may take several minutes depending on your internet connection...")
    
    try:
        subprocess.run(['ollama', 'pull', 'llama2'], check=True)
        print("✅ Llama2 model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to pull Llama2 model: {e}")
        return False

def test_ollama():
    """Test if Ollama is working"""
    print("\n🧪 Testing Ollama...")
    
    try:
        # Test with a simple prompt
        result = subprocess.run([
            'ollama', 'run', 'llama2', 'Hello, how are you?'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Ollama is working correctly!")
            print("Sample response:", result.stdout[:100] + "...")
            return True
        else:
            print("❌ Ollama test failed")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Ollama test timed out (this is normal for first run)")
        return True
    except Exception as e:
        print(f"❌ Ollama test error: {e}")
        return False

def main():
    """Main installation function"""
    print("🤖 Ollama Installation Helper")
    print("=" * 40)
    
    # Check if already installed
    if check_ollama_installed():
        print("\n🎉 Ollama is already installed!")
        
        # Check if model is available
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if 'llama2' in result.stdout:
                print("✅ Llama2 model is already downloaded")
                if test_ollama():
                    print("\n🎉 Everything is ready! You can now run the chat system.")
                    return
            else:
                print("⚠️  Llama2 model not found")
                if pull_llama_model():
                    test_ollama()
                return
        except:
            pass
    
    # Install Ollama
    os_type = detect_os()
    
    if os_type == "windows":
        install_ollama_windows()
    elif os_type == "macos":
        install_ollama_macos()
    elif os_type == "linux":
        install_ollama_linux()
    else:
        print("❌ Unsupported operating system")
        print("Please visit: https://ollama.ai/download")
        return
    
    # Verify installation
    if not check_ollama_installed():
        print("❌ Ollama installation verification failed")
        print("Please restart your terminal and try again")
        return
    
    # Pull model
    if pull_llama_model():
        # Test
        if test_ollama():
            print("\n🎉 Ollama installation complete!")
            print("You can now run the chat system with: python run.py")
        else:
            print("\n⚠️  Installation complete but testing failed")
            print("Try restarting your terminal and running: ollama run llama2 'Hello'")
    else:
        print("\n❌ Failed to download Llama2 model")
        print("Please check your internet connection and try again")

if __name__ == "__main__":
    main() 