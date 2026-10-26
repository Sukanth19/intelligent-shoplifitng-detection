#!/usr/bin/env python3
"""
Shoplifting Detection System - Web UI Launcher
Easy-to-use startup script with configuration options
"""

import os
import sys
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = ['flask', 'cv2', 'numpy', 'torch']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements_web.txt")
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    dirs = ['uploads', 'static/results', 'checkpoints']
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created/verified")

def print_banner():
    """Display startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🛡️  SHOPLIFTING DETECTION SYSTEM - WEB UI  🛡️        ║
    ║                                                           ║
    ║     Intelligent AI-Powered Surveillance                   ║
    ║     Built with Flask, PyTorch & Modern Web Tech           ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    parser = argparse.ArgumentParser(
        description='Launch the Shoplifting Detection Web UI'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    parser.add_argument(
        '--no-reload',
        action='store_true',
        help='Disable auto-reload in debug mode'
    )
    
    args = parser.parse_args()
    
    # Display banner
    print_banner()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies installed")
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Check for model checkpoint
    checkpoint_path = Path('checkpoints/best_model.pt')
    if not checkpoint_path.exists():
        print("\n⚠️  Warning: No model checkpoint found at 'checkpoints/best_model.pt'")
        print("   The UI will run but analysis may not work properly.")
        print("   Train a model first or place a checkpoint in the checkpoints/ directory.")
    else:
        print("✅ Model checkpoint found")
    
    # Import and run Flask app
    print(f"\n🚀 Starting web server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Debug: {args.debug}")
    print(f"\n🌐 Access the UI at: http://localhost:{args.port}")
    print("   Press Ctrl+C to stop\n")
    
    try:
        from app import app
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=not args.no_reload and args.debug
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
