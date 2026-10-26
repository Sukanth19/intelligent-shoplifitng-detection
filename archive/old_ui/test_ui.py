#!/usr/bin/env python3
"""
Quick test script to verify the UI setup is working
Run this to make sure everything is ready!
"""

import sys
import os
from pathlib import Path

def test_files():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'templates/index.html',
        'templates/error.html',
        'static/css/style.css',
        'static/js/app.js',
        'static/js/animations.js',
        'requirements_web.txt',
        'run_web_ui.py',
        'WEB_UI_README.md',
        'QUICKSTART.md'
    ]
    
    print("📁 Checking files...")
    missing = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING!")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing {len(missing)} files!")
        return False
    
    print("\n✅ All files present!\n")
    return True


def test_dependencies():
    """Check if required packages are installed"""
    required = [
        ('flask', 'Flask'),
        ('werkzeug', 'Werkzeug'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
    ]
    
    print("📦 Checking dependencies...")
    missing = []
    
    for module, package in required:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED!")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} packages!")
        print("   Install with: pip install -r requirements_web.txt\n")
        return False
    
    print("\n✅ All dependencies installed!\n")
    return True


def test_directories():
    """Check if required directories can be created"""
    required_dirs = ['uploads', 'static/results', 'checkpoints']
    
    print("📂 Checking directories...")
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {dir_path}")
        except Exception as e:
            print(f"  ✗ {dir_path} - ERROR: {e}")
            return False
    
    print("\n✅ All directories ready!\n")
    return True


def test_imports():
    """Test if the app can be imported"""
    print("🔍 Testing app import...")
    
    try:
        from app import app
        print("  ✓ Flask app imported successfully")
        
        # Check if routes are registered
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        print(f"  ✓ Found {len(routes)} routes")
        
        print("\n✅ App import successful!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        print("\n❌ App import failed!\n")
        return False


def print_banner():
    """Print test banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        🧪 WEB UI TEST SUITE                              ║
    ║        Verifying installation and setup                   ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_summary(results):
    """Print test summary"""
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "="*65)
    print("                        TEST SUMMARY")
    print("="*65)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test:.<50} {status}")
    
    print("="*65)
    print(f"\n  Total: {passed}/{total} tests passed\n")
    
    if passed == total:
        print("""
    🎉 SUCCESS! Everything is ready!
    
    Next steps:
      1. Run: python run_web_ui.py
      2. Open: http://localhost:5000
      3. Enjoy your beautiful UI! ✨
        """)
    else:
        print("""
    ⚠️  Some tests failed!
    
    Please fix the issues above, then run this test again.
    Check the documentation in WEB_UI_README.md for help.
        """)
    
    return passed == total


def main():
    """Run all tests"""
    print_banner()
    
    # Run tests
    results = {
        'File Structure': test_files(),
        'Dependencies': test_dependencies(),
        'Directories': test_directories(),
        'App Import': test_imports()
    }
    
    # Print summary
    success = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
