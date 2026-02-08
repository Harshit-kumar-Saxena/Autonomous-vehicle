"""
AetherNav Stack entry point.

Run with: python aethernav_stack/run.py [--single-threaded]
"""

import sys
import os

# Add parent directory to path for imports when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default to threaded executor (recommended)
if "--single-threaded" in sys.argv:
    sys.argv.remove("--single-threaded")
    from aethernav_stack.executor import main
else:
    from aethernav_stack.executor_threaded import main

if __name__ == "__main__":
    main()
