"""
AetherNav Stack entry point.

Run with: python -m aethernav_stack [--threaded]
"""

import sys

# Default to threaded executor (recommended)
if "--single-threaded" in sys.argv:
    sys.argv.remove("--single-threaded")
    from .executor import main
else:
    from .executor_threaded import main

if __name__ == "__main__":
    main()
