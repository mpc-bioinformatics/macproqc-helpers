
"""Allow running the macproqc module as a script: python -m macproqc_helpers"""

import sys
from macproqc_helpers.cli import main

if __name__ == "__main__":
    sys.exit(main())
