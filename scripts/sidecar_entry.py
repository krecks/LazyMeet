"""PyInstaller entry point: import the backend package and run the sidecar.

Using a thin launcher (instead of freezing ``backend/main.py`` directly) keeps
the backend's intra-package relative imports working inside the frozen binary.
"""

from backend.main import main

if __name__ == "__main__":
    main()
