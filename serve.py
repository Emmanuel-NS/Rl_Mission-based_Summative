"""Serve the trained rider policy as a JSON API: uv run python serve.py --port 8000"""

from service.server import main

if __name__ == "__main__":
    main()
