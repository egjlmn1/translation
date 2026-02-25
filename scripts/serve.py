"""Launch the API server + both web UIs (without training)."""
import argparse
import os
import sys

import uvicorn
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.server import create_app


def main():
    parser = argparse.ArgumentParser(description="Launch translation server")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    app = create_app(config=config)

    host = config["server"]["host"]
    port = config["server"]["port"]
    print(f"[serve] Starting server at http://localhost:{port}")
    print(f"  Translate: http://localhost:{port}/translate")
    print(f"  Dashboard: http://localhost:{port}/dashboard")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
