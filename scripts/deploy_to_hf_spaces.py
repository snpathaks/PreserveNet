"""
scripts/deploy_to_hf_spaces.py
------------------------------
Deploys the PreserveNet Gradio demo application to Hugging Face Spaces.

Usage:
    # Set your HF write token:
    set HF_TOKEN=hf_...

    # Run deployment:
    python scripts/deploy_to_hf_spaces.py --space_id <username>/preservenet-demo

    # Or pass token directly:
    python scripts/deploy_to_hf_spaces.py --space_id <username>/preservenet-demo --token hf_...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from huggingface_hub import HfApi, create_repo
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


def deploy_to_spaces(
    space_id: str,
    token: str | None = None,
    private: bool = False,
    app_dir: Path | None = None,
):
    if not HAS_HF_HUB:
        raise ImportError("huggingface_hub is required. Install via: pip install huggingface_hub")

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[-] Error: No Hugging Face token provided.")
        print("    Create a write-access token at: https://huggingface.co/settings/tokens")
        print("    Then pass --token <your_token> or set the HF_TOKEN environment variable.")
        return False

    if app_dir is None:
        candidates = [REPO_ROOT / 'app', REPO_ROOT / 'PreserveNet' / 'app']
        for c in candidates:
            if (c / 'app.py').is_file():
                app_dir = c
                break

    if app_dir is None or not (app_dir / 'app.py').is_file():
        raise FileNotFoundError(f"Could not locate valid app directory containing app.py in {REPO_ROOT}")

    print(f"[*] Deploying Space from: {app_dir}")
    print(f"[*] Target Hugging Face Space: https://huggingface.co/spaces/{space_id}")

    api = HfApi(token=hf_token)

    print("[*] Ensuring Space repository exists...")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        private=private,
        exist_ok=True,
    )

    print("[*] Uploading application files (app.py, inference.py, examples, weights)...")
    api.upload_folder(
        folder_path=str(app_dir),
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy PreserveNet Gradio demo to Hugging Face Spaces",
    )

    print("[✓] Deployment files uploaded successfully!")
    print(f"[★] Live Space URL: https://huggingface.co/spaces/{space_id}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Deploy PreserveNet to Hugging Face Spaces")
    parser.add_argument("--space_id", type=str, required=True,
                        help="Hugging Face Space ID (e.g. username/preservenet-demo)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face write token (or set HF_TOKEN environment variable)")
    parser.add_argument("--private", action="store_true",
                        help="Create a private space if it does not exist")
    args = parser.parse_args()

    print("=" * 60)
    print("PreserveNet — Hugging Face Spaces Deployer")
    print("=" * 60)
    deploy_to_spaces(
        space_id=args.space_id,
        token=args.token,
        private=args.private,
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
