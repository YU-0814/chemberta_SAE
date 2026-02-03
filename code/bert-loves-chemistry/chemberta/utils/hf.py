import os


def get_hf_token() -> str | None:
    """Return a Hugging Face token if provided via env vars."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
