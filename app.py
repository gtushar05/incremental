"""Hugging Face Spaces entrypoint — the real app lives in app/streamlit_app.py."""
from pathlib import Path

exec(compile((Path(__file__).parent / "app" / "streamlit_app.py").read_text(),
             "app/streamlit_app.py", "exec"))
