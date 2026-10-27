"""
Streamlit Cloud Entrypoint
Imports and executes main application from app.py
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("app", run_name="__main__")
