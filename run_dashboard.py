"""
Streamlit Dashboard Runner.
Run with: python run_dashboard.py
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    dashboard_path = os.path.join(
        os.path.dirname(__file__),
        "hacktronix", "dashboard", "app.py"
    )
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard_path,
        "--server.port", "8501",
        "--server.headless", "true",
        "--theme.base", "dark",
    ])
