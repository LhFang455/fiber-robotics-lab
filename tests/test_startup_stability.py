"""Cold-process regression for a concurrent pandas import during startup."""

from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def test_cold_start_waits_for_concurrent_pandas_import():
    result = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent("""
            import importlib
            import importlib.abc
            import importlib.machinery
            import sys
            import threading
            from streamlit.testing.v1 import AppTest

            started = threading.Event()
            release = threading.Event()

            class SlowPandas(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname != "pandas":
                        return None
                    spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                    original = spec.loader.exec_module
                    def execute(module):
                        started.set()
                        release.wait(timeout=8)
                        original(module)
                    spec.loader.exec_module = execute
                    return spec

            assert "pandas" not in sys.modules
            sys.meta_path.insert(0, SlowPandas())
            worker = threading.Thread(target=lambda: importlib.import_module("pandas"))
            worker.start()
            assert started.wait(timeout=5)
            try:
                app = AppTest.from_file("app.py", default_timeout=35).run()
                errors = [item.message for item in app.exception]
            finally:
                release.set()
                worker.join(timeout=15)
            assert not errors, errors
            assert app.selectbox(key="sidebar_module_navigation")
        """)],
        cwd=ROOT, capture_output=True, text=True, timeout=55,
    )
    assert result.returncode == 0, result.stdout + result.stderr
