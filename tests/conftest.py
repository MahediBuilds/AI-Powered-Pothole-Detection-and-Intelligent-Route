import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_script(stem):
    """Import scripts/<stem>.py (the numbered names aren't valid module names)."""
    path = ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"script_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def severity():
    return load_script("11_classify_severity")


@pytest.fixture(scope="session")
def build_detections():
    return load_script("12_build_detections")


@pytest.fixture(scope="session")
def finetune():
    return load_script("13_finetune_local")
