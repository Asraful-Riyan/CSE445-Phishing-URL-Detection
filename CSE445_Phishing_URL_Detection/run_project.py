import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from train_all import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate()
