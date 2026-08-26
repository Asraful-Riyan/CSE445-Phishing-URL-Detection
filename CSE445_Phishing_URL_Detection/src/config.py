from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

DATA_FILE = DATA_DIR / "url_features_extracted1.csv"
TARGET_COLUMN = "ClassLabel"
URL_COLUMN = "URL"

# LegitPhish numeric features.
FEATURE_COLUMNS = [
    "url_length",
    "has_ip_address",
    "dot_count",
    "https_flag",
    "url_entropy",
    "token_count",
    "subdomain_count",
    "query_param_count",
    "tld_length",
    "path_length",
    "has_hyphen_in_domain",
    "number_of_digits",
    "tld_popularity",
    "suspicious_file_extension",
    "domain_name_length",
    "percentage_numeric_chars",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20

for folder in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
