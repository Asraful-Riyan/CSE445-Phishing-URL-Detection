import math
import re
from collections import Counter
from urllib.parse import urlparse, parse_qs

POPULAR_TLDS = {
    "com", "org", "net", "edu", "gov", "io", "co", "uk", "de",
    "jp", "fr", "au", "us", "ca", "info", "biz", "me"
}

SUSPICIOUS_EXTENSIONS = {
    ".exe", ".zip", ".rar", ".scr", ".bat", ".cmd", ".apk", ".msi", ".js"
}


def shannon_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def contains_ip(hostname):
    if not hostname:
        return 0
    ipv4 = re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", hostname)
    return int(bool(ipv4))


def extract_url_features(url):
    """
    Approximate the LegitPhish lexical/structural features for a new raw URL.
    This is intended for the real-time demonstration.
    """
    if "://" not in url:
        parse_url = "http://" + url
    else:
        parse_url = url

    parsed = urlparse(parse_url)
    hostname = parsed.hostname or ""
    domain_parts = [p for p in hostname.split(".") if p]
    tld = domain_parts[-1].lower() if domain_parts else ""
    subdomain_count = max(0, len(domain_parts) - 2)
    path = parsed.path or ""
    query = parsed.query or ""

    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", url) if t]
    digits = sum(ch.isdigit() for ch in url)
    pct_numeric = digits / len(url) if url else 0.0

    suspicious_ext = int(any(path.lower().endswith(ext) for ext in SUSPICIOUS_EXTENSIONS))

    features = {
        "url_length": len(url),
        "has_ip_address": contains_ip(hostname),
        "dot_count": url.count("."),
        "https_flag": int(parsed.scheme.lower() == "https"),
        "url_entropy": shannon_entropy(url),
        "token_count": len(tokens),
        "subdomain_count": subdomain_count,
        "query_param_count": len(parse_qs(query, keep_blank_values=True)),
        "tld_length": len(tld),
        "path_length": len(path),
        "has_hyphen_in_domain": int("-" in hostname),
        "number_of_digits": digits,
        "tld_popularity": int(tld in POPULAR_TLDS),
        "suspicious_file_extension": suspicious_ext,
        "domain_name_length": len(hostname),
        "percentage_numeric_chars": pct_numeric,
    }
    return features
