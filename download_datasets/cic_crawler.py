import os
import re
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = sys.argv[1]
COOKIE_FILE = sys.argv[2]  # used for enumeration (browse.php) only
DOWNLOAD_COOKIE_FILES = sys.argv[3].split(",")  # one distinct session per parallel worker
OUT_DIR = sys.argv[4]
MAX_PARALLEL_DOWNLOADS = len(DOWNLOAD_COOKIE_FILES)
CURL_MAX_TIME_SECONDS = "120"
MAX_RETRIES = 3

LINK_RE = re.compile(r'href="(browse\.php\?p=[^"]*|download\.php\?file=[^"]*)"')
SKIPPED_EXTENSIONS = (".pcap",)
_visited: set[str] = set()


def curl_get(url: str) -> str:
    for attempt in range(MAX_RETRIES):
        result = subprocess.run(
            ["curl", "-s", "--max-time", CURL_MAX_TIME_SECONDS, "-c", COOKIE_FILE, "-b",
             COOKIE_FILE, url],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"curl_get failed after {MAX_RETRIES} retries: {url}")


def enumerate_files(rel_path: str, files: list[str]) -> None:
    if rel_path in _visited:
        return
    _visited.add(rel_path)
    url = BASE + "browse.php"
    if rel_path:
        url += "?p=" + urllib.parse.quote(rel_path)
    html = curl_get(url)
    for href in LINK_RE.findall(html):
        if href.startswith("browse.php?p="):
            p = urllib.parse.unquote_plus(href.split("=", 1)[1])
            if p == rel_path:
                continue
            if rel_path and not p.startswith(rel_path + "/"):
                continue
            enumerate_files(p, files)
        elif href.startswith("download.php?file="):
            f = urllib.parse.unquote_plus(href.split("=", 1)[1])
            if f.lower().endswith(SKIPPED_EXTENSIONS):
                continue
            files.append(f)


def download(rel_file: str, cookie_file: str) -> str:
    """Downloads to a `.partial` temp file, then atomically renames to the final path only on
    verified success (curl exit 0, --fail on HTTP errors). The final filename existing is
    therefore a guarantee of a complete download; a leftover `.partial` file from a killed run
    is never mistaken for a complete one, since `os.path.exists(final_path)` is the only skip
    condition and that path is never created except by the rename.

    `cookie_file` is one of `DOWNLOAD_COOKIE_FILES`, a distinct registered session per parallel
    worker slot: the server serializes downloads within one PHP session, so sharing a single
    cookie across concurrent curl processes would silently collapse them back to sequential.
    """
    local_path = f"{OUT_DIR}/{rel_file}"
    if os.path.exists(local_path):
        return f"skip (exists): {rel_file}"

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = local_path + ".partial"
    url = BASE + "download.php?file=" + urllib.parse.quote(rel_file)

    for attempt in range(MAX_RETRIES):
        result = subprocess.run(
            [
                "curl", "-s", "-L", "--fail", "--max-time", "1800",
                "-c", cookie_file, "-b", cookie_file, "-o", tmp_path, url,
            ]
        )
        if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            os.replace(tmp_path, local_path)
            return f"done: {rel_file}"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        time.sleep(10 * (attempt + 1))
    return f"FAILED after {MAX_RETRIES} retries: {rel_file}"


files: list[str] = []
enumerate_files("", files)
print(f"ENUMERATED {len(files)} files", flush=True)

already_present = sum(1 for f in files if os.path.exists(f"{OUT_DIR}/{f}"))
print(f"ALREADY_COMPLETE {already_present}/{len(files)}", flush=True)

with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DOWNLOADS) as pool:
    futures = {
        pool.submit(download, f, DOWNLOAD_COOKIE_FILES[i % MAX_PARALLEL_DOWNLOADS]): f
        for i, f in enumerate(files)
    }
    completed = 0
    for future in as_completed(futures):
        completed += 1
        print(f"[{completed}/{len(files)}] {future.result()}", flush=True)

print("DONE", flush=True)
