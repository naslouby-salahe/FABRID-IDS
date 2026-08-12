# External dataset downloads — how this was done, and how to resume

This folder holds the crawler script and session cookies used to pull the two CIC datasets
(CIC IoT-DIAD 2024, CICIoMT2024) needed for external replication / event-level validation
(roadmap EXTERNAL-001, EVENT-001). Gotham 2025 was a plain direct download, no crawler needed.

## What's downloaded, and where

| Dataset | Destination | Format | Status as of last check |
|---|---|---|---|
| CIC IoT-DIAD 2024 | `data/raw/CIC_IoT_DIAD_2024/` (symlinked to `datp-shared-data/raw/`) | CSV only (flow + packet features, no raw PCAPs exist in this dataset variant) | 314 files total, run `download_datasets/check_progress.sh` for current count |
| CICIoMT2024 | `data/raw/CICIoMT2024/` | CSV + PDF (README) — `.pcap` files skipped by the crawler | ~197 files (247 enumerated minus 50 skipped PCAPs) |
| Gotham 2025 | `data/raw/Gotham2025/` — `GothamDataset2025.zip` (23.8GB, full) + `extracted/processed/` (6.5GB of feature CSVs, unzipped) | zip + extracted CSVs | Only `processed/` was extracted; the 30GB `raw/` PCAP tree inside the zip was left compressed (needs packet-parsing tooling — scapy/tshark — not yet in this project's dependencies) |

## How the CIC downloads work

Both CIC sites (`cicresearch.ca`) gate every dataset behind a one-time registration form
(name/email/institution/job title/country) that returns a session cookie (`Token`). That
session cookie is required for every subsequent `browse.php` (directory listing) and
`download.php` (file fetch) call.

**Key discovery: the server serializes requests within one PHP session.** Opening N concurrent
curl connections with the *same* cookie does not give you N-way parallel throughput — the
server processes them one at a time regardless. To get real parallelism, register **one
session per parallel worker slot** and round-robin assign them.

`cic_crawler.py` in this folder:
1. Takes `BASE` (site root), `COOKIE_FILE` (a single session, used only for read-only directory
   enumeration), `DOWNLOAD_COOKIE_FILES` (comma-separated list — one session per parallel
   download worker), and `OUT_DIR`.
2. Recursively enumerates every file the session can see (`browse.php`), first — nothing
   downloads until the full file list and count are known, printed as `ENUMERATED N files`.
3. Downloads with a `ThreadPoolExecutor(max_workers=len(DOWNLOAD_COOKIE_FILES))`, one cookie
   file per worker slot (deterministic `i % N` assignment, so a given worker always reuses the
   same session — this is what makes the parallelism real, verified by watching multiple
   `.partial` files grow concurrently).
4. **Every download writes to `<final_name>.partial` first, and only `os.replace()`s it to the
   final name on a verified-successful curl exit.** This means: the final filename existing is
   an absolute guarantee the file is complete. A `.partial` file left over from a killed run is
   never mistaken for a complete download — the only skip condition is
   `os.path.exists(final_path)`, and that path is only ever created by the atomic rename. Any
   stray `.partial` file is safe to delete by hand if you want to force a clean re-download of
   that one file; it will never silently corrupt anything either way.
5. Retries transient failures (`curl --fail`, 3 attempts with backoff) before giving up on a
   file and reporting `FAILED after N retries: <path>` (does not crash the whole run).
6. **Skips `.pcap` files entirely** (`SKIPPED_EXTENSIONS` in the script) — raw packet captures
   are huge (CICIoMT2024's PCAP tree alone was 23GB, most of the dataset's disk footprint) and
   there's no packet-parsing tooling (scapy/tshark) in this project yet to make them useful.
   Only the already-feature-extracted CSVs (and small PDFs like README files) are pulled. If
   PCAPs are needed later, remove/extend `SKIPPED_EXTENSIONS`.

## How to resume

The crawler is fully idempotent — just re-run it. It will re-enumerate (a few seconds), report
how many files already exist, and only download what's missing:

```bash
python3 download_datasets/cic_crawler.py \
  "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/" \
  download_datasets/cic_iotdiad_cookies.txt \
  "download_datasets/cic_iotdiad_cookies_1.txt,download_datasets/cic_iotdiad_cookies_2.txt,download_datasets/cic_iotdiad_cookies_3.txt" \
  data/raw/CIC_IoT_DIAD_2024 \
  > /tmp/cic_iotdiad_download.log 2>&1 &

python3 download_datasets/cic_crawler.py \
  "https://cicresearch.ca/IOTDataset/CICIoMT2024/" \
  download_datasets/cic_iomt_cookies.txt \
  "download_datasets/cic_iomt_cookies_1.txt,download_datasets/cic_iomt_cookies_2.txt,download_datasets/cic_iomt_cookies_3.txt" \
  data/raw/CICIoMT2024 \
  > /tmp/cic_iomt_download.log 2>&1 &
```

Start the two `nohup`'d background jobs a few seconds apart (avoid a connection burst — the
server rate-limited us hard once when both fired simultaneously with too much concurrency; see
"Lessons learned" below).

Watch progress:
```bash
tail -f /tmp/cic_iotdiad_download.log   # or /tmp/cic_iomt_download.log
grep -c "^\[" /tmp/cic_iotdiad_download.log   # files attempted so far
```

## If the session cookies expire or get rate-limited

The cookies in this folder (`*_cookies*.txt`) are registered sessions from 2026-08-12/13 —
they may eventually expire or get throttled by the server. To register fresh ones:

```bash
for i in 1 2 3; do
  curl -s -c download_datasets/cic_iotdiad_cookies_$i.txt -b download_datasets/cic_iotdiad_cookies_$i.txt \
    -X POST "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/insert.php" \
    -F "first_name=<name>" -F "last_name=<name>" -F "email=<email>" \
    -F "institution=<org>" -F "job_title=<title>" -F "country=<country>"
done
# repeat for CICIoMT2024 with the CICIoMT2024 base URL
```

Then relaunch the crawler with the new cookie files — already-downloaded files are untouched
(skip-if-exists still applies regardless of which session downloads the rest).

## Lessons learned (why the script looks like this)

- **HEAD/Range requests hang** on this server — the PHP script doesn't handle them, so there's
  no way to get file sizes up front without downloading. The crawler reports file *count*
  (from enumeration) as the only reliable upfront progress signal, not total bytes.
- **The `⬅ Up` link in directory listings matches the same regex as real subdirectory links** —
  an earlier version of this script naively followed it and infinite-looped back to the site
  root instead of ever reaching leaf files. Fixed by only descending into paths that are
  actual children of the current directory (`p.startswith(rel_path + "/")`).
- **Aggressive concurrency (10 total connections across both datasets) got the whole session
  rate-limited** — even plain directory listings started hanging for 60-90s. If this happens
  again: kill everything, wait ~90s, and confirm the server responds to one plain `curl` before
  resuming.
- Gotham's `raw/` PCAP tree (30GB, 198 files, mostly multi-GB captures) was deliberately left
  compressed inside the zip — only `processed/` (6.5GB, feature CSVs) and `README.md` were
  extracted, since there's no packet-parsing dependency (scapy/tshark) in this project yet to
  make raw PCAPs useful. Extract the rest with:
  `unzip data/raw/Gotham2025/GothamDataset2025.zip "raw/*" -d data/raw/Gotham2025/extracted`
  if/when that tooling gets added.
