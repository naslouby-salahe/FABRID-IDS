#!/bin/bash
# Quick status check for the two CIC dataset downloads.
set -euo pipefail

echo "=== CIC IoT-DIAD 2024 ==="
grep "ENUMERATED\|ALREADY_COMPLETE" /tmp/cic_iotdiad_download.log 2>/dev/null || echo "no log yet"
tail -3 /tmp/cic_iotdiad_download.log 2>/dev/null
echo

echo "=== CICIoMT2024 ==="
grep "ENUMERATED\|ALREADY_COMPLETE" /tmp/cic_iomt_download.log 2>/dev/null || echo "no log yet"
tail -3 /tmp/cic_iomt_download.log 2>/dev/null
echo

echo "=== disk usage ==="
du -sh data/raw/CIC_IoT_DIAD_2024/ data/raw/CICIoMT2024/ data/raw/Gotham2025/ 2>/dev/null
