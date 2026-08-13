# Dataset acquisition and integrity

FABRID-IDS does not vendor raw datasets or maintain ad-hoc download scripts in the repository root. The experiment pipeline receives an explicit raw-data root and dataset readers own only parsing, partitioning, manifests, and scientific eligibility.

## N-BaIoT

N-BaIoT is the primary experiment dataset. Store the extracted device directories under the raw-data root expected by the N-BaIoT reader. Physical device identity is preserved as the federation client identity.

## CIC IoT-DIAD 2024

CIC IoT-DIAD is an external-replication candidate. The CIC distribution site requires authenticated/session-based acquisition. Registration and credential/session handling are intentionally outside the FABRID application and must not be committed.

Acquire the feature CSVs into the raw-data root without modifying their source hierarchy. Download tools should write incomplete transfers to a temporary path and atomically rename only after a successful transfer. Raw endpoint identity columns remain provenance/partition fields and are excluded from detector features by the typed dataset specification.

## CICIoMT2024

CICIoMT2024 is used only where its typed capability and eligibility gates permit. Preserve the supplied profiling and attack directory hierarchy. The current FABRID reader consumes feature CSVs; raw PCAP acquisition is not required by the implemented experiment path.

## Gotham 2025

Gotham is an external/event-level candidate. Preserve the processed feature CSV hierarchy. Raw PCAP extraction is unnecessary unless a future roadmap revision introduces an explicit packet-parsing experiment and corresponding dependencies.

## Integrity validation

After acquisition or copying, verify every CSV through the single FABRID CLI:

```bash
fabrid validate-data /path/to/raw/dataset
```

The validator streams every CSV row, verifies field-count consistency, rejects zero-byte/header-only files, detects missing trailing newlines, and reports encoding/read failures as structured validation evidence. A non-zero CLI exit status means the dataset tree is not ready for scientific execution.

## Reproducibility boundary

Authentication cookies, registration credentials, temporary download logs, partial-transfer files, and one-off crawler scripts are operational secrets/state and must not be committed. Scientific reproducibility begins from the acquired raw-data tree plus dataset checksums/manifests recorded by the FABRID pipeline.
