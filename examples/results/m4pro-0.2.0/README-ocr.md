# Note on `ocr-agents.txt` and `ocr-agents-compare.txt` (added 5 November 2026)

These raw files are kept as recorded. They were made with the 0.2.0 `ocr_agents.py`, which always exited 0 and never
printed a FAIL line, so `summary.txt` counts both demos as PASS although they were not clean:

- `ocr-agents.txt`: the invoice request failed (`invoice: ERROR error`, an HTTP error from the server), so 15 of 21
  fields were extracted; the other three documents were 15/15.
- `ocr-agents-compare.txt`: the receipt request failed in 4 of its 9 rounds (`{'receipt': 'error'}`); every round that
  completed scored all its fields.

`ocr_agents.py` now prints a `PASS`/`FAIL` line per document and exits 1 on any failed request or wrong field, so
`run_all.sh` reports such a run as FAIL. The intermittent server error behind these rounds was not investigated here;
an M4 Max run on the same build family scored 21/21 with no errors
([../../../docs/CHANGES-m4max-report-2026-11-04.md](../../../docs/CHANGES-m4max-report-2026-11-04.md)).
