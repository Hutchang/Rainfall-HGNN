# Component 01 test sample

Real BoM Water Data Online responses, stored exactly as returned by
`src/ingestion/gauge/fetch.py` on 2026-09-05, for one Greater Sydney station:

| File | Station | Series | Window |
|---|---|---|---|
| `gauge/tips/31531755/2022-01.json.gz` | 31531755 ABBOTSBURY (public no. 567169) | `DMQaQc.Merged.AsStored.1` | 2022-01-01 to 2022-01-31 |
| `gauge/daily/31531755/2022.json.gz` | same | `DMQaQc.Merged.DailyTotal.09HR` | 2022-01-01 to 2022-12-31 |

Used by `tests/test_ingestion_gauge.py` and available to Component 02's tests so they
run on a fresh clone with no network. Regenerate with the fetcher and copy the files
here if the layout changes.
