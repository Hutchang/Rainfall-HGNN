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

## Radar sample (D13.2, TEST02)

`radar/71_20220103.prcp-m15.zip` — two real 15-minute BoM Rainfields3 `prcp-m15` files
for site 71 (T_Hills), stored exactly as the NCI THREDDS archive returns them (byte for
byte, from the `71_20220103.prcp-m15.zip` day). Two of the day's 96 files, enough for
`src/canonical/radar_archive.py`'s per-file decode and coordinate checks
(`tests/test_radar_archive.py`) but not a complete four-file hour — the four/three-file
summing behaviour is tested against synthetic fixtures instead (TEST03).

## Satellite sample (D13.5, TEST05)

`satellite/S_NWC_CRRPh_HIMA08_HIMA-N-NR_20220103T*.nc` — six real, consecutive
10-minute Himawari CRRPh files (one complete hour, 2022-01-03 00:00-01:00 UTC),
confirmed against the real NCI `rv74` archive via its public THREDDS/OPeNDAP service.
Unlike the radar sample, these are not byte-identical to the server's files: a real
`crrph` file is the full 5500x5500 Himawari disk across six variables (~18.6 MB), far
too large to commit, so each file was cropped to the Sydney window
(`config/canonical/grids.yaml`'s `layers.satellite.crop` box, 62x76 cells) and reduced
to the three variables the reader actually uses (`crrph_intensity`, `nx`, `ny`,
`geostationary`) before saving — real, unmodified values within that window, at
~25 KB/file. See [SATELLITE_SOURCE.md](../../../docs/grids/SATELLITE_SOURCE.md) for the
product itself. Six files is a complete hour, so
`src/canonical/satellite_archive.py`'s full pipeline (crop, average, subsample) runs
successfully against this sample, unlike radar's incomplete one.
