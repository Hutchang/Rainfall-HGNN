"""Thin client for BoM Water Data Online (KiWIS query services).

Three requests, matching docs/requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md:
getStationList, getTimeseriesList, getTimeseriesValues. Every function returns the
server's response bytes untouched alongside the parsed JSON, so callers can store the
raw bytes and only use the parsed form for counting and bookkeeping.

Verified against the live API on 2026-09-05: bbox filtering, both series names, and
the station 568163 storm check from the requirements doc.
"""

import time

import httpx

COMMON = {
    "service": "kisters",
    "type": "queryServices",
    "datasource": "0",
    "format": "json",
}

STATION_FIELDS = "station_id,station_no,station_name,station_latitude,station_longitude"
SERIES_FIELDS = "station_id,ts_id,ts_name,ts_unitname,ts_spacing,coverage"
VALUE_FIELDS = "Timestamp,Value,Quality Code"


class KiwisClient:
    def __init__(
        self,
        base_url: str,
        user_agent: str,
        timeout_seconds: int = 120,
        request_delay_seconds: float = 1.0,
        retries: int = 3,
    ):
        self.base_url = base_url
        self.timeout = timeout_seconds
        self.delay = request_delay_seconds
        self.retries = retries
        self._http = httpx.Client(
            headers={"User-Agent": user_agent}, timeout=timeout_seconds
        )

    def _get(self, request: str, **params) -> tuple[bytes, object]:
        """One KiWIS request. Returns (raw_bytes, parsed_json). Retries transient errors
        with exponential backoff; never follows redirects, so a moved API fails loudly."""
        q = dict(COMMON, request=request, **params)
        last_error = None
        for attempt in range(self.retries):
            try:
                r = self._http.get(self.base_url, params=q)
                r.raise_for_status()
                time.sleep(self.delay)
                return r.content, r.json()
            except (httpx.HTTPError, ValueError) as e:
                last_error = e
                time.sleep(2**attempt)
        raise RuntimeError(
            f"{request} failed after {self.retries} attempts: {last_error}"
        )

    # ---- the three calls ---------------------------------------------------

    def station_list(
        self, bounds: dict | None = None, station_no: str | None = None
    ) -> tuple[bytes, list[dict]]:
        """Rainfall stations, optionally inside a bounding box or by public number.
        Returns raw bytes and rows as dicts keyed by STATION_FIELDS."""
        params = {"parametertype_name": "Rainfall", "returnfields": STATION_FIELDS}
        if station_no:
            params["station_no"] = station_no
        if bounds:
            # KiWIS bbox order is min_lon,min_lat,max_lon,max_lat
            params["bbox"] = ",".join(
                str(bounds[k]) for k in ("min_lon", "min_lat", "max_lon", "max_lat")
            )
        raw, payload = self._get("getStationList", **params)
        return raw, _rows_as_dicts(payload)

    def series_list(
        self, station_id: str | None = None, ts_name: str | None = None
    ) -> tuple[bytes, list[dict]]:
        """Rainfall series, filtered by internal station_id and/or exact series name.
        With neither filter this is the national list, one row per station-series."""
        params = {"parametertype_name": "Rainfall", "returnfields": SERIES_FIELDS}
        if station_id:
            params["station_id"] = station_id
        if ts_name:
            params["ts_name"] = ts_name
        raw, payload = self._get("getTimeseriesList", **params)
        return raw, _rows_as_dicts(payload)

    def values(self, ts_id: str, start: str, end: str) -> tuple[bytes, int]:
        """Records for one series between two YYYY-MM-DD dates.
        Returns raw bytes and the record count (the only thing 01 reads from the body)."""
        raw, payload = self._get(
            "getTimeseriesValues",
            ts_id=ts_id,
            returnfields=VALUE_FIELDS,
            **{"from": start, "to": end},
        )
        if not (isinstance(payload, list) and payload and "data" in payload[0]):
            raise RuntimeError(
                f"unexpected values response for ts_id {ts_id}: {str(payload)[:200]}"
            )
        return raw, len(payload[0]["data"])


def _rows_as_dicts(payload) -> list[dict]:
    """List endpoints return [[header...], [row...], ...]. Zip each row with the header."""
    if not (isinstance(payload, list) and payload and isinstance(payload[0], list)):
        raise RuntimeError(f"unexpected list response: {str(payload)[:200]}")
    header, *rows = payload
    return [dict(zip(header, r)) for r in rows]
