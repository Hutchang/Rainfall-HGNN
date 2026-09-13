"""Thin client for the NCI THREDDS radar archive (project `rq0`, BoM Rainfields 3).

One function: download one site's one calendar day as a zip of NetCDF files, exactly
as the server returns it. The URL is deterministic — no catalog listing is needed for
normal operation:

    <base_url>/<site_id>/<year>/<product>/<site_id>_<YYYYMMDD>.<product>.zip

Verified against the live server on 2026-09-13 for site 71 (T_Hills, the Sydney
radar), product `prcp-m15` ("Merged Gauge/Radar Accumulation (15 min)").
"""

import time
from datetime import date

import httpx


class NciClient:
    """A small HTTP client for one BoM Rainfields3 radar site's daily archive on
    NCI THREDDS. Holds the connection settings (base URL, timeout, retry count,
    politeness delay) so callers don't repeat them on every `daily_zip` call."""

    def __init__(
        self,
        base_url: str,
        user_agent: str,
        timeout_seconds: int = 120,
        request_delay_seconds: float = 1.0,
        retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.delay = request_delay_seconds
        self.retries = retries
        self._http = httpx.Client(
            headers={"User-Agent": user_agent}, timeout=timeout_seconds
        )

    def daily_zip(self, site_id: str, product: str, day: date) -> bytes:
        """One day's zip of NetCDF files for one site and product, as the server
        returned it. Retries transient errors with exponential backoff; never follows
        redirects, so a moved archive fails loudly instead of silently."""
        ymd = day.strftime("%Y%m%d")
        url = f"{self.base_url}/{site_id}/{day.year}/{product}/{site_id}_{ymd}.{product}.zip"
        last_error = None
        for attempt in range(self.retries):
            try:
                r = self._http.get(url)
                r.raise_for_status()
                time.sleep(self.delay)
                return r.content
            except httpx.HTTPError as e:
                last_error = e
                time.sleep(2**attempt)
        raise RuntimeError(f"{url} failed after {self.retries} attempts: {last_error}")
