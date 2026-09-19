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


class DayNotInArchiveError(Exception):
    """The server says this site/product/day is not in the archive. Unlike a timeout
    or a 5xx, a rerun will not change that, so the caller records the day as
    `skipped` rather than `failed`."""


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
        # Not following redirects is httpx's default, but the contract above
        # promises it, so it is stated rather than inherited.
        self._http = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout_seconds,
            follow_redirects=False,
        )

    def daily_zip(self, site_id: str, product: str, day: date) -> bytes:
        """One day's zip of NetCDF files for one site and product, as the server
        returned it.

        Retries only what a retry can fix — timeouts, connection errors, 5xx and 429.
        A 404 means the day is not in the archive, which no amount of retrying will
        change, so it raises DayNotInArchiveError immediately; other 4xx fail loudly for
        the same reason. Never follows redirects, so a moved archive fails rather
        than silently succeeding somewhere else."""
        ymd = day.strftime("%Y%m%d")
        url = (
            f"{self.base_url}/{site_id}/{day.year}/{product}/"
            f"{site_id}_{ymd}.{product}.zip"
        )
        last_error = None
        for attempt in range(self.retries):
            try:
                r = self._http.get(url)
                r.raise_for_status()
                time.sleep(self.delay)
                return r.content
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code == 404:
                    raise DayNotInArchiveError(
                        f"not in archive (HTTP 404): {url}"
                    ) from e
                if 400 <= code < 500 and code != 429:
                    raise RuntimeError(f"{url}: HTTP {code}, not retried") from e
                last_error = e
            except httpx.HTTPError as e:
                last_error = e
            # No backoff after the final attempt; we are about to give up anyway.
            if attempt < self.retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError(f"{url} failed after {self.retries} attempts: {last_error}")

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._http.close()

    def __enter__(self) -> NciClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
