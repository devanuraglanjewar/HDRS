"""Small, optional Supabase REST adapter for benchmark reports.

The backend uses the service-role key because benchmark writes are trusted
server operations. If Supabase is not configured or is temporarily unavailable,
callers keep using the existing local JSON files.
"""

import logging
from typing import Any, Optional

import requests

from src.core.settings import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)


class SupabaseReportStore:
    def __init__(self, url: str = SUPABASE_URL, service_role_key: str = SUPABASE_SERVICE_ROLE_KEY):
        self.url = (url or "").rstrip("/")
        self.service_role_key = service_role_key or ""
        self.table_url = f"{self.url}/rest/v1/benchmark_reports" if self.url else ""

    @property
    def configured(self) -> bool:
        return bool(self.url and self.service_role_key)

    def _headers(self) -> dict:
        return {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def save(self, summary: dict, dataset: list) -> Optional[dict]:
        if not self.configured:
            return None
        payload = {
            "benchmark_scope": summary.get("benchmark_scope"),
            "mode": summary.get("mode"),
            "run_timestamp": summary.get("timestamp"),
            "summary": summary,
            "dataset": dataset,
        }
        try:
            response = requests.post(self.table_url, headers=self._headers(), json=payload, timeout=(3, 15))
            response.raise_for_status()
            rows = response.json()
            return rows[0] if isinstance(rows, list) and rows else payload
        except Exception as error:
            logger.warning("Supabase benchmark report upload failed; local report retained: %s", error)
            return None

    def latest(self, source_filter: Optional[str] = None) -> Optional[dict]:
        if not self.configured:
            return None
        params = {"select": "*", "order": "created_at.desc", "limit": "1"}
        if source_filter:
            params["benchmark_scope"] = f"eq.{source_filter}"
        try:
            response = requests.get(self.table_url, headers=self._headers(), params=params, timeout=(3, 10))
            response.raise_for_status()
            rows = response.json()
            return rows[0] if isinstance(rows, list) and rows else None
        except Exception as error:
            logger.warning("Supabase benchmark report read failed; local report will be used: %s", error)
            return None

    def delete_latest(self, source_filter: Optional[str] = None) -> bool:
        if not self.configured:
            return False
        latest = self.latest(source_filter)
        report_id = latest.get("id") if latest else None
        if not report_id:
            return False
        try:
            response = requests.delete(
                self.table_url,
                headers=self._headers(),
                params={"id": f"eq.{report_id}"},
                timeout=(3, 10),
            )
            response.raise_for_status()
            return True
        except Exception as error:
            logger.warning("Supabase benchmark report delete failed: %s", error)
            return False

    def save_document(self, source_name: str, text: str, chunks_count: int, metadata: Optional[dict] = None) -> Optional[dict]:
        if not self.configured:
            return None
        # PostgreSQL text fields reject NUL bytes occasionally emitted by PDF
        # extraction libraries. Remove only that invalid control character;
        # preserve all other extracted content.
        text = (text or "").replace("\x00", "")
        payload = {
            "source_name": source_name,
            "full_text": text,
            "character_count": len(text or ""),
            "chunks_count": chunks_count,
            "metadata": metadata or {},
        }
        headers = {**self._headers(), "Prefer": "resolution=merge-duplicates,return=representation"}
        try:
            response = requests.post(
                f"{self.url}/rest/v1/documents",
                headers=headers,
                json=payload,
                params={"on_conflict": "source_name"},
                timeout=(3, 15),
            )
            response.raise_for_status()
            rows = response.json()
            return rows[0] if isinstance(rows, list) and rows else payload
        except Exception as error:
            logger.warning("Supabase document upload failed; vector index remains available: %s", error)
            return None

    def list_documents(self) -> list:
        if not self.configured:
            return []
        try:
            response = requests.get(
                f"{self.url}/rest/v1/documents",
                headers=self._headers(),
                params={"select": "*", "order": "updated_at.desc"},
                timeout=(3, 10),
            )
            response.raise_for_status()
            return response.json() or []
        except Exception as error:
            logger.warning("Supabase document read failed; vector index will be used: %s", error)
            return []

    def delete_document(self, source_name: str) -> bool:
        if not self.configured:
            return False
        try:
            response = requests.delete(
                f"{self.url}/rest/v1/documents",
                headers=self._headers(),
                params={"source_name": f"eq.{source_name}"},
                timeout=(3, 10),
            )
            response.raise_for_status()
            return True
        except Exception as error:
            logger.warning("Supabase document delete failed: %s", error)
            return False

    def clear_documents(self) -> bool:
        if not self.configured:
            return False
        try:
            response = requests.delete(
                f"{self.url}/rest/v1/documents",
                headers=self._headers(),
                params={"id": "not.is.null"},
                timeout=(3, 10),
            )
            response.raise_for_status()
            return True
        except Exception as error:
            logger.warning("Supabase document clear failed: %s", error)
            return False

    def save_detector_metrics(self, metrics: dict) -> Optional[dict]:
        if not self.configured:
            return None
        try:
            response = requests.post(
                f"{self.url}/rest/v1/detector_metrics",
                headers=self._headers(),
                json={"metrics": metrics},
                timeout=(3, 15),
            )
            response.raise_for_status()
            rows = response.json()
            return rows[0] if isinstance(rows, list) and rows else metrics
        except Exception as error:
            logger.warning("Supabase detector metrics upload failed; local metrics retained: %s", error)
            return None

    def latest_detector_metrics(self) -> Optional[dict]:
        if not self.configured:
            return None
        try:
            response = requests.get(
                f"{self.url}/rest/v1/detector_metrics",
                headers=self._headers(),
                params={"select": "metrics,created_at", "order": "created_at.desc", "limit": "1"},
                timeout=(3, 10),
            )
            response.raise_for_status()
            rows = response.json()
            return rows[0].get("metrics") if rows else None
        except Exception as error:
            logger.warning("Supabase detector metrics read failed; local metrics will be used: %s", error)
            return None


supabase_report_store = SupabaseReportStore()
