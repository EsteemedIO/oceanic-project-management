"""Shared Vespa client for oceanic-project-management repositories.

Provides connection pooling and basic CRUD operations for Vespa document store.
Based on Echo's Vespa client patterns but simplified for project management use case.
"""

import logging
import os
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class VespaClient:
    """HTTP client for Vespa operations with connection pooling.

    Provides low-level CRUD operations for Vespa document store. All operations
    use httpx.Client for HTTP/2 support and connection pooling.

    Environment Variables:
        VESPA_HOST: Vespa endpoint URL (default: http://localhost:8081)
        VESPA_TIMEOUT: Request timeout in seconds (default: 30)

    Attributes:
        vespa_host: Base URL for Vespa application endpoint
        timeout: HTTP request timeout in seconds

    Example:
        >>> client = VespaClient()
        >>> client.feed_data_point("my_schema", "doc-123", {"title": "Test"})
        >>> data = client.get_data("my_schema", "doc-123")
    """

    def __init__(
        self,
        vespa_host: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        """Initialize Vespa client with connection settings.

        Args:
            vespa_host: Vespa endpoint URL (overrides VESPA_HOST env var)
            timeout: Request timeout in seconds (default: 30)
        """
        self.vespa_host = vespa_host or os.getenv("VESPA_HOST", "http://localhost:8081")
        self.timeout = timeout
        logger.info(
            "vespa_client_initialized",
            extra={"vespa_host": self.vespa_host, "timeout": timeout}
        )

    def _get_client(self) -> httpx.Client:
        """Create httpx client with HTTP/2 support and timeouts.

        Returns:
            Configured httpx.Client instance
        """
        return httpx.Client(
            timeout=self.timeout,
            http2=True,
            verify=False,  # For local development; enable for production
        )

    def _document_url(self, schema_name: str, document_id: str) -> str:
        """Construct Vespa document API URL.

        Args:
            schema_name: Vespa schema name (e.g., "oceanic_project")
            document_id: Document identifier

        Returns:
            Full document URL for CRUD operations
        """
        # Vespa document API: /document/v1/{namespace}/{document-type}/docid/{docid}
        return f"{self.vespa_host}/document/v1/oceanic/{schema_name}/docid/{document_id}"

    def _search_url(self) -> str:
        """Construct Vespa search API URL.

        Returns:
            Search endpoint URL
        """
        return f"{self.vespa_host}/search/"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def feed_data_point(
        self,
        schema_name: str,
        document_id: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Insert or replace a document in Vespa.

        Uses PUT with full document replacement. Retries on transient failures.

        Args:
            schema_name: Vespa schema name
            document_id: Unique document identifier
            fields: Document fields as key-value pairs

        Returns:
            Vespa response as dictionary

        Raises:
            httpx.HTTPStatusError: If request fails after retries

        Example:
            >>> client.feed_data_point(
            ...     "oceanic_project",
            ...     "proj_123",
            ...     {"name": "Test Project", "org_id": "org_456"}
            ... )
        """
        url = self._document_url(schema_name, document_id)

        with self._get_client() as client:
            logger.debug(
                "vespa_feed_document",
                extra={
                    "schema": schema_name,
                    "document_id": document_id,
                    "url": url,
                }
            )

            response = client.put(
                url,
                json={"fields": fields},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            logger.info(
                "vespa_document_fed",
                extra={
                    "schema": schema_name,
                    "document_id": document_id,
                    "status": response.status_code,
                }
            )

            return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def get_data(
        self,
        schema_name: str,
        document_id: str,
    ) -> Optional[dict[str, Any]]:
        """Retrieve a document from Vespa by ID.

        Args:
            schema_name: Vespa schema name
            document_id: Document identifier

        Returns:
            Document fields or None if not found

        Raises:
            httpx.HTTPStatusError: If request fails (except 404)

        Example:
            >>> data = client.get_data("oceanic_project", "proj_123")
            >>> if data:
            ...     print(data["name"])
        """
        url = self._document_url(schema_name, document_id)

        with self._get_client() as client:
            logger.debug(
                "vespa_get_document",
                extra={"schema": schema_name, "document_id": document_id}
            )

            response = client.get(url)

            if response.status_code == 404:
                logger.debug(
                    "vespa_document_not_found",
                    extra={"schema": schema_name, "document_id": document_id}
                )
                return None

            response.raise_for_status()

            result = response.json()
            return result.get("fields", {})

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def update_data(
        self,
        schema_name: str,
        document_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Partially update a document in Vespa.

        Uses Vespa's update API with "assign" operation for each field.

        Args:
            schema_name: Vespa schema name
            document_id: Document identifier
            updates: Fields to update with new values

        Returns:
            Vespa response as dictionary

        Raises:
            httpx.HTTPStatusError: If request fails after retries

        Example:
            >>> client.update_data(
            ...     "oceanic_project",
            ...     "proj_123",
            ...     {"status": "active", "updated_at": "2025-12-12T00:00:00Z"}
            ... )
        """
        url = self._document_url(schema_name, document_id)

        # Convert updates to Vespa update format: {"field": {"assign": value}}
        update_fields = {
            field: {"assign": value}
            for field, value in updates.items()
        }

        with self._get_client() as client:
            logger.debug(
                "vespa_update_document",
                extra={
                    "schema": schema_name,
                    "document_id": document_id,
                    "fields": list(updates.keys()),
                }
            )

            response = client.put(
                url,
                json={"fields": update_fields},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            logger.info(
                "vespa_document_updated",
                extra={
                    "schema": schema_name,
                    "document_id": document_id,
                    "status": response.status_code,
                }
            )

            return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def delete_data(
        self,
        schema_name: str,
        document_id: str,
    ) -> bool:
        """Delete a document from Vespa.

        Args:
            schema_name: Vespa schema name
            document_id: Document identifier

        Returns:
            True if deleted, False if not found

        Raises:
            httpx.HTTPStatusError: If request fails (except 404)

        Example:
            >>> deleted = client.delete_data("oceanic_project", "proj_123")
        """
        url = self._document_url(schema_name, document_id)

        with self._get_client() as client:
            logger.debug(
                "vespa_delete_document",
                extra={"schema": schema_name, "document_id": document_id}
            )

            response = client.delete(url)

            if response.status_code == 404:
                logger.warning(
                    "vespa_document_not_found_for_delete",
                    extra={"schema": schema_name, "document_id": document_id}
                )
                return False

            response.raise_for_status()

            logger.info(
                "vespa_document_deleted",
                extra={
                    "schema": schema_name,
                    "document_id": document_id,
                    "status": response.status_code,
                }
            )

            return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def query(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a YQL query against Vespa search API.

        Args:
            params: Query parameters including 'yql', 'hits', 'offset', etc.

        Returns:
            Search results as dictionary with 'root' containing hits

        Raises:
            httpx.HTTPStatusError: If query fails after retries

        Example:
            >>> results = client.query({
            ...     "yql": "select * from oceanic_project where org_id='org_123'",
            ...     "hits": 10,
            ...     "offset": 0,
            ... })
            >>> for hit in results["root"]["children"]:
            ...     print(hit["fields"]["name"])
        """
        url = self._search_url()

        with self._get_client() as client:
            logger.debug(
                "vespa_query",
                extra={"params": params}
            )

            response = client.get(url, params=params)
            response.raise_for_status()

            result = response.json()
            hit_count = len(result.get("root", {}).get("children", []))

            logger.info(
                "vespa_query_completed",
                extra={
                    "hits_returned": hit_count,
                    "status": response.status_code,
                }
            )

            return result
