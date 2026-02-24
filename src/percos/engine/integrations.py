"""External integrations – Calendar, Email, Task adapters (§7.2 – GAP-L1).

Provides pluggable adapters that convert external events into
``RawEvent`` objects for the ingestion pipeline.

Each adapter implements ``fetch_events()`` which returns a list of
``RawEvent`` ready for ``Brain.ingest_event()``.

Configuration is via environment variables or ``Settings`` fields.
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from percos.logging import get_logger
from percos.models.enums import EventType
from percos.models.events import RawEvent

log = get_logger("integrations")


# ── Base Adapter ───────────────────────────────────────

class ExternalAdapter(ABC):
    """Base class for external service adapters."""

    name: str = "base"

    @abstractmethod
    async def fetch_events(
        self,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[RawEvent]:
        """Fetch new events from the external service.

        Args:
            since: Only fetch events after this timestamp (incremental sync).
            limit: Maximum number of events to return.

        Returns:
            List of ``RawEvent`` objects ready for ingestion.
        """
        ...

    @abstractmethod
    async def test_connection(self) -> dict[str, Any]:
        """Verify the adapter can connect to its service.

        Returns:
            Dict with ``connected: bool`` and optional ``error`` message.
        """
        ...


# ── Google Calendar Adapter ────────────────────────────

class GoogleCalendarAdapter(ExternalAdapter):
    """Fetch events from Google Calendar via the Calendar API.

    Requires ``GOOGLE_CALENDAR_CREDENTIALS`` env var pointing to a
    service-account JSON file, and ``GOOGLE_CALENDAR_ID`` for the
    calendar to sync.

    Uses the ``google-api-python-client`` and ``google-auth`` packages
    (install separately: ``pip install google-api-python-client google-auth``).
    """

    name = "google_calendar"

    def __init__(
        self,
        credentials_path: str | None = None,
        calendar_id: str = "primary",
    ):
        self._credentials_path = credentials_path
        self._calendar_id = calendar_id
        self._service = None

    def _get_service(self):
        """Lazy-build the Google Calendar API service client."""
        if self._service is not None:
            return self._service
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds = service_account.Credentials.from_service_account_file(
                self._credentials_path,
                scopes=["https://www.googleapis.com/auth/calendar.readonly"],
            )
            self._service = build("calendar", "v3", credentials=creds)
            return self._service
        except ImportError:
            raise RuntimeError(
                "Google Calendar integration requires: "
                "pip install google-api-python-client google-auth"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Google Calendar: {exc}")

    async def test_connection(self) -> dict[str, Any]:
        try:
            svc = self._get_service()
            # Attempt a minimal API call (blocking → offload to thread)
            await asyncio.to_thread(
                lambda: svc.calendarList().list(maxResults=1).execute()
            )
            return {"connected": True, "calendar_id": self._calendar_id}
        except Exception as exc:
            return {"connected": False, "error": str(exc)}

    async def fetch_events(
        self,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[RawEvent]:
        svc = self._get_service()
        time_min = (since or datetime.now(tz=timezone.utc)).isoformat()
        try:
            result = await asyncio.to_thread(
                lambda: svc.events().list(
                    calendarId=self._calendar_id,
                    timeMin=time_min,
                    maxResults=limit,
                    singleEvents=True,
                    orderBy="startTime",
                ).execute()
            )
        except Exception as exc:
            log.error("google_calendar_fetch_failed", error=str(exc))
            return []

        events: list[RawEvent] = []
        for item in result.get("items", []):
            summary = item.get("summary", "Untitled event")
            start = item.get("start", {}).get("dateTime", item.get("start", {}).get("date", ""))
            end = item.get("end", {}).get("dateTime", item.get("end", {}).get("date", ""))
            content = (
                f"Calendar event: {summary}\n"
                f"Start: {start}\n"
                f"End: {end}\n"
                f"Location: {item.get('location', '')}\n"
                f"Description: {item.get('description', '')}"
            )
            events.append(RawEvent(
                event_type=EventType.EXTERNAL,
                source="google_calendar",
                content=content,
                metadata_extra={
                    "calendar_id": self._calendar_id,
                    "google_event_id": item.get("id", ""),
                    "summary": summary,
                    "start": start,
                    "end": end,
                    "location": item.get("location", ""),
                    "attendees": [
                        a.get("email", "") for a in item.get("attendees", [])
                    ],
                    "recurrence": item.get("recurrence", []),
                },
            ))

        log.info("google_calendar_fetched", events=len(events))
        return events


# ── IMAP Email Adapter ─────────────────────────────────

class IMAPEmailAdapter(ExternalAdapter):
    """Fetch emails via IMAP.

    Requires ``IMAP_HOST``, ``IMAP_USER``, ``IMAP_PASSWORD`` env vars.
    Optionally ``IMAP_PORT`` (default 993) and ``IMAP_FOLDER`` (default INBOX).
    """

    name = "imap_email"

    def __init__(
        self,
        host: str = "",
        port: int = 993,
        user: str = "",
        password: str = "",
        folder: str = "INBOX",
        use_ssl: bool = True,
    ):
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._folder = folder
        self._use_ssl = use_ssl

    async def test_connection(self) -> dict[str, Any]:
        import imaplib
        try:
            def _test():
                if self._use_ssl:
                    conn = imaplib.IMAP4_SSL(self._host, self._port)
                else:
                    conn = imaplib.IMAP4(self._host, self._port)
                conn.login(self._user, self._password)
                conn.logout()
            await asyncio.to_thread(_test)
            return {"connected": True, "host": self._host, "folder": self._folder}
        except Exception as exc:
            return {"connected": False, "error": str(exc)}

    async def fetch_events(
        self,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[RawEvent]:
        import imaplib
        import email as email_lib
        from email.header import decode_header

        def _fetch_sync() -> list[RawEvent]:
            try:
                if self._use_ssl:
                    conn = imaplib.IMAP4_SSL(self._host, self._port)
                else:
                    conn = imaplib.IMAP4(self._host, self._port)
                conn.login(self._user, self._password)
                conn.select(self._folder)
            except Exception as exc:
                log.error("imap_connect_failed", error=str(exc))
                return []

            # Search criteria
            criteria = "ALL"
            if since:
                date_str = since.strftime("%d-%b-%Y")
                criteria = f'(SINCE {date_str})'

            try:
                _, msg_ids = conn.search(None, criteria)
                id_list = msg_ids[0].split()[-limit:]  # latest N
            except Exception as exc:
                log.error("imap_search_failed", error=str(exc))
                conn.logout()
                return []

            events: list[RawEvent] = []
            for msg_id in id_list:
                try:
                    _, data = conn.fetch(msg_id, "(RFC822)")
                    raw_email = data[0][1]  # type: ignore[index]
                    msg = email_lib.message_from_bytes(raw_email)  # type: ignore[arg-type]

                    subject = ""
                    raw_subject = msg.get("Subject", "")
                    if raw_subject:
                        decoded, encoding = decode_header(raw_subject)[0]
                        subject = decoded.decode(encoding or "utf-8") if isinstance(decoded, bytes) else str(decoded)

                    from_addr = msg.get("From", "")
                    date_str_val = msg.get("Date", "")
                    to_addr = msg.get("To", "")

                    # Get plain-text body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                payload = part.get_payload(decode=True)
                                if isinstance(payload, bytes):
                                    body = payload.decode("utf-8", errors="replace")
                                break
                    else:
                        payload = msg.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            body = payload.decode("utf-8", errors="replace")

                    content = (
                        f"Email from: {from_addr}\n"
                        f"To: {to_addr}\n"
                        f"Subject: {subject}\n"
                        f"Date: {date_str_val}\n\n"
                        f"{body[:3000]}"  # cap body length
                    )

                    events.append(RawEvent(
                        event_type=EventType.EXTERNAL,
                        source="imap_email",
                        content=content,
                        metadata_extra={
                            "email_from": from_addr,
                            "email_to": to_addr,
                            "subject": subject,
                            "date": date_str_val,
                            "message_id": msg.get("Message-ID", ""),
                        },
                    ))
                except Exception as exc:
                    log.warning("imap_message_parse_failed", msg_id=str(msg_id), error=str(exc))

            conn.logout()
            return events

        result = await asyncio.to_thread(_fetch_sync)
        log.info("imap_emails_fetched", count=len(result))
        return result


# ── Microsoft Outlook/Graph Adapter ────────────────────

class MicrosoftGraphAdapter(ExternalAdapter):
    """Fetch calendar events and emails from Microsoft 365 / Outlook via Graph API.

    Requires ``MS_GRAPH_CLIENT_ID``, ``MS_GRAPH_CLIENT_SECRET``,
    ``MS_GRAPH_TENANT_ID`` env vars.

    Uses the ``msal`` and ``requests`` packages
    (install: ``pip install msal requests``).
    """

    name = "microsoft_graph"

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        tenant_id: str = "",
        user_id: str = "me",
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._token: str | None = None

    def _get_token(self) -> str:
        """Acquire an access token using MSAL."""
        if self._token:
            return self._token
        try:
            import msal
            app = msal.ConfidentialClientApplication(
                self._client_id,
                authority=f"https://login.microsoftonline.com/{self._tenant_id}",
                client_credential=self._client_secret,
            )
            result = app.acquire_token_for_client(
                scopes=["https://graph.microsoft.com/.default"]
            )
            if result and "access_token" in result:
                self._token = str(result["access_token"])
                return self._token
            desc = result.get("error_description", "Token acquisition failed") if result else "No result"
            raise RuntimeError(desc)
        except ImportError:
            raise RuntimeError(
                "Microsoft Graph integration requires: pip install msal requests"
            )

    async def test_connection(self) -> dict[str, Any]:
        try:
            import requests
            token = self._get_token()
            resp = await asyncio.to_thread(
                lambda: requests.get(
                    f"https://graph.microsoft.com/v1.0/users/{self._user_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10,
                )
            )
            resp.raise_for_status()
            return {"connected": True, "user": resp.json().get("displayName", "")}
        except Exception as exc:
            return {"connected": False, "error": str(exc)}

    async def fetch_events(
        self,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[RawEvent]:
        try:
            import requests
        except ImportError:
            log.error("requests_not_installed")
            return []

        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        since_iso = (since or datetime.now(tz=timezone.utc)).isoformat()

        def _fetch_sync() -> list[RawEvent]:
            events: list[RawEvent] = []

            # Fetch calendar events
            try:
                url = (
                    f"https://graph.microsoft.com/v1.0/users/{self._user_id}"
                    f"/calendarView?startDateTime={since_iso}"
                    f"&endDateTime=9999-12-31T00:00:00Z&$top={limit}"
                )
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                for item in resp.json().get("value", []):
                    summary = item.get("subject", "Untitled")
                    start = item.get("start", {}).get("dateTime", "")
                    end = item.get("end", {}).get("dateTime", "")
                    content = (
                        f"Calendar event: {summary}\n"
                        f"Start: {start}\nEnd: {end}\n"
                        f"Location: {item.get('location', {}).get('displayName', '')}\n"
                        f"Body: {item.get('bodyPreview', '')}"
                    )
                    events.append(RawEvent(
                        event_type=EventType.EXTERNAL,
                        source="microsoft_graph",
                        content=content,
                        metadata_extra={
                            "ms_event_id": item.get("id", ""),
                            "summary": summary,
                            "start": start,
                            "end": end,
                            "organizer": item.get("organizer", {}).get("emailAddress", {}).get("address", ""),
                        },
                    ))
            except Exception as exc:
                log.error("ms_graph_calendar_failed", error=str(exc))

            # Fetch recent emails
            try:
                url = (
                    f"https://graph.microsoft.com/v1.0/users/{self._user_id}"
                    f"/messages?$top={limit}&$orderby=receivedDateTime desc"
                )
                if since:
                    url += f"&$filter=receivedDateTime ge {since_iso}"
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                for item in resp.json().get("value", []):
                    subject = item.get("subject", "")
                    from_addr = item.get("from", {}).get("emailAddress", {}).get("address", "")
                    body = item.get("bodyPreview", "")
                    content = (
                        f"Email from: {from_addr}\n"
                        f"Subject: {subject}\n"
                        f"Date: {item.get('receivedDateTime', '')}\n\n"
                        f"{body[:3000]}"
                    )
                    events.append(RawEvent(
                        event_type=EventType.EXTERNAL,
                        source="microsoft_graph",
                        content=content,
                        metadata_extra={
                            "ms_message_id": item.get("id", ""),
                            "email_from": from_addr,
                            "subject": subject,
                            "date": item.get("receivedDateTime", ""),
                        },
                    ))
            except Exception as exc:
                log.error("ms_graph_email_failed", error=str(exc))

            return events

        result = await asyncio.to_thread(_fetch_sync)
        log.info("ms_graph_fetched", events=len(result))
        return result


# ── Integration Manager ────────────────────────────────

class IntegrationManager:
    """Manages external adapters and syncs their events into the Brain.

    Usage:
        mgr = IntegrationManager(brain)
        mgr.register(GoogleCalendarAdapter(creds_path="..."))
        await mgr.sync_all()
    """

    def __init__(self, brain):
        self._brain = brain
        self._adapters: dict[str, ExternalAdapter] = {}

    def register(self, adapter: ExternalAdapter) -> None:
        """Register an external adapter."""
        self._adapters[adapter.name] = adapter
        log.info("adapter_registered", name=adapter.name)

    def unregister(self, name: str) -> None:
        """Remove an adapter by name."""
        self._adapters.pop(name, None)

    def list_adapters(self) -> list[dict[str, str]]:
        """List all registered adapters."""
        return [{"name": a.name, "type": type(a).__name__} for a in self._adapters.values()]

    async def test_all(self) -> dict[str, dict]:
        """Test connectivity for all registered adapters."""
        results = {}
        for name, adapter in self._adapters.items():
            results[name] = await adapter.test_connection()
        return results

    async def sync(
        self,
        adapter_name: str,
        since: datetime | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Sync events from a specific adapter into the brain."""
        adapter = self._adapters.get(adapter_name)
        if not adapter:
            return {"error": f"Adapter '{adapter_name}' not found"}

        raw_events = await adapter.fetch_events(since=since, limit=limit)
        ingested_ids: list[str] = []
        for event in raw_events:
            try:
                eid = await self._brain.ingest_event(event)
                ingested_ids.append(eid)
            except Exception as exc:
                log.warning("sync_ingest_failed", adapter=adapter_name, error=str(exc))

        log.info("sync_complete", adapter=adapter_name, fetched=len(raw_events), ingested=len(ingested_ids))
        return {
            "adapter": adapter_name,
            "fetched": len(raw_events),
            "ingested": len(ingested_ids),
            "event_ids": ingested_ids,
        }

    async def sync_all(
        self,
        since: datetime | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Sync events from all registered adapters."""
        results = {}
        for name in self._adapters:
            results[name] = await self.sync(name, since=since, limit=limit)
        return results
