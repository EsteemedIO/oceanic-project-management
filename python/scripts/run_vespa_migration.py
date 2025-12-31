"""
Migrate data from Supabase to Vespa.

This script reads projects and tasks from Supabase and writes them to Vespa.
It uses direct HTTP calls to avoid complex dependency setup.

Usage:
    cd /packages/oceanic-project-management/python
    uv run python scripts/run_vespa_migration.py

    # Dry run (no writes to Vespa):
    uv run python scripts/run_vespa_migration.py --dry-run

Environment:
    SUPABASE_URL: Supabase project URL
    SUPABASE_KEY: Supabase service role key (or anon key)
    VESPA_HOST: Vespa application endpoint (default: http://localhost:8081)
    OPENAI_API_KEY: OpenAI API key for embeddings (optional)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "") or os.getenv("SUPABASE_SERVICE_KEY", "")
VESPA_HOST = os.getenv("VESPA_HOST", "http://localhost:8081")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ORG_ID = "default"

# Vespa namespace (must match deployed schemas)
VESPA_NAMESPACE = "danswer_index"


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.projects_total = 0
        self.projects_migrated = 0
        self.projects_skipped = 0
        self.tasks_total = 0
        self.tasks_migrated = 0
        self.tasks_skipped = 0
        self.errors: list[dict] = []

    def summary(self) -> str:
        return (
            f"Projects: {self.projects_migrated}/{self.projects_total} migrated, "
            f"{self.projects_skipped} skipped\n"
            f"Tasks: {self.tasks_migrated}/{self.tasks_total} migrated, "
            f"{self.tasks_skipped} skipped\n"
            f"Errors: {len(self.errors)}"
        )


def get_embedding(text: str) -> list[float]:
    """Generate embedding using OpenAI API."""
    if not OPENAI_API_KEY:
        # Return zero vector if no API key
        return [0.0] * 1536

    try:
        with httpx.Client() as client:
            response = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": text[:8000],  # Truncate to avoid token limit
                    "model": "text-embedding-ada-002",
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"  Warning: Embedding failed: {e}")
        return [0.0] * 1536


def fetch_supabase_projects() -> list[dict]:
    """Fetch all projects from Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set")
        sys.exit(1)

    url = f"{SUPABASE_URL}/rest/v1/archon_projects"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }

    with httpx.Client() as client:
        response = client.get(url, headers=headers, params={"select": "*"})
        response.raise_for_status()
        return response.json()


def fetch_supabase_tasks() -> list[dict]:
    """Fetch all tasks from Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set")
        sys.exit(1)

    url = f"{SUPABASE_URL}/rest/v1/archon_tasks"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }

    with httpx.Client() as client:
        response = client.get(url, headers=headers, params={"select": "*"})
        response.raise_for_status()
        return response.json()


def write_to_vespa(schema: str, doc_id: str, fields: dict, dry_run: bool = False) -> bool:
    """Write a document to Vespa using POST."""
    url = f"{VESPA_HOST}/document/v1/{VESPA_NAMESPACE}/{schema}/docid/{doc_id}"

    if dry_run:
        print(f"  [DRY RUN] Would write to {schema}: {doc_id}")
        return True

    try:
        with httpx.Client() as client:
            response = client.post(
                url,
                json={"fields": fields},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            return True
    except Exception as e:
        print(f"  Error writing {doc_id}: {e}")
        return False


def check_vespa_exists(schema: str, doc_id: str) -> bool:
    """Check if document exists in Vespa."""
    url = f"{VESPA_HOST}/document/v1/{VESPA_NAMESPACE}/{schema}/docid/{doc_id}"

    try:
        with httpx.Client() as client:
            response = client.get(url, timeout=10)
            return response.status_code == 200
    except:
        return False


def migrate_project(project: dict, stats: MigrationStats, dry_run: bool = False) -> bool:
    """Migrate a single project to Vespa."""
    project_id = project.get("id")
    title = project.get("title", "")

    # Check if already exists
    if not dry_run and check_vespa_exists("oceanic_project", project_id):
        print(f"  Skipping (exists): {title}")
        stats.projects_skipped += 1
        return True

    # Generate embedding from title + description
    description = project.get("description", "") or ""
    embed_text = f"{title} {description}"
    embedding = get_embedding(embed_text)

    # Map to Vespa schema fields
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Handle timestamps
    created_at = project.get("created_at", "")
    updated_at = project.get("updated_at", "")

    # Metadata must be JSON string (schema type is string, not map)
    metadata_obj = {
        "docs": project.get("docs", []),
        "features": project.get("features", []),
        "data": project.get("data", []),
        "source": "supabase_migration",
        "original_created_at": created_at,
        "original_updated_at": updated_at,
    }

    fields = {
        "project_id": project_id,
        "org_id": ORG_ID,
        "name": title,
        "description": description,
        "status": "active",
        "github_repo": project.get("github_repo", "") or "",
        "pinned": project.get("pinned", False) or False,
        "metadata": json.dumps(metadata_obj),  # Serialize to string
        "embedding": {"values": embedding},
        "created_at": now_ms,
        "updated_at": now_ms,
    }

    success = write_to_vespa("oceanic_project", project_id, fields, dry_run)
    if success:
        stats.projects_migrated += 1
        print(f"  Migrated: {title}")
    else:
        stats.errors.append({"type": "project", "id": project_id, "title": title})

    return success


def migrate_task(task: dict, stats: MigrationStats, dry_run: bool = False) -> bool:
    """Migrate a single task to Vespa."""
    task_id = task.get("id")
    title = task.get("title", "")

    # Check if already exists
    if not dry_run and check_vespa_exists("oceanic_task", task_id):
        print(f"  Skipping (exists): {title}")
        stats.tasks_skipped += 1
        return True

    # Generate embedding from title + description
    description = task.get("description", "") or ""
    embed_text = f"{title} {description}"
    embedding = get_embedding(embed_text)

    # Map to Vespa schema fields
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Priority is stored as string in Vespa schema
    priority_str = task.get("priority", "medium") or "medium"

    # Metadata must be JSON string (schema type is string, not map)
    metadata_obj = {
        "source": "supabase_migration",
    }

    fields = {
        "task_id": task_id,
        "project_id": task.get("project_id", ""),
        "org_id": ORG_ID,
        "title": title,
        "description": description,
        "status": task.get("status", "todo") or "todo",
        "priority": priority_str,
        "assignee": task.get("assignee", "") or "",
        "task_order": task.get("task_order", 0) or 0,
        "feature_id": task.get("feature", "") or "",  # Schema uses feature_id
        "tags": [],  # Required by schema
        "metadata": json.dumps(metadata_obj),  # Serialize to string
        "embedding": {"values": embedding},
        "created_at": now_ms,
        "updated_at": now_ms,
    }

    success = write_to_vespa("oceanic_task", task_id, fields, dry_run)
    if success:
        stats.tasks_migrated += 1
        print(f"  Migrated: {title}")
    else:
        stats.errors.append({"type": "task", "id": task_id, "title": title})

    return success


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate Supabase data to Vespa")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Vespa")
    args = parser.parse_args()

    print("=" * 60)
    print("Supabase → Vespa Migration")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No data will be written ***\n")

    # Check Vespa connection
    print(f"\nVespa endpoint: {VESPA_HOST}")
    try:
        with httpx.Client() as client:
            response = client.get(f"{VESPA_HOST}/ApplicationStatus", timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(f"Vespa status: OK (generation {data.get('application', {}).get('generation', 'N/A')})")
            else:
                print(f"Warning: Vespa returned status {response.status_code}")
    except Exception as e:
        print(f"Error: Cannot connect to Vespa: {e}")
        sys.exit(1)

    stats = MigrationStats()

    # Fetch and migrate projects
    print("\n--- Fetching projects from Supabase ---")
    try:
        projects = fetch_supabase_projects()
        stats.projects_total = len(projects)
        print(f"Found {len(projects)} projects")
    except Exception as e:
        print(f"Error fetching projects: {e}")
        sys.exit(1)

    print("\n--- Migrating projects to Vespa ---")
    for project in projects:
        migrate_project(project, stats, dry_run=args.dry_run)

    # Fetch and migrate tasks
    print("\n--- Fetching tasks from Supabase ---")
    try:
        tasks = fetch_supabase_tasks()
        stats.tasks_total = len(tasks)
        print(f"Found {len(tasks)} tasks")
    except Exception as e:
        print(f"Error fetching tasks: {e}")
        sys.exit(1)

    print("\n--- Migrating tasks to Vespa ---")
    for task in tasks:
        migrate_task(task, stats, dry_run=args.dry_run)

    # Summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(stats.summary())

    if stats.errors:
        print("\nErrors:")
        for error in stats.errors[:10]:
            print(f"  - {error['type']}: {error['id']} ({error.get('title', 'N/A')})")

    print("\nDone!")


if __name__ == "__main__":
    main()
