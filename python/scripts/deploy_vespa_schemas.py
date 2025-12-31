"""
Deploy Oceanic schemas to Vespa.

This script deploys the oceanic_project, oceanic_task, oceanic_work_order, and
oceanic_work_order_step schemas to the running Vespa instance.

Usage:
    cd /packages/oceanic-project-management/python
    PYTHONPATH=. python scripts/deploy_vespa_schemas.py

Environment:
    VESPA_HOST: Vespa config server URL (default: http://localhost:19071)
    VESPA_APPLICATION_URL: Vespa application endpoint (default: http://localhost:8081)
"""

import io
import os
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import jinja2
import requests

# Vespa config server (different from application endpoint)
VESPA_CONFIG_URL = os.getenv("VESPA_CONFIG_URL", "http://localhost:19071")
VESPA_APPLICATION_URL = os.getenv("VESPA_APPLICATION_URL", "http://localhost:8081")

# Path to Echo's Vespa app config (packages/echo/backend/...)
ECHO_APP_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "echo" / "backend" / "onyx" / "document_index" / "vespa" / "app_config"

# Oceanic schema names
OCEANIC_SCHEMAS = [
    "oceanic_project",
    "oceanic_task",
    "oceanic_memory",
    "oceanic_work_order",
    "oceanic_work_order_step",
]

# Default Danswer schema (generated at runtime)
DANSWER_SCHEMA_NAME = "danswer_chunk_nomic_ai_nomic_embed_text_v1"


def get_current_schemas() -> list[str]:
    """Get list of currently deployed schemas from Vespa."""
    try:
        response = requests.get(f"{VESPA_APPLICATION_URL}/ApplicationStatus")
        if response.status_code == 200:
            data = response.json()
            # Parse schemas from application status
            return list(data.get("application", {}).get("schemas", {}).keys())
    except Exception as e:
        print(f"Warning: Could not fetch current schemas: {e}")
    return []


def create_document_elements(schema_names: list[str]) -> str:
    """Create document XML elements for services.xml."""
    elements = []
    for name in schema_names:
        elements.append(f'            <document type="{name}" mode="index" />')
    return "\n".join(elements)


def create_zip_package(schema_names: list[str], schemas_dir: Path) -> bytes:
    """Create a zip package with all schemas and services.xml."""
    jinja_env = jinja2.Environment()

    # Read services.xml template
    services_template_path = schemas_dir.parent / "services.xml.jinja"
    with open(services_template_path, "r") as f:
        services_template = jinja_env.from_string(f.read())

    # Render services.xml with all document types
    document_elements = create_document_elements(schema_names)
    services_xml = services_template.render(
        document_elements=document_elements,
        num_search_threads="2",
    )

    # Create validation overrides (allow schema changes)
    overrides_template_path = schemas_dir.parent / "validation-overrides.xml.jinja"
    with open(overrides_template_path, "r") as f:
        overrides_template = jinja_env.from_string(f.read())

    # Set override date to 7 days from now
    until_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    validation_overrides = overrides_template.render(until_date=until_date)

    # Create zip buffer
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add services.xml
        zipf.writestr("services.xml", services_xml.encode("utf-8"))

        # Add validation overrides
        zipf.writestr("validation-overrides.xml", validation_overrides.encode("utf-8"))

        # Add all schema files
        for schema_name in schema_names:
            # Try .sd file first (static schema)
            schema_path = schemas_dir / f"{schema_name}.sd"
            if schema_path.exists():
                with open(schema_path, "r") as f:
                    zipf.writestr(f"schemas/{schema_name}.sd", f.read().encode("utf-8"))
                print(f"  Added schema: {schema_name}.sd")
            else:
                # Try .sd.jinja template
                jinja_path = schemas_dir / f"{schema_name}.sd.jinja"
                if jinja_path.exists():
                    with open(jinja_path, "r") as f:
                        template = jinja_env.from_string(f.read())
                    # Render with default values for Danswer schema
                    schema_content = template.render(
                        schema_name=schema_name,
                        multi_tenant=False,
                        dim=768,  # Nomic embedding dimension
                        embedding_precision="float32",
                    )
                    zipf.writestr(f"schemas/{schema_name}.sd", schema_content.encode("utf-8"))
                    print(f"  Added schema from template: {schema_name}.sd.jinja")
                else:
                    print(f"  Warning: Schema not found: {schema_name}")

    zip_buffer.seek(0)
    return zip_buffer.read()


def deploy_application(zip_data: bytes) -> bool:
    """Deploy the application package to Vespa."""
    deploy_url = f"{VESPA_CONFIG_URL}/application/v2/tenant/default/prepareandactivate"

    print(f"\nDeploying to {deploy_url}...")

    headers = {"Content-Type": "application/zip"}
    response = requests.post(deploy_url, headers=headers, data=zip_data)

    if response.status_code == 200:
        print("Deployment successful!")
        result = response.json()
        print(f"  Session ID: {result.get('session-id', 'N/A')}")
        return True
    else:
        print(f"Deployment failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def main():
    print("=" * 60)
    print("Oceanic Vespa Schema Deployment")
    print("=" * 60)

    # Check if schema directory exists
    schemas_dir = ECHO_APP_CONFIG_PATH / "schemas"
    if not schemas_dir.exists():
        print(f"Error: Schemas directory not found: {schemas_dir}")
        sys.exit(1)

    print(f"\nSchemas directory: {schemas_dir}")

    # Get current schemas
    print("\nFetching current Vespa schemas...")
    current_schemas = get_current_schemas()
    print(f"Current schemas: {current_schemas}")

    # Determine full list of schemas to deploy
    # Only include current schemas that are not Danswer (jinja templates)
    # Danswer schema is generated at runtime by Echo, not deployed statically
    all_schemas = list(set(current_schemas + OCEANIC_SCHEMAS))
    # Remove Danswer schema if present - it requires jinja rendering which Echo handles
    all_schemas = [s for s in all_schemas if not s.startswith("danswer_chunk")]

    print(f"\nSchemas to deploy: {all_schemas}")

    # Create deployment package
    print("\nCreating deployment package...")
    zip_data = create_zip_package(all_schemas, schemas_dir)
    print(f"Package size: {len(zip_data)} bytes")

    # Deploy
    success = deploy_application(zip_data)

    if success:
        print("\n" + "=" * 60)
        print("Deployment complete!")
        print("\nNext steps:")
        print("1. Verify schemas at http://localhost:8081/ApplicationStatus")
        print("2. Run migration: python -c 'from src.server.services.vespa_migration_service import run_migration; import asyncio; asyncio.run(run_migration())'")
        print("=" * 60)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
