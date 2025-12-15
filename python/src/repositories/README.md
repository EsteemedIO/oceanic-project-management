# Vespa Repository Layer

Python repository classes for Vespa integration in the Oceanic platform, migrating from Supabase to Vespa document store.

## Architecture

```
repositories/
├── __init__.py                        # Export all repositories
├── vespa_client.py                    # Shared Vespa HTTP client
├── vespa_project_repository.py        # Project CRUD with embeddings
├── vespa_task_repository.py           # Task CRUD with embeddings
└── vespa_work_order_repository.py     # Work order CRUD (no embeddings)
```

## Key Features

### VespaClient
- HTTP/2 connection pooling via httpx
- Retry logic with exponential backoff (tenacity)
- CRUD operations: feed_data_point, get_data, update_data, delete_data, query
- YQL query support for complex searches

### VespaProjectRepository
- **Embeddings**: Generated from name + description (1536-dim)
- **Search**: Hybrid semantic + text search
- **Multi-tenant**: Scoped by org_id
- **Schema**: oceanic_project

### VespaTaskRepository
- **Embeddings**: Generated from title + description (1536-dim)
- **Search**: Hybrid semantic + text search
- **Project-scoped**: Optional project_id for project tasks
- **Standalone**: Empty project_id for standalone tasks
- **Schema**: oceanic_task

### VespaWorkOrderRepository
- **No embeddings**: Work orders accessed by ID/status only
- **Workflow tracking**: Agent execution state
- **Schema**: oceanic_work_order

## Environment Variables

```bash
VESPA_HOST=http://localhost:8081  # Vespa endpoint URL
VESPA_TIMEOUT=30                  # Request timeout in seconds
```

## Usage Examples

### Projects

```python
from repositories import VespaProjectRepository

# Initialize with embedding service
repo = VespaProjectRepository(embedding_service=generate_embedding)

# Create project
project = await repo.create({
    "project_id": "proj_123",
    "org_id": "org_456",
    "name": "Q4 Roadmap",
    "description": "Platform improvements for Q4 2025"
})

# Get by ID
project = await repo.get("proj_123")

# List with filters
projects = await repo.list(
    "org_456",
    filters={"status": "active", "limit": 10}
)

# Semantic search
results = await repo.search(
    "platform improvements Q4",
    "org_456",
    limit=5
)

# Update
project = await repo.update("proj_123", {"status": "archived"})

# Delete
deleted = await repo.delete("proj_123")
```

### Tasks

```python
from repositories import VespaTaskRepository

repo = VespaTaskRepository(embedding_service=generate_embedding)

# Create task
task = await repo.create({
    "task_id": "task_123",
    "org_id": "org_456",
    "project_id": "proj_789",  # Optional - empty for standalone
    "title": "Implement user auth",
    "description": "Add OAuth2 authentication flow"
})

# List tasks for project
tasks = await repo.list(
    "org_456",
    filters={"project_id": "proj_789", "status": "doing"}
)

# Search tasks
results = await repo.search("authentication OAuth", "org_456")

# Update status
task = await repo.update("task_123", {"status": "done"})
```

### Work Orders

```python
from repositories import VespaWorkOrderRepository

repo = VespaWorkOrderRepository()

# Create work order
work_order = await repo.create({
    "work_order_id": "wo_123",
    "org_id": "org_456",
    "agent_work_order_id": "awo_789",
    "repository_url": "https://github.com/org/repo",
    "sandbox_identifier": "sandbox-123"
})

# List running work orders
work_orders = await repo.list(
    "org_456",
    filters={"status": "running"}
)

# Update status
work_order = await repo.update("wo_123", {
    "status": "completed",
    "metadata": {"result": "success"}
})
```

## Embedding Service Integration

Repositories expect an async embedding service:

```python
async def generate_embedding(text: str) -> List[float]:
    """Generate 1536-dimensional embedding vector."""
    # Call OpenAI, Cohere, or local embedding model
    response = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Inject into repository
repo = VespaProjectRepository(embedding_service=generate_embedding)
```

**Placeholder Behavior**: If no embedding service provided, repositories return zero vectors `[0.0] * 1536` with a warning log.

## Vespa Schema Requirements

### oceanic_project
```
field project_id type string { indexing: summary | attribute }
field org_id type string { indexing: summary | attribute }
field name type string { indexing: summary | index | attribute }
field description type string { indexing: summary | index }
field status type string { indexing: summary | attribute }
field metadata type map<string, string> { indexing: summary }
field embedding type tensor<float>(x[1536]) { indexing: attribute | index }
field created_at type long { indexing: summary | attribute }
field updated_at type long { indexing: summary | attribute }
```

### oceanic_task
```
field task_id type string { indexing: summary | attribute }
field org_id type string { indexing: summary | attribute }
field project_id type string { indexing: summary | attribute }
field title type string { indexing: summary | index | attribute }
field description type string { indexing: summary | index }
field status type string { indexing: summary | attribute }
field priority type string { indexing: summary | attribute }
field assignee type string { indexing: summary | attribute }
field metadata type map<string, string> { indexing: summary }
field embedding type tensor<float>(x[1536]) { indexing: attribute | index }
field created_at type long { indexing: summary | attribute }
field updated_at type long { indexing: summary | attribute }
field due_date type long { indexing: summary | attribute }
```

### oceanic_work_order
```
field work_order_id type string { indexing: summary | attribute }
field org_id type string { indexing: summary | attribute }
field agent_work_order_id type string { indexing: summary | attribute }
field repository_url type string { indexing: summary }
field sandbox_identifier type string { indexing: summary | attribute }
field git_branch_name type string { indexing: summary }
field agent_session_id type string { indexing: summary }
field status type string { indexing: summary | attribute }
field metadata type map<string, string> { indexing: summary }
field created_at type long { indexing: summary | attribute }
field updated_at type long { indexing: summary | attribute }
field completed_at type long { indexing: summary | attribute }
```

**Ranking Profile** (for hybrid search):
```
rank-profile hybrid_search {
    inputs {
        query(query_embedding) tensor<float>(x[1536])
    }
    first-phase {
        expression: closeness(field, embedding) + bm25(name) + bm25(description)
    }
}
```

## Error Handling

All repository methods:
- Use structured logging with operation context
- Retry transient failures (3 attempts, exponential backoff)
- Raise httpx.HTTPStatusError on permanent failures
- Log detailed error context for debugging

## Testing

Unit tests should mock VespaClient:

```python
@pytest.fixture
def mock_vespa_client(mocker):
    client = mocker.Mock(spec=VespaClient)
    return client

async def test_create_project(mock_vespa_client):
    repo = VespaProjectRepository(client=mock_vespa_client)

    project = await repo.create({
        "project_id": "proj_123",
        "org_id": "org_456",
        "name": "Test Project"
    })

    mock_vespa_client.feed_data_point.assert_called_once()
```

## Migration from Supabase

Replace Supabase repository imports:

```python
# Before
from agent_work_orders.state_manager.supabase_repository import SupabaseWorkOrderRepository

# After
from repositories import VespaWorkOrderRepository
```

Key differences:
- **Async all the way**: All methods are async (Supabase was sync)
- **Embedding generation**: Projects/tasks now auto-generate embeddings
- **Semantic search**: New search() method for semantic queries
- **Timestamps**: Unix milliseconds (long) instead of ISO strings
- **Metadata**: map<string, string> instead of JSONB

## Next Steps

1. **Deploy Vespa schemas** - Use Vespa schema definitions above
2. **Integrate embedding service** - Connect OpenAI/Cohere/local embeddings
3. **Update service layer** - Replace Supabase calls with Vespa repositories
4. **Add unit tests** - Test each repository with mocked VespaClient
5. **Performance tuning** - Optimize query patterns and indexing
