"""
Spec-Kit File Watcher Service

Watches the specs/ directory for spec-kit generated files and automatically
creates projects and tasks in Archon when new specifications are created.

Backend Support:
    - Supabase (default): Uses direct Supabase client for CRUD operations
    - Vespa: Uses Vespa repository layer with hybrid search capabilities

Set STORAGE_BACKEND=vespa to use Vespa backend.
"""

import asyncio
from pathlib import Path
from typing import Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .spec_kit_parser import SpecKitParser
from .projects import get_project_service, get_task_service, is_vespa_enabled
from ..config.logfire_config import get_logger

logger = get_logger(__name__)


class SpecKitFileHandler(FileSystemEventHandler):
    """Watches specs/ directory for spec-kit generated files"""

    def __init__(self, specs_dir: str = "specs", loop: Optional[asyncio.AbstractEventLoop] = None):
        self.specs_dir = Path(specs_dir)
        # Use service factory for backend-agnostic service creation
        self.project_service = get_project_service()
        self.task_service = get_task_service()
        self.parser = SpecKitParser()
        self.processing = set()  # Track files being processed to avoid duplicates
        self.loop = loop  # Store reference to main event loop for thread-safe calls

        # Log which backend is being used
        backend = "Vespa" if is_vespa_enabled() else "Supabase"
        logger.info(f"📦 SpecKitFileHandler initialized with {backend} backend")

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for thread-safe async calls"""
        self.loop = loop

    def _run_async(self, coro):
        """Run async coroutine from watchdog thread safely"""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        else:
            # Fallback for when called during initial scan (same thread)
            try:
                asyncio.create_task(coro)
            except RuntimeError:
                logger.warning("No event loop available, skipping async operation")

    def on_created(self, event):
        """Handle new file/directory creation"""
        if event.is_directory:
            # Check if it's a feature directory (e.g., feature-001-chat)
            path = Path(event.src_path)
            if self._is_feature_directory(path):
                self._run_async(self._handle_new_feature(path))

        elif event.src_path.endswith('tasks.md'):
            # Tasks file created - scaffold work products
            path = Path(event.src_path)
            if path not in self.processing:
                self.processing.add(path)
                feature_dir = path.parent
                self._run_async(self._scaffold_tasks(feature_dir, path))

    def on_modified(self, event):
        """Handle file modifications"""
        if not event.is_directory:
            path = Path(event.src_path)
            if path.name in ['feature.md', 'plan.md', 'tasks.md']:
                feature_dir = path.parent
                self._run_async(self._update_project(feature_dir))

    async def _handle_new_feature(self, feature_dir: Path):
        """Create Archon project when new feature directory detected"""
        feature_name = feature_dir.name
        logger.info(f"📁 New feature directory detected: {feature_name}")

        # Wait for feature.md to be created (give spec-kit time)
        feature_file = feature_dir / "feature.md"
        for _ in range(10):  # Wait up to 5 seconds
            if feature_file.exists():
                break
            await asyncio.sleep(0.5)

        if not feature_file.exists():
            logger.warning(f"⚠️ feature.md not found in {feature_dir}")
            return

        try:
            # Parse feature specification
            spec_data = await self.parser.parse_feature_spec(feature_file)

            # Build metadata for the project
            metadata = {
                'source': 'spec-kit',
                'feature_number': spec_data['feature_number'],
                'feature_dir': str(feature_dir),
                'branch': spec_data.get('branch'),
                'description': spec_data['description'],
                'spec_files': {
                    'feature': str(feature_file)
                }
            }

            # Create project with description including metadata
            # For Vespa backend, description is embedded and searchable
            description = spec_data.get('description', '')
            if metadata.get('feature_number'):
                description = f"[{metadata['feature_number']}] {description}"

            success, result = self.project_service.create_project(
                title=spec_data['name'],
                github_repo=spec_data.get('branch'),  # Store branch in github_repo field
                description=description
            )

            if not success:
                logger.error(f"Failed to create project: {result.get('error')}")
                return

            project = result['project']
            project_id = project['id']

            # For Supabase backend, store additional metadata in data field
            if not is_vespa_enabled() and hasattr(self.project_service, 'supabase_client'):
                self.project_service.supabase_client.table("archon_projects").update({
                    "data": [metadata]
                }).eq("id", project_id).execute()

            logger.info(f"✅ Project created: {project_id} ({project['title']})")

        except Exception as e:
            logger.error(f"❌ Failed to create project: {e}", exc_info=True)

    async def _scaffold_tasks(self, feature_dir: Path, tasks_file: Path):
        """Parse tasks.md and create tasks in Archon"""
        try:
            # Find the corresponding Archon project
            feature_name = feature_dir.name
            project = await self._find_project_by_feature(feature_name)

            if not project:
                logger.warning(f"⚠️ No project found for feature: {feature_name}")
                return

            # Parse tasks from tasks.md
            tasks_data = await self.parser.parse_tasks_md(tasks_file)

            if not tasks_data:
                logger.warning(f"⚠️ No tasks found in {tasks_file}")
                self.processing.discard(tasks_file)
                return

            # Create tasks using Archon's task service
            created_tasks = []
            for i, task_data in enumerate(tasks_data):
                success, result = await self.task_service.create_task(
                    project_id=project['id'],
                    title=task_data['title'],
                    description=task_data['description'],
                    assignee='User',  # Default to User, can be reassigned later
                    task_order=i
                )

                if success:
                    task = result['task']
                    created_tasks.append(task)
                else:
                    logger.error(f"Failed to create task: {result.get('error')}")

            logger.info(f"✅ Created {len(created_tasks)} tasks for project {project['title']}")

        except Exception as e:
            logger.error(f"❌ Failed to scaffold tasks: {e}", exc_info=True)
        finally:
            self.processing.discard(tasks_file)

    async def _update_project(self, feature_dir: Path):
        """Update project when specification files are modified"""
        feature_name = feature_dir.name
        logger.info(f"🔄 Updating project for feature: {feature_name}")

        try:
            project = await self._find_project_by_feature(feature_name)
            if not project:
                return

            project_id = project['id']

            # Re-parse specification files
            feature_file = feature_dir / "feature.md"
            if feature_file.exists():
                spec_data = await self.parser.parse_feature_spec(feature_file)

                # Update project using service method (works for both backends)
                success, result = self.project_service.update_project(
                    project_id=project_id,
                    title=spec_data['name']
                )

                if success:
                    logger.info(f"✅ Updated project {project_id}")
                else:
                    logger.error(f"Failed to update project {project_id}: {result.get('error')}")

            # Sync tasks from tasks.md
            tasks_file = feature_dir / "tasks.md"
            if tasks_file.exists():
                await self._sync_tasks(project_id, tasks_file, project['title'])

        except Exception as e:
            logger.error(f"❌ Failed to update project: {e}", exc_info=True)

    async def _sync_tasks(self, project_id: str, tasks_file: Path, project_title: str):
        """Sync tasks from tasks.md to Archon - adds new tasks, preserves existing"""
        try:
            # Parse tasks from file
            file_tasks = await self.parser.parse_tasks_md(tasks_file)
            if not file_tasks:
                return

            # Get existing tasks using the task service (works for both backends)
            success, result = self.task_service.list_tasks(project_id=project_id)
            if not success:
                logger.error(f"Failed to list existing tasks: {result.get('error')}")
                return

            existing_tasks = result.get('tasks', [])
            existing_titles = {task['title'] for task in existing_tasks}

            # Find new tasks (in file but not in database)
            new_tasks = [t for t in file_tasks if t['title'] not in existing_titles]

            if not new_tasks:
                logger.info(f"📋 No new tasks to sync for {project_title}")
                return

            # Create new tasks
            created_count = 0
            for i, task_data in enumerate(new_tasks):
                success, result = await self.task_service.create_task(
                    project_id=project_id,
                    title=task_data['title'],
                    description=task_data['description'],
                    assignee='User',
                    task_order=len(existing_tasks) + i
                )
                if success:
                    created_count += 1
                else:
                    logger.warning(f"Failed to create task '{task_data['title'][:30]}...': {result.get('error')}")

            logger.info(f"📋 Synced {created_count} new tasks for {project_title} (total: {len(existing_tasks) + created_count})")

        except Exception as e:
            logger.error(f"❌ Failed to sync tasks: {e}", exc_info=True)

    async def _find_project_by_feature(self, feature_name: str) -> Optional[dict]:
        """Find Archon project by feature directory name"""
        try:
            # Query Archon's database for project with matching metadata
            success, result = self.project_service.list_projects(include_content=True)

            if not success:
                logger.error(f"Failed to list projects: {result.get('error')}")
                return None

            projects = result.get('projects', [])
            for project in projects:
                # Check if project data contains feature_dir matching feature_name
                data_list = project.get('data', [])
                for data in data_list:
                    if isinstance(data, dict):
                        feature_dir = data.get('feature_dir', '')
                        if feature_dir.endswith(feature_name):
                            return project

            return None
        except Exception as e:
            logger.error(f"Error finding project by feature: {e}", exc_info=True)
            return None

    def _is_feature_directory(self, path: Path) -> bool:
        """Check if path matches spec-kit feature directory pattern"""
        # Pattern: feature-NNN-name
        name = path.name
        parts = name.split('-')
        if len(parts) >= 3 and parts[0] == 'feature':
            # Check if second part is a number
            try:
                int(parts[1])
                return True
            except ValueError:
                pass
        return False


class SpecKitWatcherService:
    """Main service to start/stop the file watcher"""

    def __init__(self, specs_dir: str = "specs"):
        self.specs_dir = Path(specs_dir)
        self.observer: Optional[Observer] = None
        self.event_handler = SpecKitFileHandler(specs_dir)

    def start(self):
        """Start watching the specs directory"""
        if not self.specs_dir.exists():
            logger.info(f"⚠️ Creating specs directory: {self.specs_dir}")
            self.specs_dir.mkdir(parents=True, exist_ok=True)

        # Get the current event loop and pass to handler for thread-safe async calls
        try:
            loop = asyncio.get_running_loop()
            self.event_handler.set_loop(loop)
        except RuntimeError:
            logger.warning("No running event loop during watcher start")

        self.observer = Observer()
        self.observer.schedule(
            self.event_handler,
            str(self.specs_dir),
            recursive=True
        )
        self.observer.start()
        logger.info(f"👁️ Spec-Kit watcher started on {self.specs_dir}")

        # Scan existing specs on startup
        asyncio.create_task(self._initial_scan())

    async def _initial_scan(self):
        """Scan existing feature directories and create projects/tasks if missing"""
        logger.info(f"🔍 Scanning existing specs in {self.specs_dir}")

        try:
            for item in self.specs_dir.iterdir():
                if item.is_dir() and self.event_handler._is_feature_directory(item):
                    feature_name = item.name
                    logger.info(f"📁 Found existing feature: {feature_name}")

                    # Check if project already exists
                    existing_project = await self.event_handler._find_project_by_feature(feature_name)

                    if existing_project:
                        logger.info(f"✓ Project already exists for {feature_name}")
                        # Check if tasks need to be created using the task service
                        tasks_file = item / "tasks.md"
                        if tasks_file.exists():
                            # Query task count using service method (works for both backends)
                            project_id = existing_project.get('id')
                            try:
                                success, result = self.event_handler.task_service.list_tasks(project_id=project_id)
                                task_count = len(result.get('tasks', [])) if success else 0
                            except Exception as e:
                                logger.error(f"Error checking task count: {e}")
                                task_count = 0

                            if task_count == 0:
                                logger.info(f"📋 Creating tasks for existing project {feature_name} (no tasks found)")
                                await self.event_handler._scaffold_tasks(item, tasks_file)
                            else:
                                logger.info(f"✓ Project {feature_name} already has {task_count} tasks, skipping")
                    else:
                        # Create new project from existing spec
                        logger.info(f"➕ Creating project for existing spec: {feature_name}")
                        await self.event_handler._handle_new_feature(item)

                        # Also create tasks if tasks.md exists
                        tasks_file = item / "tasks.md"
                        if tasks_file.exists():
                            logger.info(f"📋 Creating tasks for {feature_name}")
                            await self.event_handler._scaffold_tasks(item, tasks_file)

            logger.info("✅ Initial spec scan complete")

        except Exception as e:
            logger.error(f"❌ Error during initial scan: {e}", exc_info=True)

    def stop(self):
        """Stop watching"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("🛑 Spec-Kit watcher stopped")
