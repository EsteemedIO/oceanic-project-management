"""
Spec-Kit Parser Service

Parses spec-kit generated markdown files (feature.md, plan.md, tasks.md)
and extracts structured data for Archon project creation.
"""

import re
from pathlib import Path
from typing import Dict, List, Any


class SpecKitParser:
    """Parses spec-kit generated markdown files"""

    async def parse_feature_spec(self, feature_file: Path) -> Dict[str, Any]:
        """
        Parse feature.md file

        Supports multiple formats:
        - # Feature NNN: Feature Name
        - # Feature Name (Spec Kit Feature)
        - **Feature ID:** XYZ-NNN

        Args:
            feature_file: Path to feature.md file

        Returns:
            Dict containing:
                - feature_number: str (e.g., "001")
                - name: str (feature name)
                - description: str (feature description)
                - branch: str (git branch name)
        """
        content = feature_file.read_text()

        # Try multiple title patterns
        feature_name = None
        feature_number = None

        # Pattern 1: # Feature 001: Chat Feature
        title_match = re.search(r'^#\s+Feature\s+(\d{3}):\s+(.+)$', content, re.MULTILINE)
        if title_match:
            feature_number = title_match.group(1)
            feature_name = title_match.group(2).strip()

        # Pattern 2: # Feature Name (Spec Kit Feature) or similar
        if not feature_name:
            title_match = re.search(r'^#\s+([^#\n]+?)(?:\s+\(.*\))?\s*$', content, re.MULTILINE)
            if title_match:
                feature_name = title_match.group(1).strip()

        # Extract feature ID from metadata: **Feature ID:** ODP-003 or AR-002
        if not feature_number:
            id_match = re.search(r'\*\*Feature ID:\*\*\s*([A-Z]+-)?(\d+)', content)
            if id_match:
                feature_number = id_match.group(2).zfill(3)

        # Fallback: extract from directory name
        if not feature_number:
            dir_match = re.search(r'feature-(\d+)', str(feature_file.parent))
            if dir_match:
                feature_number = dir_match.group(1).zfill(3)

        if not feature_name:
            raise ValueError(f"Could not parse feature title from {feature_file}")

        if not feature_number:
            feature_number = "000"

        # Extract description - try multiple patterns
        description = ""
        # Pattern 1: ### Problem section
        desc_match = re.search(r'###\s+Problem\s*\n(.+?)(?=\n###|\n##|\Z)', content, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()
        else:
            # Pattern 2: ## Description section
            desc_match = re.search(r'##\s+Description\s*\n\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
            if desc_match:
                description = desc_match.group(1).strip()

        # Extract branch name if present
        branch_match = re.search(r'Branch:\s*`([^`]+)`', content)
        branch = branch_match.group(1) if branch_match else f"feature/{feature_number}-{self._slugify(feature_name)}"

        return {
            'feature_number': feature_number,
            'name': feature_name,
            'description': description[:500] if description else f"Specification for {feature_name}",
            'branch': branch
        }

    async def parse_tasks_md(self, tasks_file: Path) -> List[Dict[str, Any]]:
        """
        Parse tasks.md file

        Supports multiple formats:
        - [ ] Task description [P] [Group: A] [Est: 2h]
        - [ ] **Task title**
        - [ ] Task with description below

        Args:
            tasks_file: Path to tasks.md file

        Returns:
            List of task dicts containing:
                - title: str (task title)
                - description: str (expanded description)
                - parallelizable: bool
                - parallel_group: str | None (A, B, C, etc.)
                - estimated_minutes: int
                - line_number: int (source line)
        """
        content = tasks_file.read_text()
        tasks = []
        current_section = None
        lines = content.split('\n')

        for i, line in enumerate(lines, start=1):
            # Track section headers for context
            if line.startswith('## ') or line.startswith('### '):
                current_section = line.lstrip('#').strip()
                continue

            # Skip completed tasks and non-task lines
            if '- [x]' in line.lower() or '- [ ]' not in line:
                continue

            # Multiple patterns for task extraction
            task_title = None
            description = ""

            # Pattern 1: - [ ] **Bold title**
            bold_match = re.search(r'- \[\s?\] \*\*(.+?)\*\*', line)
            if bold_match:
                task_title = bold_match.group(1).strip()
                # Check if there's text after the bold title
                remaining = line[bold_match.end():].strip()
                if remaining:
                    description = remaining

            # Pattern 2: - [ ] Task description [P] [Group: A] [Est: 2h]
            if not task_title:
                task_match = re.search(
                    r'- \[\s?\] (.+?)(?:\s+\[P\])?(?:\s+\[Group:\s*([A-Z])\])?(?:\s+\[Est:\s*(\d+)([hm])\])?$',
                    line
                )
                if task_match:
                    task_title = task_match.group(1).strip()
                    # Remove markdown formatting from title
                    task_title = re.sub(r'\*\*(.+?)\*\*', r'\1', task_title)

            if not task_title:
                continue

            # Collect ALL indented sub-items as the description
            sub_items = []
            j = i  # i is 1-indexed, lines[i] is the next line after current task
            while j < len(lines):
                next_line = lines[j]
                # Check if line is indented (part of this task's sub-items)
                if next_line.startswith('  ') and not next_line.strip().startswith('- [ ]'):
                    # Include sub-item lines (both `- text` and plain text)
                    sub_items.append(next_line.strip())
                    j += 1
                elif next_line.strip() == '':
                    # Skip empty lines but continue looking
                    j += 1
                else:
                    # Stop at non-indented content or new task
                    break

            # Build description from sub-items
            if sub_items:
                description = '\n'.join(sub_items)

            # Build full description with section context
            full_description = f"[{current_section}] " if current_section else ""
            full_description += task_title
            if description:
                full_description += f"\n\n{description}"

            # Check for parallelization markers
            parallelizable = '[P]' in line
            parallel_group = None
            group_match = re.search(r'\[Group:\s*([A-Z])\]', line)
            if group_match:
                parallel_group = group_match.group(1)

            # Check for estimation
            estimated_minutes = None
            est_match = re.search(r'\[Est:\s*(\d+)([hm])\]', line)
            if est_match:
                minutes = int(est_match.group(1)) * (60 if est_match.group(2) == 'h' else 1)
                estimated_minutes = minutes

            tasks.append({
                'title': task_title[:200],  # Truncate long titles
                'description': full_description[:500],  # Truncate long descriptions
                'parallelizable': parallelizable,
                'parallel_group': parallel_group,
                'estimated_minutes': estimated_minutes or self._estimate_effort(task_title),
                'line_number': i
            })

        return tasks

    async def parse_plan_md(self, plan_file: Path) -> Dict[str, Any]:
        """
        Parse plan.md file for technical details

        Args:
            plan_file: Path to plan.md file

        Returns:
            Dict of section_name -> section_content
        """
        content = plan_file.read_text()

        # Extract key sections
        sections = {}
        current_section = None
        section_content = []

        for line in content.split('\n'):
            if line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(section_content).strip()
                current_section = line[3:].strip().lower().replace(' ', '_')
                section_content = []
            elif current_section:
                section_content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(section_content).strip()

        return sections

    def _slugify(self, text: str) -> str:
        """
        Convert text to URL-friendly slug

        Args:
            text: Text to slugify

        Returns:
            Slugified text (lowercase, hyphens, no special chars)
        """
        return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

    def _expand_task_description(self, title: str) -> str:
        """
        Expand task title into fuller description

        Could be enhanced with LLM for better expansion.

        Args:
            title: Task title

        Returns:
            Expanded description (currently just returns title)
        """
        # TODO: Could use LLM here for better expansion
        return title

    def _estimate_effort(self, description: str) -> int:
        """
        Estimate task effort in minutes based on keywords

        Args:
            description: Task description

        Returns:
            Estimated minutes (240 for complex, 120 for simple)
        """
        complexity_markers = ['integrate', 'refactor', 'migrate', 'architect', 'design']
        is_complex = any(marker in description.lower() for marker in complexity_markers)
        return 240 if is_complex else 120  # 4 hours or 2 hours
