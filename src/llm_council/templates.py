"""US-04: Custom persona templates with YAML/JSON support.

Provides persona templates, built-in library, and inheritance support.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Union
import yaml

from .models import Persona
from .schemas import PersonaTemplateSchema, validate_persona_template, ValidationErrors


@dataclass
class PersonaTemplate:
    """Template for creating personas."""
    name: str
    role: str
    expertise: list[str]
    personality_traits: list[str]
    perspective: str
    extends: Optional[str] = None  # Parent template name for inheritance
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_persona(self) -> Persona:
        """Convert template to Persona instance."""
        return Persona(
            name=self.name,
            role=self.role,
            expertise=self.expertise,
            personality_traits=self.personality_traits,
            perspective=self.perspective,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise,
            "personality_traits": self.personality_traits,
            "perspective": self.perspective,
            "extends": self.extends,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaTemplate":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            role=data["role"],
            expertise=data.get("expertise", []),
            personality_traits=data.get("personality_traits", []),
            perspective=data.get("perspective", ""),
            extends=data.get("extends"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AuditEntry:
    """Audit log entry for persona usage."""
    timestamp: datetime
    action: str
    template_name: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "template_name": self.template_name,
            "details": self.details,
        }


class PersonaTemplateLibrary:
    """Library of persona templates with inheritance support."""

    def __init__(self):
        self._templates: dict[str, PersonaTemplate] = {}
        self._audit_log: list[AuditEntry] = []
        self._register_builtin_templates()

    def _register_builtin_templates(self):
        """Register built-in persona templates."""
        # Security Expert
        self.register(PersonaTemplate(
            name="Security Expert",
            role="Security Analyst",
            expertise=["cybersecurity", "threat modeling", "vulnerability assessment", "compliance"],
            personality_traits=["vigilant", "detail-oriented", "cautious", "systematic"],
            perspective="Identify and mitigate security risks before they become exploits",
            tags=["security", "risk", "compliance"],
        ))

        # Performance Engineer
        self.register(PersonaTemplate(
            name="Performance Engineer",
            role="Performance Optimization Specialist",
            expertise=["profiling", "optimization", "scalability", "benchmarking"],
            personality_traits=["analytical", "data-driven", "efficiency-focused", "methodical"],
            perspective="Optimize for speed, efficiency, and resource utilization",
            tags=["performance", "optimization", "scalability"],
        ))

        # UX Designer
        self.register(PersonaTemplate(
            name="UX Designer",
            role="User Experience Designer",
            expertise=["user research", "interaction design", "usability", "accessibility"],
            personality_traits=["empathetic", "creative", "user-focused", "iterative"],
            perspective="Design for the best possible user experience",
            tags=["ux", "design", "usability"],
        ))

        # Software Architect
        self.register(PersonaTemplate(
            name="Software Architect",
            role="System Architecture Expert",
            expertise=["system design", "patterns", "scalability", "integration"],
            personality_traits=["strategic", "holistic", "forward-thinking", "principled"],
            perspective="Design robust, maintainable, and scalable systems",
            tags=["architecture", "design", "patterns"],
        ))

        # DevOps Engineer
        self.register(PersonaTemplate(
            name="DevOps Engineer",
            role="DevOps and Infrastructure Specialist",
            expertise=["CI/CD", "infrastructure", "automation", "monitoring"],
            personality_traits=["practical", "automation-minded", "reliable", "collaborative"],
            perspective="Streamline development and deployment processes",
            tags=["devops", "infrastructure", "automation"],
        ))

        # Data Scientist
        self.register(PersonaTemplate(
            name="Data Scientist",
            role="Data Analysis Expert",
            expertise=["statistics", "machine learning", "data visualization", "analytics"],
            personality_traits=["curious", "rigorous", "evidence-based", "insightful"],
            perspective="Extract insights and make data-driven decisions",
            tags=["data", "analytics", "ml"],
        ))

        # QA Engineer
        self.register(PersonaTemplate(
            name="QA Engineer",
            role="Quality Assurance Specialist",
            expertise=["testing", "automation", "quality metrics", "bug tracking"],
            personality_traits=["meticulous", "systematic", "persistent", "quality-focused"],
            perspective="Ensure software quality through comprehensive testing",
            tags=["qa", "testing", "quality"],
        ))

        # Product Manager
        self.register(PersonaTemplate(
            name="Product Manager",
            role="Product Strategy Lead",
            expertise=["product strategy", "roadmapping", "stakeholder management", "market analysis"],
            personality_traits=["strategic", "customer-focused", "decisive", "communicative"],
            perspective="Balance user needs with business goals",
            tags=["product", "strategy", "business"],
        ))

    def register(self, template: PersonaTemplate, audit: bool = True) -> None:
        """Register a persona template."""
        self._templates[template.name] = template
        if audit:
            self._log_action("register", template.name, {"tags": template.tags})

    def get(self, name: str) -> Optional[PersonaTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def list_templates(self, tag: Optional[str] = None) -> list[str]:
        """List available template names, optionally filtered by tag."""
        if tag:
            return [
                name for name, tmpl in self._templates.items()
                if tag in tmpl.tags
            ]
        return list(self._templates.keys())

    def list_tags(self) -> list[str]:
        """List all unique tags."""
        tags = set()
        for tmpl in self._templates.values():
            tags.update(tmpl.tags)
        return sorted(tags)

    def create_persona(self, template_name: str) -> Optional[Persona]:
        """Create a persona from a template."""
        template = self._resolve_template(template_name)
        if template:
            self._log_action("create_persona", template_name)
            return template.to_persona()
        return None

    def _resolve_template(self, name: str) -> Optional[PersonaTemplate]:
        """Resolve a template with inheritance."""
        template = self.get(name)
        if not template:
            return None

        if template.extends:
            parent = self._resolve_template(template.extends)
            if parent:
                # Merge with parent (child overrides parent)
                return self._merge_templates(parent, template)

        return template

    def _merge_templates(
        self,
        parent: PersonaTemplate,
        child: PersonaTemplate,
    ) -> PersonaTemplate:
        """Merge child template with parent (child overrides)."""
        return PersonaTemplate(
            name=child.name,
            role=child.role or parent.role,
            expertise=child.expertise or parent.expertise,
            personality_traits=child.personality_traits or parent.personality_traits,
            perspective=child.perspective or parent.perspective,
            extends=None,  # Resolved
            tags=list(set(parent.tags + child.tags)),
            metadata={**parent.metadata, **child.metadata},
        )

    def _log_action(self, action: str, template_name: str, details: Optional[dict] = None):
        """Log an audit entry."""
        self._audit_log.append(AuditEntry(
            timestamp=datetime.now(),
            action=action,
            template_name=template_name,
            details=details or {},
        ))

    def get_audit_log(self) -> list[AuditEntry]:
        """Get the audit log."""
        return self._audit_log.copy()

    def clear_audit_log(self):
        """Clear the audit log."""
        self._audit_log.clear()


class TemplateLoader:
    """Load persona templates from YAML/JSON files."""

    def __init__(self, library: Optional[PersonaTemplateLibrary] = None):
        self.library = library or PersonaTemplateLibrary()

    def load_yaml(self, path: Union[str, Path]) -> list[PersonaTemplate]:
        """Load templates from YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return self._load_data(data, str(path))

    def load_json(self, path: Union[str, Path]) -> list[PersonaTemplate]:
        """Load templates from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self._load_data(data, str(path))

    def load_string(self, content: str, format: str = "yaml") -> list[PersonaTemplate]:
        """Load templates from string content."""
        if format == "yaml":
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        return self._load_data(data, "string")

    def _load_data(self, data: Any, source: str) -> list[PersonaTemplate]:
        """Load templates from parsed data."""
        templates = []

        # Handle single template or list of templates
        if isinstance(data, dict):
            if "templates" in data:
                template_list = data["templates"]
            else:
                template_list = [data]
        elif isinstance(data, list):
            template_list = data
        else:
            raise ValueError(f"Invalid template format in {source}")

        for tmpl_data in template_list:
            template = self._validate_and_create(tmpl_data)
            templates.append(template)
            self.library.register(template)

        return templates

    def _validate_and_create(self, data: dict) -> PersonaTemplate:
        """Validate and create a template."""
        start_time = time.perf_counter()

        # Validate using schema
        try:
            validate_persona_template(data)
        except ValidationErrors as e:
            raise ValueError(f"Template validation failed: {e}")

        # Create template
        template = PersonaTemplate.from_dict(data)

        # Check validation time < 200ms
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 200:
            # Log warning but don't fail
            pass

        return template


# Default library instance
_template_library = PersonaTemplateLibrary()


def get_template_library() -> PersonaTemplateLibrary:
    """Get the default template library."""
    return _template_library


def create_persona_from_template(template_name: str) -> Optional[Persona]:
    """Create a persona from a template name."""
    return _template_library.create_persona(template_name)


def list_builtin_templates() -> list[str]:
    """List all built-in template names."""
    return _template_library.list_templates()


def get_builtin_template(name: str) -> Optional[PersonaTemplate]:
    """Get a built-in template by name."""
    return _template_library.get(name)
