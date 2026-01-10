"""Tests for US-04: Custom persona templates."""

import pytest
import json
import tempfile
from pathlib import Path
from llm_council.templates import (
    PersonaTemplate,
    PersonaTemplateLibrary,
    TemplateLoader,
    AuditEntry,
    get_template_library,
    create_persona_from_template,
    list_builtin_templates,
    get_builtin_template,
)
from llm_council.models import Persona


class TestPersonaTemplate:
    """Tests for persona templates."""

    def test_template_creation(self):
        template = PersonaTemplate(
            name="Test Persona",
            role="Tester",
            expertise=["testing"],
            personality_traits=["analytical"],
            perspective="Test everything",
        )
        assert template.name == "Test Persona"
        assert template.role == "Tester"

    def test_template_to_persona(self):
        template = PersonaTemplate(
            name="Test Persona",
            role="Tester",
            expertise=["testing"],
            personality_traits=["analytical"],
            perspective="Test perspective",
        )
        persona = template.to_persona()
        assert isinstance(persona, Persona)
        assert persona.name == "Test Persona"
        assert persona.role == "Tester"

    def test_template_to_dict(self):
        template = PersonaTemplate(
            name="Test",
            role="Role",
            expertise=["exp"],
            personality_traits=["trait"],
            perspective="perspective",
            tags=["tag1"],
        )
        d = template.to_dict()
        assert d["name"] == "Test"
        assert d["tags"] == ["tag1"]

    def test_template_from_dict(self):
        data = {
            "name": "Test",
            "role": "Role",
            "expertise": ["exp"],
            "personality_traits": ["trait"],
            "perspective": "perspective",
        }
        template = PersonaTemplate.from_dict(data)
        assert template.name == "Test"

    def test_template_with_inheritance(self):
        template = PersonaTemplate(
            name="Child",
            role="Child Role",
            expertise=["child_exp"],
            personality_traits=["child_trait"],
            perspective="child perspective",
            extends="Parent",
        )
        assert template.extends == "Parent"


class TestPersonaTemplateLibrary:
    """Tests for persona template library."""

    def test_builtin_templates_registered(self):
        library = PersonaTemplateLibrary()
        templates = library.list_templates()
        assert "Security Expert" in templates
        assert "Performance Engineer" in templates
        assert "UX Designer" in templates
        assert "Software Architect" in templates

    def test_register_custom_template(self):
        library = PersonaTemplateLibrary()
        template = PersonaTemplate(
            name="Custom Template",
            role="Custom Role",
            expertise=["custom"],
            personality_traits=["unique"],
            perspective="custom view",
        )
        library.register(template)
        assert library.get("Custom Template") is not None

    def test_get_template(self):
        library = PersonaTemplateLibrary()
        template = library.get("Security Expert")
        assert template is not None
        assert template.name == "Security Expert"

    def test_get_nonexistent_template(self):
        library = PersonaTemplateLibrary()
        template = library.get("Nonexistent")
        assert template is None

    def test_list_templates_by_tag(self):
        library = PersonaTemplateLibrary()
        security_templates = library.list_templates(tag="security")
        assert "Security Expert" in security_templates

    def test_list_all_tags(self):
        library = PersonaTemplateLibrary()
        tags = library.list_tags()
        assert "security" in tags
        assert "performance" in tags
        assert "ux" in tags

    def test_create_persona_from_library(self):
        library = PersonaTemplateLibrary()
        persona = library.create_persona("Security Expert")
        assert persona is not None
        assert persona.name == "Security Expert"

    def test_create_persona_nonexistent(self):
        library = PersonaTemplateLibrary()
        persona = library.create_persona("Nonexistent")
        assert persona is None

    def test_inheritance_resolution(self):
        library = PersonaTemplateLibrary()

        # Register parent
        parent = PersonaTemplate(
            name="Parent",
            role="Parent Role",
            expertise=["parent_exp"],
            personality_traits=["parent_trait"],
            perspective="parent perspective",
            tags=["parent"],
        )
        library.register(parent)

        # Register child that extends parent
        child = PersonaTemplate(
            name="Child",
            role="Child Role",
            expertise=["child_exp"],
            personality_traits=["child_trait"],
            perspective="child perspective",
            extends="Parent",
            tags=["child"],
        )
        library.register(child)

        # Create persona from child - should resolve inheritance
        persona = library.create_persona("Child")
        assert persona is not None
        assert persona.name == "Child"
        assert persona.role == "Child Role"


class TestAuditLog:
    """Tests for audit logging."""

    def test_audit_log_on_register(self):
        library = PersonaTemplateLibrary()
        library.clear_audit_log()

        template = PersonaTemplate(
            name="Audit Test",
            role="Role",
            expertise=["exp"],
            personality_traits=["trait"],
            perspective="perspective",
        )
        library.register(template)

        log = library.get_audit_log()
        assert len(log) >= 1
        assert any(e.action == "register" and e.template_name == "Audit Test" for e in log)

    def test_audit_log_on_create_persona(self):
        library = PersonaTemplateLibrary()
        library.clear_audit_log()

        library.create_persona("Security Expert")

        log = library.get_audit_log()
        assert any(e.action == "create_persona" for e in log)

    def test_audit_entry_to_dict(self):
        from datetime import datetime
        entry = AuditEntry(
            timestamp=datetime.now(),
            action="test",
            template_name="Test",
            details={"key": "value"},
        )
        d = entry.to_dict()
        assert "timestamp" in d
        assert d["action"] == "test"

    def test_clear_audit_log(self):
        library = PersonaTemplateLibrary()
        library.create_persona("Security Expert")
        assert len(library.get_audit_log()) > 0
        library.clear_audit_log()
        assert len(library.get_audit_log()) == 0


class TestTemplateLoader:
    """Tests for template loading from files."""

    def test_load_yaml_string(self):
        library = PersonaTemplateLibrary()
        loader = TemplateLoader(library)

        yaml_content = """
name: YAML Persona
role: YAML Role
expertise:
  - yaml
personality_traits:
  - structured
perspective: YAML perspective
"""
        templates = loader.load_string(yaml_content, format="yaml")
        assert len(templates) == 1
        assert templates[0].name == "YAML Persona"

    def test_load_json_string(self):
        library = PersonaTemplateLibrary()
        loader = TemplateLoader(library)

        json_content = json.dumps({
            "name": "JSON Persona",
            "role": "JSON Role",
            "expertise": ["json"],
            "personality_traits": ["structured"],
            "perspective": "JSON perspective",
        })
        templates = loader.load_string(json_content, format="json")
        assert len(templates) == 1
        assert templates[0].name == "JSON Persona"

    def test_load_yaml_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("""
name: File Persona
role: File Role
expertise:
  - files
personality_traits:
  - organized
perspective: File perspective
""")
            f.flush()

            library = PersonaTemplateLibrary()
            loader = TemplateLoader(library)
            templates = loader.load_yaml(f.name)

            assert len(templates) == 1
            assert templates[0].name == "File Persona"

        Path(f.name).unlink()

    def test_load_json_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({
                "name": "JSON File Persona",
                "role": "Role",
                "expertise": ["json"],
                "personality_traits": ["precise"],
                "perspective": "JSON file perspective",
            }, f)
            f.flush()

            library = PersonaTemplateLibrary()
            loader = TemplateLoader(library)
            templates = loader.load_json(f.name)

            assert len(templates) == 1
            assert templates[0].name == "JSON File Persona"

        Path(f.name).unlink()

    def test_load_multiple_templates(self):
        library = PersonaTemplateLibrary()
        loader = TemplateLoader(library)

        yaml_content = """
templates:
  - name: Template 1
    role: Role 1
    expertise: [exp1]
    personality_traits: [trait1]
    perspective: perspective 1
  - name: Template 2
    role: Role 2
    expertise: [exp2]
    personality_traits: [trait2]
    perspective: perspective 2
"""
        templates = loader.load_string(yaml_content, format="yaml")
        assert len(templates) == 2

    def test_validation_failure(self):
        library = PersonaTemplateLibrary()
        loader = TemplateLoader(library)

        # Missing required field
        json_content = json.dumps({
            "name": "Invalid",
            "role": "Role",
            # Missing expertise, personality_traits, perspective
        })
        with pytest.raises(ValueError) as exc_info:
            loader.load_string(json_content, format="json")
        assert "validation failed" in str(exc_info.value).lower()


class TestGlobalFunctions:
    """Tests for global helper functions."""

    def test_get_template_library(self):
        library = get_template_library()
        assert library is not None
        assert isinstance(library, PersonaTemplateLibrary)

    def test_create_persona_from_template(self):
        persona = create_persona_from_template("Security Expert")
        assert persona is not None
        assert persona.name == "Security Expert"

    def test_list_builtin_templates(self):
        templates = list_builtin_templates()
        assert len(templates) >= 8
        assert "Security Expert" in templates

    def test_get_builtin_template(self):
        template = get_builtin_template("Performance Engineer")
        assert template is not None
        assert template.name == "Performance Engineer"


class TestValidationPerformance:
    """Tests for validation performance."""

    def test_validation_under_200ms(self):
        import time

        library = PersonaTemplateLibrary()
        loader = TemplateLoader(library)

        json_content = json.dumps({
            "name": "Perf Test",
            "role": "Role",
            "expertise": ["exp1", "exp2", "exp3"],
            "personality_traits": ["trait1", "trait2"],
            "perspective": "Test perspective for performance validation",
        })

        start = time.perf_counter()
        loader.load_string(json_content, format="json")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200, f"Validation took {elapsed_ms:.2f}ms, expected < 200ms"
