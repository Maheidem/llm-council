"""Tests for persona management."""

import pytest

from llm_council.models import Persona, DEFAULT_PERSONAS, PersonaProviderConfig
from llm_council.personas import PersonaManager


class TestPersonaManager:
    """Tests for PersonaManager."""

    def test_get_default_personas(self):
        manager = PersonaManager()
        personas = manager.get_default_personas(3)
        assert len(personas) == 3
        for p in personas:
            assert isinstance(p, Persona)

    def test_get_default_personas_limit(self):
        manager = PersonaManager()
        # Request more than available
        personas = manager.get_default_personas(100)
        assert len(personas) == len(DEFAULT_PERSONAS)

    def test_add_custom_persona(self):
        manager = PersonaManager()
        custom = Persona(
            name="Custom Expert",
            role="Custom Role",
            expertise=["custom"],
            personality_traits=["unique"],
            perspective="Custom perspective",
        )
        manager.add_custom_persona(custom)
        all_personas = manager.get_all_personas()
        assert custom in all_personas

    def test_create_persona(self):
        manager = PersonaManager()
        persona = manager.create_persona(
            name="Created Expert",
            role="Created Role",
            expertise=["skill1", "skill2"],
            personality_traits=["trait1"],
            perspective="Created perspective",
        )
        assert persona.name == "Created Expert"
        assert persona in manager.get_all_personas()

    def test_parse_persona_response_valid(self):
        manager = PersonaManager()
        response = '''
personas = [
    {
        "name": "AI Expert",
        "role": "Machine Learning Specialist",
        "expertise": ["deep learning", "NLP"],
        "personality_traits": ["analytical", "curious"],
        "perspective": "Focus on AI capabilities"
    }
]
'''
        personas = manager._parse_persona_response(response, 1)
        assert len(personas) == 1
        assert personas[0].name == "AI Expert"

    def test_parse_persona_response_invalid_fallback(self):
        manager = PersonaManager()
        response = "This is not valid JSON or Python"
        personas = manager._parse_persona_response(response, 3)
        # Should fall back to defaults
        assert len(personas) == 3
        assert personas[0] in DEFAULT_PERSONAS

    def test_generate_personas_without_provider(self):
        manager = PersonaManager(provider=None)
        personas = manager.generate_personas_for_topic("AI Ethics", 3)
        # Without provider, should return defaults
        assert len(personas) == 3
        for p in personas:
            assert p in DEFAULT_PERSONAS


class TestPersonaFileOperations:
    """Tests for persona file save/load operations."""

    def test_save_and_load_yaml(self, tmp_path):
        """Test saving and loading personas from YAML."""
        manager = PersonaManager()
        personas = manager.get_default_personas(3)

        yaml_path = tmp_path / "personas.yaml"
        manager.save_personas(personas, str(yaml_path))

        assert yaml_path.exists()

        loaded = manager.load_personas(str(yaml_path))
        assert len(loaded) == 3
        assert loaded[0].name == personas[0].name
        assert loaded[0].role == personas[0].role
        assert loaded[0].perspective == personas[0].perspective

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading personas from JSON."""
        manager = PersonaManager()
        personas = manager.get_default_personas(2)

        json_path = tmp_path / "personas.json"
        manager.save_personas(personas, str(json_path))

        assert json_path.exists()

        loaded = manager.load_personas(str(json_path))
        assert len(loaded) == 2
        assert loaded[0].name == personas[0].name

    def test_save_and_load_with_provider_config(self, tmp_path):
        """Test saving and loading personas with provider configs."""
        manager = PersonaManager()

        persona = Persona(
            name="Custom Expert",
            role="Test Role",
            expertise=["testing"],
            personality_traits=["analytical"],
            perspective="Test perspective",
            provider_config=PersonaProviderConfig(
                model="test-model",
                temperature=0.9,
            ),
        )

        yaml_path = tmp_path / "persona_with_config.yaml"
        manager.save_personas([persona], str(yaml_path))

        loaded = manager.load_personas(str(yaml_path))
        assert len(loaded) == 1
        assert loaded[0].name == "Custom Expert"
        assert loaded[0].provider_config is not None
        assert loaded[0].provider_config.model == "test-model"
        assert loaded[0].provider_config.temperature == 0.9

    def test_load_nonexistent_file_raises(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        manager = PersonaManager()
        with pytest.raises(FileNotFoundError):
            manager.load_personas("/nonexistent/path/personas.yaml")

    def test_generate_with_save_to_fallback(self, tmp_path):
        """Test that save_to works even when falling back to defaults."""
        manager = PersonaManager(provider=None)  # No provider = will fallback

        yaml_path = tmp_path / "fallback_personas.yaml"
        personas = manager.generate_personas_for_topic(
            "Test Topic",
            count=2,
            save_to=str(yaml_path)
        )

        # Should return defaults
        assert len(personas) == 2
        # File should still be saved
        assert yaml_path.exists()

        # Load and verify
        loaded = manager.load_personas(str(yaml_path))
        assert len(loaded) == 2
