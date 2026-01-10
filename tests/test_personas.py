"""Tests for persona management."""

import pytest

from llm_council.models import Persona, DEFAULT_PERSONAS
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
