"""Persona management and generation."""

import json
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

from .models import Persona, PersonaProviderConfig, DEFAULT_PERSONAS
from .providers import LLMProvider


# Default persona generation prompt
DEFAULT_GENERATION_PROMPT = """You are an expert at designing discussion panels.
Given a topic, create diverse personas that would provide valuable, different perspectives.
Each persona should have a unique viewpoint that contributes to a well-rounded discussion.

Output ONLY valid Python code that creates a list of Persona objects. No explanations.
Use this exact format:

personas = [
    {
        "name": "Name Here",
        "role": "Role Title",
        "expertise": ["skill1", "skill2", "skill3"],
        "personality_traits": ["trait1", "trait2", "trait3"],
        "perspective": "One sentence describing their viewpoint"
    },
    # more personas...
]"""


class PersonaManager:
    """Manages personas for council sessions."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        generation_provider: Optional[LLMProvider] = None,
        generation_prompt: Optional[str] = None,
    ):
        """Initialize persona manager.

        Args:
            provider: LLM provider for discussion (used as fallback for generation)
            generation_provider: Separate provider for persona generation
            generation_prompt: Custom prompt template for persona generation
        """
        self.provider = provider
        self.generation_provider = generation_provider or provider
        self.generation_prompt = generation_prompt or DEFAULT_GENERATION_PROMPT
        self._custom_personas: list[Persona] = []

    def get_default_personas(self, count: int = 3) -> list[Persona]:
        """Get a subset of default personas.

        Args:
            count: Number of personas to return (max 5)

        Returns:
            List of default personas
        """
        return DEFAULT_PERSONAS[:min(count, len(DEFAULT_PERSONAS))]

    def add_custom_persona(self, persona: Persona) -> None:
        """Add a custom persona."""
        self._custom_personas.append(persona)

    def create_persona(
        self,
        name: str,
        role: str,
        expertise: list[str],
        personality_traits: list[str],
        perspective: str,
    ) -> Persona:
        """Create and register a custom persona."""
        persona = Persona(
            name=name,
            role=role,
            expertise=expertise,
            personality_traits=personality_traits,
            perspective=perspective,
        )
        self.add_custom_persona(persona)
        return persona

    def generate_personas_for_topic(
        self,
        topic: str,
        count: int = 3,
        save_to: Optional[str] = None,
        provider_configs: Optional[Dict[str, PersonaProviderConfig]] = None,
    ) -> list[Persona]:
        """Generate appropriate personas based on the topic.

        Uses LLM to analyze the topic and create relevant personas.

        Args:
            topic: The discussion topic
            count: Number of personas to generate
            save_to: Optional file path to save generated personas (YAML/JSON)
            provider_configs: Optional per-persona provider configs to apply

        Returns:
            List of generated personas
        """
        gen_provider = self.generation_provider or self.provider
        if not gen_provider:
            # Fall back to defaults if no provider
            personas = self.get_default_personas(count)
        else:
            user_prompt = f"""Create {count} diverse personas for discussing this topic:

Topic: {topic}

Remember: Output ONLY the Python dictionary list, no other text."""

            try:
                response = gen_provider.complete(self.generation_prompt, user_prompt)
                # Parse the response
                personas = self._parse_persona_response(response, count)

                # Apply provider configs if specified
                if provider_configs:
                    personas = self._apply_provider_configs(personas, provider_configs)

            except Exception as e:
                print(f"Failed to generate personas: {e}")
                personas = self.get_default_personas(count)

        # Save if requested - always save, even for fallback personas
        if save_to:
            self.save_personas(personas, save_to)

        return personas

    def _apply_provider_configs(
        self,
        personas: list[Persona],
        configs: Dict[str, PersonaProviderConfig],
    ) -> list[Persona]:
        """Apply provider configs to personas by name."""
        result = []
        for persona in personas:
            if persona.name in configs:
                result.append(persona.with_provider_config(configs[persona.name]))
            else:
                result.append(persona)
        return result

    def save_personas(self, personas: list[Persona], path: str) -> None:
        """Save personas to file (YAML or JSON).

        Args:
            personas: List of personas to save
            path: File path (extension determines format: .yaml/.yml or .json)
        """
        data = [self._persona_to_dict(p) for p in personas]
        path_obj = Path(path)

        if path_obj.suffix.lower() in ('.yaml', '.yml'):
            with open(path_obj, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path_obj, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

    def load_personas(self, path: str) -> list[Persona]:
        """Load personas from file (YAML or JSON).

        Args:
            path: File path to load from

        Returns:
            List of loaded personas
        """
        path_obj = Path(path)

        if path_obj.suffix.lower() in ('.yaml', '.yml'):
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)

        personas = []
        for item in data:
            provider_config = None
            if 'provider_config' in item and item['provider_config']:
                provider_config = PersonaProviderConfig.from_dict(item['provider_config'])

            personas.append(Persona(
                name=item.get('name', 'Unknown'),
                role=item.get('role', 'Participant'),
                expertise=item.get('expertise', []),
                personality_traits=item.get('personality_traits', []),
                perspective=item.get('perspective', 'General perspective'),
                provider_config=provider_config,
            ))

        return personas

    def _persona_to_dict(self, persona: Persona) -> Dict[str, Any]:
        """Convert persona to dictionary for serialization."""
        data = {
            'name': persona.name,
            'role': persona.role,
            'expertise': persona.expertise,
            'personality_traits': persona.personality_traits,
            'perspective': persona.perspective,
        }
        if persona.provider_config:
            data['provider_config'] = persona.provider_config.to_dict()
        return data

    def _parse_persona_response(self, response: str, expected_count: int) -> list[Persona]:
        """Parse LLM response into Persona objects."""
        import re
        import ast

        # Try to extract the list from the response using balanced bracket matching
        # First, try to find 'personas = [' and then find the matching ']'
        start_patterns = [
            r'personas\s*=\s*\[',
            r'\[',
        ]

        for start_pattern in start_patterns:
            match = re.search(start_pattern, response)
            if match:
                start_idx = match.end() - 1  # Position of the '['
                # Find matching closing bracket
                bracket_count = 0
                end_idx = start_idx
                for i, char in enumerate(response[start_idx:]):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = start_idx + i + 1
                            break

                if end_idx > start_idx:
                    try:
                        list_str = response[start_idx:end_idx]
                        parsed = ast.literal_eval(list_str)

                        if isinstance(parsed, list):
                            personas = []
                            for p in parsed:
                                if isinstance(p, dict):
                                    personas.append(Persona(
                                        name=p.get("name", "Unknown"),
                                        role=p.get("role", "Participant"),
                                        expertise=p.get("expertise", []),
                                        personality_traits=p.get("personality_traits", []),
                                        perspective=p.get("perspective", "General perspective"),
                                    ))
                            if personas:
                                return personas[:expected_count]
                    except (SyntaxError, ValueError):
                        continue

        # Fall back to defaults
        return self.get_default_personas(expected_count)

    def get_all_personas(self) -> list[Persona]:
        """Get all registered personas (default + custom)."""
        return DEFAULT_PERSONAS + self._custom_personas
