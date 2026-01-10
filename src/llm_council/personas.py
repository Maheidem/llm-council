"""Persona management and generation."""

from typing import Optional

from .models import Persona, DEFAULT_PERSONAS
from .providers import LLMProvider


class PersonaManager:
    """Manages personas for council sessions."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        """Initialize persona manager.

        Args:
            provider: LLM provider for generating custom personas
        """
        self.provider = provider
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
    ) -> list[Persona]:
        """Generate appropriate personas based on the topic.

        Uses LLM to analyze the topic and create relevant personas.

        Args:
            topic: The discussion topic
            count: Number of personas to generate

        Returns:
            List of generated personas
        """
        if not self.provider:
            # Fall back to defaults if no provider
            return self.get_default_personas(count)

        system_prompt = """You are an expert at designing discussion panels.
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

        user_prompt = f"""Create {count} diverse personas for discussing this topic:

Topic: {topic}

Remember: Output ONLY the Python dictionary list, no other text."""

        try:
            response = self.provider.complete(system_prompt, user_prompt)
            # Parse the response
            personas = self._parse_persona_response(response, count)
            return personas
        except Exception as e:
            print(f"Failed to generate personas: {e}")
            return self.get_default_personas(count)

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
