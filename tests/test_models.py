"""Tests for data models."""

import pytest

from llm_council.models import (
    PersonaProviderConfig,
    Persona,
    Message,
    Vote,
    VoteChoice,
    RoundResult,
    CouncilSession,
    ConsensusType,
    DEFAULT_PERSONAS,
)


class TestPersona:
    """Tests for Persona model."""

    def test_persona_creation(self):
        persona = Persona(
            name="Test Expert",
            role="Tester",
            expertise=["testing", "validation"],
            personality_traits=["thorough", "precise"],
            perspective="Focus on quality and correctness",
        )
        assert persona.name == "Test Expert"
        assert persona.role == "Tester"
        assert len(persona.expertise) == 2
        assert len(persona.personality_traits) == 2

    def test_persona_system_prompt(self):
        persona = Persona(
            name="Test Expert",
            role="Tester",
            expertise=["testing"],
            personality_traits=["thorough"],
            perspective="Focus on quality",
        )
        prompt = persona.to_system_prompt()
        assert "Test Expert" in prompt
        assert "Tester" in prompt
        assert "testing" in prompt
        assert "thorough" in prompt

    def test_default_personas_exist(self):
        assert len(DEFAULT_PERSONAS) >= 3
        for persona in DEFAULT_PERSONAS:
            assert persona.name
            assert persona.role
            assert len(persona.expertise) > 0

    def test_with_provider_config(self):
        """Tests Persona.with_provider_config() method."""
        original_persona = Persona(
            name="Config Test Expert",
            role="Config Tester",
            expertise=["config", "testing"],
            personality_traits=["methodical"],
            perspective="Focus on configuration",
            is_mediator=True,
        )

        config = PersonaProviderConfig(
            model="test-model",
            temperature=0.5,
            max_tokens=1024,
        )

        new_persona = original_persona.with_provider_config(config)

        # Verify it returns a new instance
        assert new_persona is not original_persona

        # Verify all persona fields are preserved
        assert new_persona.name == original_persona.name
        assert new_persona.role == original_persona.role
        assert new_persona.expertise == original_persona.expertise
        assert new_persona.personality_traits == original_persona.personality_traits
        assert new_persona.perspective == original_persona.perspective
        assert new_persona.is_mediator == original_persona.is_mediator

        # Verify the provider config is set
        assert new_persona.provider_config is config
        assert new_persona.provider_config.model == "test-model"
        assert new_persona.provider_config.temperature == 0.5
        assert new_persona.provider_config.max_tokens == 1024

        # Verify original is unchanged
        assert original_persona.provider_config is None


class TestMessage:
    """Tests for Message model."""

    def test_message_creation(self):
        msg = Message(
            persona_name="The Expert",
            content="This is my perspective.",
            round_number=1,
            message_type="discussion",
        )
        assert msg.persona_name == "The Expert"
        assert msg.round_number == 1


class TestVote:
    """Tests for Vote model."""

    def test_vote_creation(self):
        vote = Vote(
            persona_name="The Expert",
            choice=VoteChoice.AGREE,
            reasoning="This makes sense.",
        )
        assert vote.choice == VoteChoice.AGREE

    def test_vote_choices(self):
        assert VoteChoice.AGREE.value == "agree"
        assert VoteChoice.DISAGREE.value == "disagree"
        assert VoteChoice.ABSTAIN.value == "abstain"


class TestRoundResult:
    """Tests for RoundResult model."""

    def test_round_result_creation(self):
        messages = [
            Message("Expert1", "Content1", 1),
            Message("Expert2", "Content2", 1),
        ]
        result = RoundResult(
            round_number=1,
            messages=messages,
            consensus_reached=False,
        )
        assert result.round_number == 1
        assert len(result.messages) == 2
        assert not result.consensus_reached


class TestCouncilSession:
    """Tests for CouncilSession model."""

    def test_session_creation(self):
        personas = [DEFAULT_PERSONAS[0]]
        session = CouncilSession(
            topic="Test Topic",
            objective="Reach decision",
            personas=personas,
        )
        assert session.topic == "Test Topic"
        assert len(session.personas) == 1
        assert len(session.rounds) == 0
        assert not session.consensus_reached

    def test_session_to_dict(self):
        personas = [DEFAULT_PERSONAS[0]]
        session = CouncilSession(
            topic="Test Topic",
            objective="Reach decision",
            personas=personas,
        )
        result = session.to_dict()
        assert result["topic"] == "Test Topic"
        assert result["objective"] == "Reach decision"
        assert len(result["personas"]) == 1
        assert result["consensus_reached"] is False


class TestConsensusType:
    """Tests for ConsensusType enum."""

    def test_consensus_types(self):
        assert ConsensusType.UNANIMOUS.value == "unanimous"
        assert ConsensusType.SUPERMAJORITY.value == "supermajority"
        assert ConsensusType.MAJORITY.value == "majority"
        assert ConsensusType.PLURALITY.value == "plurality"


class TestPersonaProviderConfig:
    """Tests for PersonaProviderConfig serialization."""

    def test_default_values_all_none(self):
        """Verify all fields default to None."""
        config = PersonaProviderConfig()
        assert config.model is None
        assert config.provider is None
        assert config.api_base is None
        assert config.api_key is None
        assert config.temperature is None
        assert config.top_p is None
        assert config.top_k is None
        assert config.max_tokens is None
        assert config.frequency_penalty is None
        assert config.presence_penalty is None
        assert config.repeat_penalty is None
        assert config.stop is None
        assert config.seed is None
        assert config.timeout is None

    def test_set_all_sampling_params(self):
        """Set temperature, top_p, top_k, max_tokens."""
        config = PersonaProviderConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=1024,
        )
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.max_tokens == 1024

    def test_set_all_repetition_params(self):
        """Set frequency_penalty, presence_penalty, repeat_penalty."""
        config = PersonaProviderConfig(
            frequency_penalty=0.5,
            presence_penalty=0.3,
            repeat_penalty=1.1,
        )
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.3
        assert config.repeat_penalty == 1.1

    def test_set_control_params(self):
        """Set stop, seed, timeout."""
        config = PersonaProviderConfig(
            stop=["END", "STOP"],
            seed=42,
            timeout=30,
        )
        assert config.stop == ["END", "STOP"]
        assert config.seed == 42
        assert config.timeout == 30

    def test_set_all_params(self):
        """Set every field."""
        config = PersonaProviderConfig(
            model="gpt-4",
            provider="openai",
            api_base="https://api.openai.com/v1",
            api_key="sk-test-key",
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            max_tokens=2048,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            repeat_penalty=1.05,
            stop=["###"],
            seed=123,
            timeout=60,
        )
        assert config.model == "gpt-4"
        assert config.provider == "openai"
        assert config.api_base == "https://api.openai.com/v1"
        assert config.api_key == "sk-test-key"
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.max_tokens == 2048
        assert config.frequency_penalty == 0.2
        assert config.presence_penalty == 0.1
        assert config.repeat_penalty == 1.05
        assert config.stop == ["###"]
        assert config.seed == 123
        assert config.timeout == 60

    def test_to_dict_excludes_none(self):
        """to_dict() only includes non-None values."""
        config = PersonaProviderConfig()
        result = config.to_dict()
        assert result == {}

    def test_to_dict_partial_values(self):
        """Set some values, verify correct output."""
        config = PersonaProviderConfig(
            model="llama-3",
            temperature=0.5,
            max_tokens=512,
        )
        result = config.to_dict()
        assert result == {
            "model": "llama-3",
            "temperature": 0.5,
            "max_tokens": 512,
        }
        # Verify None values are not present
        assert "provider" not in result
        assert "api_base" not in result
        assert "top_p" not in result

    def test_to_dict_with_stop_sequences(self):
        """Verify list serializes correctly."""
        config = PersonaProviderConfig(
            stop=["STOP", "END", "###"],
        )
        result = config.to_dict()
        assert result == {"stop": ["STOP", "END", "###"]}
        assert isinstance(result["stop"], list)
        assert len(result["stop"]) == 3

    def test_from_dict_empty(self):
        """from empty dict."""
        config = PersonaProviderConfig.from_dict({})
        assert config.model is None
        assert config.provider is None
        assert config.api_base is None
        assert config.api_key is None
        assert config.temperature is None
        assert config.top_p is None
        assert config.top_k is None
        assert config.max_tokens is None
        assert config.frequency_penalty is None
        assert config.presence_penalty is None
        assert config.repeat_penalty is None
        assert config.stop is None
        assert config.seed is None
        assert config.timeout is None

    def test_from_dict_all_params(self):
        """from dict with all fields."""
        data = {
            "model": "claude-3",
            "provider": "anthropic",
            "api_base": "https://api.anthropic.com",
            "api_key": "sk-ant-test",
            "temperature": 0.6,
            "top_p": 0.85,
            "top_k": 30,
            "max_tokens": 4096,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.25,
            "repeat_penalty": 1.2,
            "stop": ["<END>"],
            "seed": 999,
            "timeout": 120,
        }
        config = PersonaProviderConfig.from_dict(data)
        assert config.model == "claude-3"
        assert config.provider == "anthropic"
        assert config.api_base == "https://api.anthropic.com"
        assert config.api_key == "sk-ant-test"
        assert config.temperature == 0.6
        assert config.top_p == 0.85
        assert config.top_k == 30
        assert config.max_tokens == 4096
        assert config.frequency_penalty == 0.4
        assert config.presence_penalty == 0.25
        assert config.repeat_penalty == 1.2
        assert config.stop == ["<END>"]
        assert config.seed == 999
        assert config.timeout == 120

    def test_from_dict_partial(self):
        """from dict with some fields."""
        data = {
            "model": "mistral-7b",
            "temperature": 0.3,
            "seed": 42,
        }
        config = PersonaProviderConfig.from_dict(data)
        assert config.model == "mistral-7b"
        assert config.temperature == 0.3
        assert config.seed == 42
        # Verify unset fields are None
        assert config.provider is None
        assert config.api_base is None
        assert config.max_tokens is None
        assert config.stop is None

    def test_round_trip_serialization(self):
        """to_dict then from_dict preserves values."""
        original = PersonaProviderConfig(
            model="gpt-4-turbo",
            provider="openai",
            api_base="https://api.openai.com/v1",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=1024,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            repeat_penalty=1.1,
            stop=["DONE", "COMPLETE"],
            seed=777,
            timeout=90,
        )

        # Serialize to dict
        serialized = original.to_dict()

        # Deserialize back
        restored = PersonaProviderConfig.from_dict(serialized)

        # Verify all values match
        assert restored.model == original.model
        assert restored.provider == original.provider
        assert restored.api_base == original.api_base
        assert restored.temperature == original.temperature
        assert restored.top_p == original.top_p
        assert restored.top_k == original.top_k
        assert restored.max_tokens == original.max_tokens
        assert restored.frequency_penalty == original.frequency_penalty
        assert restored.presence_penalty == original.presence_penalty
        assert restored.repeat_penalty == original.repeat_penalty
        assert restored.stop == original.stop
        assert restored.seed == original.seed
        assert restored.timeout == original.timeout

    def test_from_dict_ignores_unknown_keys(self):
        """unknown keys do not cause errors."""
        data = {
            "model": "test-model",
            "temperature": 0.5,
            "unknown_key": "should be ignored",
            "another_unknown": 123,
            "nested_unknown": {"foo": "bar"},
        }
        # Should not raise any exception
        config = PersonaProviderConfig.from_dict(data)

        # Known fields should be set
        assert config.model == "test-model"
        assert config.temperature == 0.5

        # Unknown fields should not be present as attributes
        assert not hasattr(config, "unknown_key")
        assert not hasattr(config, "another_unknown")
        assert not hasattr(config, "nested_unknown")
