"""Tests for council engine.

POLICY: NO MOCKED API TESTS - Session tests use real LM Studio.
VoteParser and VotingMachine tests are pure logic (no API).
See CLAUDE.md for rationale.
"""

import pytest

from llm_council.models import (
    Persona,
    Message,
    Vote,
    VoteChoice,
    ConsensusType,
    DEFAULT_PERSONAS,
)
from llm_council.council import CouncilEngine
from llm_council.discussion import DiscussionState
from llm_council.voting import VoteParser, VotingMachine, StructuredVote


class TestCouncilEngine:
    """Tests for CouncilEngine with real LM Studio API."""

    def test_engine_creation(self, stub_provider):
        """Test engine instantiation."""
        engine = CouncilEngine(
            provider=stub_provider,
            consensus_type=ConsensusType.MAJORITY,
            max_rounds=5,
        )
        assert engine.consensus_type == ConsensusType.MAJORITY
        assert engine.max_rounds == 5

    @pytest.mark.api
    def test_conduct_round(self, lmstudio_provider, simple_personas):
        """Test conducting a discussion round with real API."""
        engine = CouncilEngine(provider=lmstudio_provider, max_rounds=1)
        personas = simple_personas
        discussion_state = DiscussionState()
        discussion_state.advance_round()

        result = engine._conduct_round(
            round_num=1,
            topic="API Testing",
            objective="Validate real API integration",
            personas=personas,
            history=[],
            initial_context=None,
            discussion_state=discussion_state,
        )

        assert result.round_number == 1
        assert len(result.messages) == 3  # One per persona
        for msg in result.messages:
            assert msg.round_number == 1
            assert len(msg.content) > 0  # Real response

    def test_format_history(self, stub_provider):
        """Test history formatting - pure logic, no API."""
        engine = CouncilEngine(provider=stub_provider)
        messages = [
            Message("Expert1", "First message", 1),
            Message("Expert2", "Second message", 1),
            Message("Expert1", "Third message", 2),
        ]

        history_text = engine._format_history(messages)

        assert "Round 1:" in history_text
        assert "Round 2:" in history_text
        assert "Expert1" in history_text
        assert "Expert2" in history_text

    @pytest.mark.api
    def test_conduct_vote(self, lmstudio_provider, simple_personas):
        """Test voting with real LLM responses."""
        engine = CouncilEngine(
            provider=lmstudio_provider,
            consensus_type=ConsensusType.MAJORITY,
        )

        # Mark first persona as mediator (as run_session does)
        personas = simple_personas.copy()
        personas[0] = Persona(
            name=personas[0].name,
            role=personas[0].role,
            expertise=personas[0].expertise,
            personality_traits=personas[0].personality_traits,
            perspective=personas[0].perspective,
            is_mediator=True,
        )

        result = engine._conduct_vote(
            topic="Test Topic",
            objective="Reach a decision",
            personas=personas,
            history=[],
        )

        # Should have votes from non-mediator personas
        assert "votes" in result
        assert len(result["votes"]) == 2  # 3 personas - 1 mediator = 2 votes
        assert "consensus_reached" in result

    @pytest.mark.api
    def test_run_session_completes(self, council_engine_factory, simple_personas):
        """Test full session runs to completion with real API."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Quick Decision",
            objective="Make a choice between A and B",
            personas=simple_personas,
        )

        # Session should complete
        assert len(session.rounds) >= 1
        assert len(session.personas) == 3
        # Should have attempted voting
        has_votes = any(r.votes for r in session.rounds)
        assert has_votes or session.consensus_reached

    @pytest.mark.api
    def test_run_session_tracks_messages(self, council_engine_factory, simple_personas):
        """Verify session tracks all messages from real API."""
        engine = council_engine_factory(max_rounds=2)

        session = engine.run_session(
            topic="Message Tracking Test",
            objective="Verify all responses are captured",
            personas=simple_personas,
        )

        # Each round should have messages from all personas
        for round_data in session.rounds:
            if round_data.messages:
                for msg in round_data.messages:
                    assert len(msg.content) > 0, "All messages should have content"
                    assert msg.persona_name in [p.name for p in simple_personas]


class TestVoteParser:
    """Tests for deterministic vote parsing - pure logic, no API."""

    def test_parse_structured_vote_agree(self):
        response = "[VOTE] AGREE\n[CONFIDENCE] 0.85\n[REASONING] This is a good proposal."
        vote = VoteParser.parse("TestPersona", response)

        assert vote.choice == VoteChoice.AGREE
        assert vote.confidence == 0.85
        assert "good proposal" in vote.reasoning
        assert vote.parse_success is True

    def test_parse_structured_vote_disagree(self):
        response = "[VOTE] DISAGREE\n[CONFIDENCE] 0.6\n[REASONING] I have concerns."
        vote = VoteParser.parse("TestPersona", response)

        assert vote.choice == VoteChoice.DISAGREE
        assert vote.confidence == 0.6
        assert vote.parse_success is True

    def test_parse_simple_format(self):
        response = "VOTE: AGREE\nCONFIDENCE: 0.9\nREASON: Sounds good."
        vote = VoteParser.parse("TestPersona", response)

        assert vote.choice == VoteChoice.AGREE
        assert vote.confidence == 0.9

    def test_parse_fallback_keyword(self):
        response = "I think we should AGREE with this proposal because it makes sense."
        vote = VoteParser.parse("TestPersona", response)

        assert vote.choice == VoteChoice.AGREE
        assert vote.confidence == 0.5  # Default

    def test_parse_abstain_default(self):
        response = "I'm not sure what to think about this."
        vote = VoteParser.parse("TestPersona", response)

        assert vote.choice == VoteChoice.ABSTAIN
        assert vote.parse_success is False  # Had to default

    def test_to_legacy_vote(self):
        structured = StructuredVote(
            persona_name="Test",
            choice=VoteChoice.AGREE,
            confidence=0.8,
            reasoning="Good idea",
        )
        legacy = VoteParser.to_legacy_vote(structured)

        assert legacy.persona_name == "Test"
        assert legacy.choice == VoteChoice.AGREE
        assert legacy.reasoning == "Good idea"


class TestVotingMachine:
    """Tests for deterministic vote tallying - pure logic, no API."""

    def test_tally_unanimous_agree(self):
        votes = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, "Yes"),
            StructuredVote("P2", VoteChoice.AGREE, 0.8, "Agreed"),
            StructuredVote("P3", VoteChoice.AGREE, 0.7, "Sounds good"),
        ]
        machine = VotingMachine(ConsensusType.UNANIMOUS)
        tally = machine.tally(votes)

        assert tally.agree_count == 3
        assert tally.disagree_count == 0
        assert tally.agree_ratio == 1.0
        assert tally.consensus_reached is True

    def test_tally_unanimous_fails_with_disagree(self):
        votes = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, "Yes"),
            StructuredVote("P2", VoteChoice.AGREE, 0.8, "Agreed"),
            StructuredVote("P3", VoteChoice.DISAGREE, 0.7, "No"),
        ]
        machine = VotingMachine(ConsensusType.UNANIMOUS)
        tally = machine.tally(votes)

        assert tally.agree_count == 2
        assert tally.disagree_count == 1
        assert tally.consensus_reached is False

    def test_tally_majority_passes(self):
        votes = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, "Yes"),
            StructuredVote("P2", VoteChoice.AGREE, 0.8, "Yes"),
            StructuredVote("P3", VoteChoice.DISAGREE, 0.7, "No"),
        ]
        machine = VotingMachine(ConsensusType.MAJORITY)
        tally = machine.tally(votes)

        assert tally.agree_count == 2
        assert tally.agree_ratio == 2/3
        assert tally.consensus_reached is True  # 66% > 50%

    def test_tally_majority_fails_on_tie(self):
        votes = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, "Yes"),
            StructuredVote("P2", VoteChoice.DISAGREE, 0.8, "No"),
        ]
        machine = VotingMachine(ConsensusType.MAJORITY)
        tally = machine.tally(votes)

        assert tally.agree_ratio == 0.5
        assert tally.consensus_reached is False  # 50% not > 50%

    def test_tally_supermajority(self):
        votes = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, "Yes"),
            StructuredVote("P2", VoteChoice.AGREE, 0.8, "Yes"),
            StructuredVote("P3", VoteChoice.AGREE, 0.7, "Yes"),
            StructuredVote("P4", VoteChoice.DISAGREE, 0.6, "No"),
        ]
        machine = VotingMachine(ConsensusType.SUPERMAJORITY)
        tally = machine.tally(votes)

        assert tally.agree_ratio == 0.75
        assert tally.consensus_reached is True  # 75% > 66.67%

    def test_tally_plurality(self):
        votes = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, "Yes"),
            StructuredVote("P2", VoteChoice.DISAGREE, 0.8, "No"),
            StructuredVote("P3", VoteChoice.ABSTAIN, 0.5, "Unsure"),
        ]
        machine = VotingMachine(ConsensusType.PLURALITY)
        tally = machine.tally(votes)

        # 1 agree vs 1 disagree = tie, no winner
        assert tally.winning_choice is None
        assert tally.consensus_reached is False

    def test_tally_abstain_excluded(self):
        votes = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, "Yes"),
            StructuredVote("P2", VoteChoice.ABSTAIN, 0.5, "Unsure"),
            StructuredVote("P3", VoteChoice.ABSTAIN, 0.5, "Unsure"),
        ]
        machine = VotingMachine(ConsensusType.MAJORITY)
        tally = machine.tally(votes)

        assert tally.total_voting == 1  # Only 1 non-abstain
        assert tally.agree_ratio == 1.0  # 1/1 = 100%
        assert tally.consensus_reached is True

    def test_tally_to_dict(self):
        votes = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, "Yes"),
        ]
        machine = VotingMachine(ConsensusType.MAJORITY)
        tally = machine.tally(votes)
        result = machine.to_dict(tally)

        assert "agree_count" in result
        assert "disagree_count" in result
        assert "consensus_reached" in result
        assert result["consensus_type"] == "majority"


class TestPerPersonaProviderConfig:
    """Tests for per-persona provider configuration in CouncilEngine.

    Tests verify that _get_provider_for_persona correctly resolves:
    1. Persona's provider_config (if set)
    2. Provider registry lookup by persona name
    3. Default provider fallback
    """

    # Helper to create a test provider without requiring LM Studio connection
    @staticmethod
    def _create_test_provider(
        model="openai/test-model",
        api_base="http://localhost:1234/v1",
        temperature=0.7,
        max_tokens=1024,
        **kwargs
    ):
        """Create a provider for testing without connection test."""
        from llm_council.providers import LiteLLMProvider, ProviderConfig

        config = ProviderConfig(
            model=model,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=kwargs.get("timeout", 120),
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            repeat_penalty=kwargs.get("repeat_penalty"),
            stop=kwargs.get("stop"),
            seed=kwargs.get("seed"),
        )
        return LiteLLMProvider(config)

    # Non-API tests - pure logic, no LLM calls

    def test_persona_with_provider_config_creates_isolated_provider(self):
        """Verify persona.provider_config creates a new isolated provider."""
        from llm_council.models import PersonaProviderConfig, Persona
        from llm_council.council import CouncilEngine

        # Create a test provider
        default_provider = self._create_test_provider()

        # Create persona with explicit provider_config
        creative_config = PersonaProviderConfig(
            temperature=1.0,
            max_tokens=500,
        )
        creative_persona = Persona(
            name="Creative",
            role="Creator",
            expertise=["creativity"],
            personality_traits=["imaginative"],
            perspective="Think outside the box",
            provider_config=creative_config,
        )

        engine = CouncilEngine(provider=default_provider)

        # Get provider for persona - should create a new one, not return default
        provider = engine._get_provider_for_persona(creative_persona)

        # Verify provider was created with persona's config values
        assert provider.config.temperature == 1.0
        assert provider.config.max_tokens == 500
        # Should NOT be the same object as the default provider
        assert provider is not default_provider

    def test_persona_provider_config_uses_all_params(self):
        """Verify all 10 inference params are used when creating provider."""
        from llm_council.models import PersonaProviderConfig, Persona
        from llm_council.council import CouncilEngine

        # Create a test provider
        default_provider = self._create_test_provider()

        # Set all 10 inference parameters
        full_config = PersonaProviderConfig(
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            max_tokens=2048,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            repeat_penalty=1.1,
            stop=["END"],
            seed=42,
            timeout=60,
        )
        persona = Persona(
            name="FullConfig",
            role="Tester",
            expertise=["testing"],
            personality_traits=["thorough"],
            perspective="Test everything",
            provider_config=full_config,
        )

        engine = CouncilEngine(provider=default_provider)
        provider = engine._get_provider_for_persona(persona)

        # Verify all parameters are set correctly
        assert provider.config.temperature == 0.9
        assert provider.config.top_p == 0.95
        assert provider.config.top_k == 50
        assert provider.config.max_tokens == 2048
        assert provider.config.frequency_penalty == 0.5
        assert provider.config.presence_penalty == 0.3
        assert provider.config.repeat_penalty == 1.1
        assert provider.config.stop == ["END"]
        assert provider.config.seed == 42
        assert provider.config.timeout == 60

    def test_multiple_personas_different_configs(self):
        """Each persona gets its own config, isolated from others."""
        from llm_council.models import PersonaProviderConfig, Persona
        from llm_council.council import CouncilEngine

        # Create a test provider
        default_provider = self._create_test_provider()

        # Creative persona - high temperature
        creative_config = PersonaProviderConfig(temperature=1.5, max_tokens=1000)
        creative = Persona(
            name="Creative",
            role="Creator",
            expertise=["creativity"],
            personality_traits=["imaginative"],
            perspective="Think outside the box",
            provider_config=creative_config,
        )

        # Precise persona - low temperature
        precise_config = PersonaProviderConfig(temperature=0.1, max_tokens=500)
        precise = Persona(
            name="Precise",
            role="Analyst",
            expertise=["analysis"],
            personality_traits=["precise"],
            perspective="Be exact",
            provider_config=precise_config,
        )

        engine = CouncilEngine(provider=default_provider)

        creative_provider = engine._get_provider_for_persona(creative)
        precise_provider = engine._get_provider_for_persona(precise)

        # Each should have its own distinct config
        assert creative_provider.config.temperature == 1.5
        assert creative_provider.config.max_tokens == 1000

        assert precise_provider.config.temperature == 0.1
        assert precise_provider.config.max_tokens == 500

        # They should be different provider instances
        assert creative_provider is not precise_provider

    def test_provider_config_inherits_from_default(self):
        """Missing params fall back to default provider's config."""
        from llm_council.models import PersonaProviderConfig, Persona
        from llm_council.council import CouncilEngine

        # Create a test provider with specific values
        default_provider = self._create_test_provider(
            model="openai/default-model",
            api_base="http://localhost:1234/v1",
            max_tokens=1024,
        )

        # Only set temperature, leave everything else None
        partial_config = PersonaProviderConfig(temperature=0.5)
        persona = Persona(
            name="Partial",
            role="Tester",
            expertise=["testing"],
            personality_traits=["thorough"],
            perspective="Test fallbacks",
            provider_config=partial_config,
        )

        engine = CouncilEngine(provider=default_provider)
        provider = engine._get_provider_for_persona(persona)

        # Temperature should be from persona config
        assert provider.config.temperature == 0.5

        # Other params should fall back to default provider's values
        assert provider.config.model == default_provider.config.model
        assert provider.config.api_base == default_provider.config.api_base
        assert provider.config.max_tokens == default_provider.config.max_tokens

    def test_provider_config_overrides_default(self):
        """Persona params override defaults when explicitly set."""
        from llm_council.models import PersonaProviderConfig, Persona
        from llm_council.council import CouncilEngine

        # Default provider has max_tokens=1024
        default_provider = self._create_test_provider(max_tokens=1024)

        # Override with different value
        override_config = PersonaProviderConfig(max_tokens=2048, temperature=0.3)
        persona = Persona(
            name="Override",
            role="Tester",
            expertise=["testing"],
            personality_traits=["thorough"],
            perspective="Override defaults",
            provider_config=override_config,
        )

        engine = CouncilEngine(provider=default_provider)
        provider = engine._get_provider_for_persona(persona)

        # Persona overrides should take precedence
        assert provider.config.max_tokens == 2048
        assert provider.config.temperature == 0.3

        # Verify default was different
        assert default_provider.config.max_tokens == 1024

    def test_registry_lookup_by_persona_name(self):
        """Registry resolves provider by persona name from persona_configs."""
        from llm_council.providers import ProviderRegistry
        from llm_council.config import (
            ResolvedConfig,
            ProviderSettings,
            GenerationSettings,
            CouncilSettings,
            PersistenceSettings,
        )
        from llm_council.models import Persona
        from llm_council.council import CouncilEngine

        # Create resolved config with persona-specific settings
        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model="openai/default-model",
                api_base="http://localhost:1234/v1",
                temperature=0.7,
            ),
            generation=GenerationSettings(),
            providers={},
            persona_configs={
                "Creative": ProviderSettings(temperature=1.0, max_tokens=2000),
            },
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )
        registry = ProviderRegistry(resolved)

        # Create persona without provider_config - should use registry
        creative = Persona(
            name="Creative",
            role="Creator",
            expertise=["creativity"],
            personality_traits=["imaginative"],
            perspective="Think outside the box",
            provider_config=None,  # No explicit config
        )

        engine = CouncilEngine(provider_registry=registry)
        provider = engine._get_provider_for_persona(creative)

        # Should get config from registry persona_configs
        assert provider.config.temperature == 1.0
        assert provider.config.max_tokens == 2000

    def test_fallback_to_default_provider(self):
        """No config/registry entry = default provider returned."""
        from llm_council.models import Persona
        from llm_council.council import CouncilEngine

        # Create a test provider
        default_provider = self._create_test_provider()

        # Persona with NO provider_config
        plain_persona = Persona(
            name="Plain",
            role="Tester",
            expertise=["testing"],
            personality_traits=["plain"],
            perspective="Use defaults",
            provider_config=None,
        )

        # Engine with only default provider (no registry persona_configs)
        engine = CouncilEngine(provider=default_provider)
        provider = engine._get_provider_for_persona(plain_persona)

        # Should return the default provider itself
        assert provider is default_provider

    # API tests - require real LLM calls

    @pytest.mark.api
    def test_discussion_with_per_persona_configs(
        self, lmstudio_provider, simple_personas
    ):
        """Run 1 round with 2 personas having different provider configs."""
        from llm_council.models import PersonaProviderConfig, Persona, ConsensusType
        from llm_council.council import CouncilEngine
        from llm_council.discussion import DiscussionState

        # Create two personas with different configs
        creative_config = PersonaProviderConfig(
            temperature=1.0,
            top_p=0.95,
            max_tokens=500,
        )
        creative_persona = Persona(
            name="Creative",
            role="Creative Thinker",
            expertise=["creativity", "brainstorming"],
            personality_traits=["imaginative", "bold"],
            perspective="Think outside the box and propose novel solutions",
            provider_config=creative_config,
        )

        precise_config = PersonaProviderConfig(
            temperature=0.2,
            top_p=0.8,
            max_tokens=300,
        )
        precise_persona = Persona(
            name="Precise",
            role="Analytical Expert",
            expertise=["analysis", "logic"],
            personality_traits=["precise", "methodical"],
            perspective="Focus on accuracy and logical reasoning",
            provider_config=precise_config,
        )

        # Third persona without custom config (uses default)
        default_persona = Persona(
            name="Default",
            role="Generalist",
            expertise=["general knowledge"],
            personality_traits=["balanced"],
            perspective="Provide balanced perspective",
            provider_config=None,
        )

        personas = [creative_persona, precise_persona, default_persona]

        engine = CouncilEngine(
            provider=lmstudio_provider,
            consensus_type=ConsensusType.MAJORITY,
            max_rounds=1,
        )

        discussion_state = DiscussionState()
        discussion_state.advance_round()

        result = engine._conduct_round(
            round_num=1,
            topic="Testing Per-Persona Configs",
            objective="Verify each persona uses its own configuration",
            personas=personas,
            history=[],
            initial_context=None,
            discussion_state=discussion_state,
        )

        # Verify we got responses from all personas
        assert result.round_number == 1
        assert len(result.messages) == 3
        for msg in result.messages:
            assert len(msg.content) > 0  # Real response from API

        # Verify persona names match
        persona_names = {m.persona_name for m in result.messages}
        assert "Creative" in persona_names
        assert "Precise" in persona_names
        assert "Default" in persona_names

    @pytest.mark.api
    def test_discussion_mixed_provider_configs(
        self, lmstudio_provider, council_engine_factory
    ):
        """Mix of personas with and without provider_config in full session."""
        from llm_council.models import PersonaProviderConfig, Persona

        # One persona with custom config
        custom_config = PersonaProviderConfig(
            temperature=0.9,
            max_tokens=600,
        )
        custom_persona = Persona(
            name="Custom",
            role="Custom Config User",
            expertise=["customization"],
            personality_traits=["unique"],
            perspective="Use custom settings",
            provider_config=custom_config,
        )

        # Two personas without custom config
        default1 = Persona(
            name="DefaultOne",
            role="First Default",
            expertise=["general"],
            personality_traits=["standard"],
            perspective="Standard approach",
            provider_config=None,
        )
        default2 = Persona(
            name="DefaultTwo",
            role="Second Default",
            expertise=["general"],
            personality_traits=["standard"],
            perspective="Another standard view",
            provider_config=None,
        )

        personas = [custom_persona, default1, default2]

        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Mixed Config Test",
            objective="Verify mixed per-persona and default configs work together",
            personas=personas,
        )

        # Session should complete successfully
        assert len(session.rounds) >= 1
        assert len(session.personas) == 3

        # All personas should have participated
        first_round = session.rounds[0]
        persona_names = {m.persona_name for m in first_round.messages}
        assert "Custom" in persona_names
        assert "DefaultOne" in persona_names
        assert "DefaultTwo" in persona_names

        # Each message should have actual content
        for msg in first_round.messages:
            assert len(msg.content) > 0
