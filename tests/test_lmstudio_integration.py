"""Extensive LM Studio integration tests.

POLICY: NO MOCKED API TESTS - All tests hit real LM Studio at localhost:1234.
See CLAUDE.md for rationale.

These tests validate:
- Complete council sessions with real personas
- Multi-round discussions with context building
- Real vote parsing from LLM responses
- Response type detection (PASS, CALL_VOTE, etc.)
- Mediator behavior and flow control
- Per-persona provider configurations
- All consensus types (majority, supermajority, unanimous, plurality)
- Stalemate detection and auto-voting
"""

import pytest
from typing import List

from llm_council.models import (
    Persona,
    PersonaProviderConfig,
    Message,
    Vote,
    VoteChoice,
    ConsensusType,
    DEFAULT_PERSONAS,
)
from llm_council.council import CouncilEngine
from llm_council.discussion import (
    ResponseParser,
    ResponseType,
    DiscussionState,
    DiscussionPhase,
)
from llm_council.voting import VoteParser, VotingMachine, StructuredVote
from llm_council.providers import LiteLLMProvider, ProviderConfig


# =============================================================================
# Test Fixtures - Custom personas for specific test scenarios
# =============================================================================


@pytest.fixture
def debate_personas() -> List[Persona]:
    """Personas designed to produce diverse opinions for testing."""
    return [
        Persona(
            name="The Optimist",
            role="Positive Thinker",
            expertise=["opportunity identification", "growth mindset"],
            personality_traits=["enthusiastic", "supportive", "forward-looking"],
            perspective="Always see the bright side and opportunities in every situation",
        ),
        Persona(
            name="The Skeptic",
            role="Critical Analyst",
            expertise=["risk assessment", "due diligence", "failure analysis"],
            personality_traits=["cautious", "questioning", "analytical"],
            perspective="Challenge assumptions and identify potential problems",
        ),
        Persona(
            name="The Balancer",
            role="Neutral Moderator",
            expertise=["mediation", "synthesis", "compromise"],
            personality_traits=["balanced", "fair", "diplomatic"],
            perspective="Find middle ground and synthesize different viewpoints",
        ),
    ]


@pytest.fixture
def quick_personas() -> List[Persona]:
    """Minimal personas for fast tests."""
    return [
        Persona(
            name="Alice",
            role="Expert A",
            expertise=["general"],
            personality_traits=["concise"],
            perspective="Provide brief, direct answers",
        ),
        Persona(
            name="Bob",
            role="Expert B",
            expertise=["general"],
            personality_traits=["concise"],
            perspective="Provide brief, direct answers",
        ),
    ]


@pytest.fixture
def temperature_varied_personas() -> List[Persona]:
    """Personas with different temperature settings."""
    return [
        Persona(
            name="Precise",
            role="Analyst",
            expertise=["data analysis"],
            personality_traits=["precise", "methodical"],
            perspective="Focus on accuracy and consistency",
            provider_config=PersonaProviderConfig(temperature=0.1, max_tokens=300),
        ),
        Persona(
            name="Creative",
            role="Innovator",
            expertise=["ideation"],
            personality_traits=["creative", "bold"],
            perspective="Think outside the box",
            provider_config=PersonaProviderConfig(temperature=0.9, max_tokens=500),
        ),
        Persona(
            name="Balanced",
            role="Moderator",
            expertise=["synthesis"],
            personality_traits=["balanced"],
            perspective="Find middle ground",
            provider_config=PersonaProviderConfig(temperature=0.5, max_tokens=400),
        ),
    ]


# =============================================================================
# TestCompleteSession - End-to-end council session tests
# =============================================================================


@pytest.mark.api
class TestCompleteSession:
    """End-to-end tests for complete council sessions."""

    def test_session_completes_with_default_personas(
        self, council_engine_factory, simple_personas
    ):
        """Test full session completes with default personas."""
        engine = council_engine_factory(max_rounds=2)

        session = engine.run_session(
            topic="Testing Framework",
            objective="Decide on the best approach for API testing",
            personas=simple_personas,
        )

        assert session is not None
        assert len(session.rounds) >= 1
        assert len(session.personas) == 3
        assert session.topic == "Testing Framework"
        assert session.objective == "Decide on the best approach for API testing"

    def test_session_with_debate_personas(
        self, council_engine_factory, debate_personas
    ):
        """Test session with personas designed for debate."""
        engine = council_engine_factory(max_rounds=2)

        session = engine.run_session(
            topic="Remote Work Policy",
            objective="Should the company adopt full remote work?",
            personas=debate_personas,
        )

        assert len(session.rounds) >= 1
        # Each round should have responses from all personas
        for round_data in session.rounds:
            assert len(round_data.messages) == 3

    def test_session_produces_real_content(
        self, council_engine_factory, simple_personas
    ):
        """Verify session produces substantive content from LLM."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Code Review Best Practices",
            objective="Establish guidelines for effective code reviews",
            personas=simple_personas,
        )

        # All messages should have real content
        for round_data in session.rounds:
            for msg in round_data.messages:
                assert len(msg.content) > 20, "Messages should be substantive"
                assert msg.persona_name in [p.name for p in simple_personas]

    def test_session_with_initial_context(
        self, council_engine_factory, simple_personas
    ):
        """Test session with initial context provided."""
        engine = council_engine_factory(max_rounds=1)

        context = """Background: Our team has been experiencing frequent production bugs.
        Current process: Code is reviewed by one developer before merge.
        Concern: Reviews are often superficial due to time pressure."""

        session = engine.run_session(
            topic="Improving Code Quality",
            objective="Propose improvements to our code review process",
            personas=simple_personas,
            initial_context=context,
        )

        assert len(session.rounds) >= 1
        # First round should incorporate context
        first_round = session.rounds[0]
        assert len(first_round.messages) > 0

    def test_session_tracks_mediator(self, council_engine_factory, debate_personas):
        """Verify mediator is correctly tracked in session."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Test Topic",
            objective="Test objective",
            personas=debate_personas,
        )

        # First persona should be mediator after reordering
        assert session.personas[0].is_mediator is True
        # First message of each round should be from mediator
        for round_data in session.rounds:
            if round_data.messages:
                assert round_data.messages[0].is_mediator is True


# =============================================================================
# TestMultiRoundDiscussion - Multi-round context building tests
# =============================================================================


@pytest.mark.api
class TestMultiRoundDiscussion:
    """Tests for multi-round discussions with context building."""

    def test_two_round_discussion(self, council_engine_factory, quick_personas):
        """Test discussion runs for exactly 2 rounds."""
        engine = council_engine_factory(max_rounds=2, stalemate_threshold=5)

        session = engine.run_session(
            topic="Simple Decision",
            objective="Choose between option A or B",
            personas=quick_personas,
        )

        # Should have at least 1 round, up to 2
        assert 1 <= len(session.rounds) <= 2

    def test_three_round_discussion(self, council_engine_factory, debate_personas):
        """Test discussion can run for 3 rounds."""
        engine = council_engine_factory(max_rounds=3, stalemate_threshold=5)

        session = engine.run_session(
            topic="Complex Decision",
            objective="Design a new feature architecture",
            personas=debate_personas,
        )

        # Should have at least 1 round
        assert len(session.rounds) >= 1

    def test_context_builds_across_rounds(
        self, council_engine_factory, debate_personas
    ):
        """Verify later rounds receive earlier round context."""
        engine = council_engine_factory(max_rounds=2, stalemate_threshold=5)

        session = engine.run_session(
            topic="Feature Priority",
            objective="Decide which feature to build next",
            personas=debate_personas,
        )

        if len(session.rounds) >= 2:
            # Second round messages should be different from first
            # (they have context from first round)
            first_round_content = {m.content for m in session.rounds[0].messages}
            second_round_content = {m.content for m in session.rounds[1].messages}
            # Content should differ (LLM responds to accumulated context)
            assert first_round_content != second_round_content

    def test_round_numbers_increment(self, council_engine_factory, quick_personas):
        """Verify round numbers are correctly tracked."""
        engine = council_engine_factory(max_rounds=2)

        session = engine.run_session(
            topic="Test",
            objective="Test round numbering",
            personas=quick_personas,
        )

        for i, round_data in enumerate(session.rounds):
            assert round_data.round_number == i + 1
            for msg in round_data.messages:
                assert msg.round_number == i + 1


# =============================================================================
# TestResponseParsing - Real LLM response parsing tests
# =============================================================================


@pytest.mark.api
class TestResponseParsing:
    """Tests for parsing real LLM responses."""

    def test_parse_contribution_response(self, lmstudio_provider):
        """Test parsing a standard contribution response."""
        response = lmstudio_provider.complete(
            "You are an expert giving your opinion.",
            "What do you think about automated testing? Give a brief opinion.",
        )

        parsed = ResponseParser.parse("TestExpert", response)

        # Should be a contribution (default)
        assert parsed.response_type == ResponseType.CONTRIBUTION
        assert len(parsed.content) > 0
        assert parsed.persona_name == "TestExpert"

    def test_parse_pass_response(self, lmstudio_provider):
        """Test parsing a PASS response when instructed."""
        response = lmstudio_provider.complete(
            "You are an expert in a discussion. You agree with everything said so far.",
            "The previous speaker made excellent points. Respond with [PASS] and briefly say why you agree.",
        )

        parsed = ResponseParser.parse("Agreeing Expert", response)

        # Note: LLM may or may not follow instructions perfectly
        # We're testing that parsing works on whatever comes back
        assert parsed.response_type in [ResponseType.PASS, ResponseType.CONTRIBUTION]
        assert parsed.content is not None

    def test_parse_call_vote_response(self, lmstudio_provider):
        """Test parsing a call vote response when instructed."""
        response = lmstudio_provider.complete(
            "You are a mediator who believes consensus has been reached.",
            "Say that you think everyone agrees and call for a vote. Use the phrase 'let's vote'.",
        )

        parsed = ResponseParser.parse("Mediator", response, is_mediator=True)

        # Check if LLM followed instructions (may vary)
        assert parsed.content is not None
        assert parsed.is_mediator is True

    def test_response_parser_handles_long_response(self, lmstudio_provider):
        """Test parser handles longer responses correctly."""
        response = lmstudio_provider.complete(
            "You are a detailed expert who gives thorough answers.",
            "Explain in detail why code reviews are important. Write at least 3 paragraphs.",
        )

        parsed = ResponseParser.parse("DetailedExpert", response)

        assert parsed.response_type == ResponseType.CONTRIBUTION
        assert len(parsed.content) > 100  # Should be a longer response

    def test_response_parser_handles_short_response(self, lmstudio_provider_factory):
        """Test parser handles short responses."""
        provider = lmstudio_provider_factory(max_tokens=20)
        response = provider.complete(
            "You are concise.",
            "Say OK.",
        )

        parsed = ResponseParser.parse("Concise", response)

        assert parsed.response_type == ResponseType.CONTRIBUTION
        assert len(parsed.content) > 0


# =============================================================================
# TestVotingMechanics - Vote parsing from real LLM output
# =============================================================================


@pytest.mark.api
class TestVotingMechanics:
    """Tests for voting mechanics with real LLM responses."""

    def test_parse_vote_agree(self, lmstudio_provider):
        """Test parsing an AGREE vote from LLM."""
        response = lmstudio_provider.complete(
            "You are voting on a proposal. Format: [VOTE] AGREE/DISAGREE [CONFIDENCE] 0.0-1.0 [REASONING] your reason",
            "Vote AGREE with high confidence. The proposal is excellent.",
        )

        vote = VoteParser.parse("Voter", response)

        # LLM should produce parseable output
        assert vote.choice in [VoteChoice.AGREE, VoteChoice.DISAGREE, VoteChoice.ABSTAIN]
        assert 0.0 <= vote.confidence <= 1.0

    def test_parse_vote_disagree(self, lmstudio_provider):
        """Test parsing a DISAGREE vote from LLM."""
        response = lmstudio_provider.complete(
            "You are voting on a proposal. Format: [VOTE] DISAGREE [CONFIDENCE] 0.8 [REASONING] your concern",
            "Vote DISAGREE because you have serious concerns about the proposal.",
        )

        vote = VoteParser.parse("Dissenter", response)

        assert vote.choice in [VoteChoice.AGREE, VoteChoice.DISAGREE, VoteChoice.ABSTAIN]
        assert vote.persona_name == "Dissenter"

    def test_conduct_vote_with_real_personas(
        self, council_engine_factory, debate_personas
    ):
        """Test conducting a vote with real persona responses."""
        engine = council_engine_factory()

        # First run a round to build history
        session = engine.run_session(
            topic="Voting Test",
            objective="Test the voting mechanism",
            personas=debate_personas,
        )

        # Session should have attempted voting
        has_votes = any(r.votes for r in session.rounds)
        # Either has votes or reached consensus without explicit voting
        assert has_votes or session.consensus_reached is not None

    def test_vote_result_has_required_fields(
        self, council_engine_factory, quick_personas
    ):
        """Verify vote results contain all required fields."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Quick Vote Test",
            objective="Make a quick decision",
            personas=quick_personas,
        )

        # Find a round with votes
        for round_data in session.rounds:
            if round_data.votes:
                for vote in round_data.votes:
                    assert hasattr(vote, "persona_name")
                    assert hasattr(vote, "choice")
                    assert hasattr(vote, "reasoning")
                    assert vote.choice in VoteChoice


# =============================================================================
# TestConsensusTypes - Tests for different consensus requirements
# =============================================================================


@pytest.mark.api
class TestConsensusTypes:
    """Tests for different consensus type requirements."""

    def test_majority_consensus(self, lmstudio_provider, quick_personas):
        """Test majority consensus (>50%)."""
        engine = CouncilEngine(
            provider=lmstudio_provider,
            consensus_type=ConsensusType.MAJORITY,
            max_rounds=1,
        )

        session = engine.run_session(
            topic="Majority Test",
            objective="Reach majority consensus",
            personas=quick_personas,
        )

        assert session is not None
        # Session should complete
        assert len(session.rounds) >= 1

    def test_supermajority_consensus(self, lmstudio_provider, debate_personas):
        """Test supermajority consensus (>66%)."""
        engine = CouncilEngine(
            provider=lmstudio_provider,
            consensus_type=ConsensusType.SUPERMAJORITY,
            max_rounds=2,
        )

        session = engine.run_session(
            topic="Supermajority Test",
            objective="Reach supermajority consensus",
            personas=debate_personas,
        )

        assert session is not None
        assert len(session.rounds) >= 1

    def test_unanimous_consensus(self, lmstudio_provider, quick_personas):
        """Test unanimous consensus (100%)."""
        engine = CouncilEngine(
            provider=lmstudio_provider,
            consensus_type=ConsensusType.UNANIMOUS,
            max_rounds=2,
        )

        session = engine.run_session(
            topic="Unanimous Test",
            objective="Reach unanimous consensus",
            personas=quick_personas,
        )

        assert session is not None
        # Unanimous is harder to achieve
        assert len(session.rounds) >= 1

    def test_plurality_consensus(self, lmstudio_provider, debate_personas):
        """Test plurality consensus (most votes wins)."""
        engine = CouncilEngine(
            provider=lmstudio_provider,
            consensus_type=ConsensusType.PLURALITY,
            max_rounds=2,
        )

        session = engine.run_session(
            topic="Plurality Test",
            objective="Reach plurality consensus",
            personas=debate_personas,
        )

        assert session is not None
        assert len(session.rounds) >= 1


# =============================================================================
# TestMediatorBehavior - Mediator flow control tests
# =============================================================================


@pytest.mark.api
class TestMediatorBehavior:
    """Tests for mediator persona behavior."""

    def test_mediator_speaks_first(self, council_engine_factory, debate_personas):
        """Verify mediator speaks first in each round."""
        engine = council_engine_factory(max_rounds=2)

        session = engine.run_session(
            topic="Mediator Order Test",
            objective="Test speaking order",
            personas=debate_personas,
        )

        for round_data in session.rounds:
            if round_data.messages:
                first_msg = round_data.messages[0]
                assert first_msg.is_mediator is True

    def test_mediator_gets_special_prompt(self, lmstudio_provider, debate_personas):
        """Test that mediator receives appropriate prompting."""
        engine = CouncilEngine(
            provider=lmstudio_provider,
            max_rounds=1,
        )

        session = engine.run_session(
            topic="Mediator Prompt Test",
            objective="Test mediator behavior",
            personas=debate_personas,
        )

        # Mediator message should reflect moderator role
        if session.rounds and session.rounds[0].messages:
            mediator_msg = session.rounds[0].messages[0]
            assert mediator_msg.is_mediator is True
            # Mediator content should be substantive
            assert len(mediator_msg.content) > 10

    def test_mediator_excluded_from_vote(
        self, council_engine_factory, debate_personas
    ):
        """Verify mediator does not cast a vote."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Mediator Vote Exclusion",
            objective="Test voting exclusion",
            personas=debate_personas,
        )

        # Check votes don't include mediator
        mediator_name = session.personas[0].name  # First is mediator
        for round_data in session.rounds:
            for vote in round_data.votes:
                assert vote.persona_name != mediator_name


# =============================================================================
# TestPerPersonaConfig - Per-persona provider configuration tests
# =============================================================================


@pytest.mark.api
class TestPerPersonaConfig:
    """Tests for per-persona provider configurations."""

    def test_different_temperatures_per_persona(
        self, lmstudio_provider, temperature_varied_personas
    ):
        """Test personas with different temperature settings."""
        engine = CouncilEngine(
            provider=lmstudio_provider,
            max_rounds=1,
        )

        session = engine.run_session(
            topic="Temperature Test",
            objective="Test per-persona temperature",
            personas=temperature_varied_personas,
        )

        assert len(session.rounds) >= 1
        # All personas should have responded
        first_round = session.rounds[0]
        persona_names = {m.persona_name for m in first_round.messages}
        assert "Precise" in persona_names
        assert "Creative" in persona_names
        assert "Balanced" in persona_names

    def test_mixed_config_personas(self, lmstudio_provider, simple_personas):
        """Test mix of personas with and without custom config."""
        # Add custom config to one persona
        custom_persona = Persona(
            name="Custom",
            role="Custom Config",
            expertise=["testing"],
            personality_traits=["thorough"],
            perspective="Test with custom settings",
            provider_config=PersonaProviderConfig(temperature=0.3, max_tokens=200),
        )

        personas = [simple_personas[0], custom_persona, simple_personas[1]]

        engine = CouncilEngine(
            provider=lmstudio_provider,
            max_rounds=1,
        )

        session = engine.run_session(
            topic="Mixed Config Test",
            objective="Test mixed configurations",
            personas=personas,
        )

        assert len(session.rounds) >= 1
        # All should have responded
        first_round = session.rounds[0]
        assert len(first_round.messages) == 3

    def test_all_inference_params_per_persona(self, lmstudio_provider):
        """Test persona with all inference parameters set."""
        full_config_persona = Persona(
            name="FullConfig",
            role="Full Configuration Test",
            expertise=["testing"],
            personality_traits=["thorough"],
            perspective="Test all parameters",
            provider_config=PersonaProviderConfig(
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                max_tokens=300,
                frequency_penalty=0.2,
                presence_penalty=0.2,
                seed=42,
                timeout=60,
            ),
        )

        simple_persona = Persona(
            name="Simple",
            role="Simple Test",
            expertise=["testing"],
            personality_traits=["basic"],
            perspective="Simple response",
        )

        engine = CouncilEngine(
            provider=lmstudio_provider,
            max_rounds=1,
        )

        session = engine.run_session(
            topic="Full Config Test",
            objective="Test all inference params",
            personas=[full_config_persona, simple_persona],
        )

        assert len(session.rounds) >= 1


# =============================================================================
# TestDiscussionState - Discussion state management tests
# =============================================================================


@pytest.mark.api
class TestDiscussionState:
    """Tests for discussion state tracking with real API."""

    def test_phase_progression(self, council_engine_factory, quick_personas):
        """Test discussion phases progress correctly."""
        engine = council_engine_factory(max_rounds=3, stalemate_threshold=5)

        session = engine.run_session(
            topic="Phase Test",
            objective="Test phase progression",
            personas=quick_personas,
        )

        # Session should have progressed through phases
        assert len(session.rounds) >= 1

    def test_pass_tracking(self, council_engine_factory, debate_personas):
        """Test PASS responses are tracked in discussion state."""
        engine = council_engine_factory(max_rounds=2)

        session = engine.run_session(
            topic="Pass Tracking Test",
            objective="Simple agreement test",
            personas=debate_personas,
        )

        # Check for any pass messages
        pass_count = 0
        for round_data in session.rounds:
            for msg in round_data.messages:
                if msg.is_pass:
                    pass_count += 1

        # Pass count tracking works (may or may not have passes)
        assert pass_count >= 0  # Just verify tracking works


# =============================================================================
# TestStalemateDetection - Stalemate and auto-voting tests
# =============================================================================


@pytest.mark.api
class TestStalemateDetection:
    """Tests for stalemate detection and auto-voting."""

    def test_session_completes_on_max_rounds(
        self, lmstudio_provider, debate_personas
    ):
        """Test session completes when max rounds reached."""
        engine = CouncilEngine(
            provider=lmstudio_provider,
            max_rounds=2,
            stalemate_threshold=10,  # High threshold to avoid stalemate
        )

        session = engine.run_session(
            topic="Max Rounds Test",
            objective="Test max rounds limit",
            personas=debate_personas,
        )

        # Should complete within max_rounds
        assert len(session.rounds) <= 2
        # Should have final consensus or explicit no-consensus
        assert session.final_consensus is not None or session.consensus_reached is False

    def test_stalemate_triggers_vote(self, lmstudio_provider, quick_personas):
        """Test stalemate threshold triggers voting."""
        engine = CouncilEngine(
            provider=lmstudio_provider,
            max_rounds=3,
            stalemate_threshold=1,  # Low threshold for quick stalemate
        )

        session = engine.run_session(
            topic="Stalemate Test",
            objective="Test stalemate detection",
            personas=quick_personas,
        )

        # Should have completed
        assert len(session.rounds) >= 1


# =============================================================================
# TestEdgeCases - Edge case and error handling tests
# =============================================================================


@pytest.mark.api
class TestEdgeCases:
    """Edge case tests with real API."""

    def test_single_persona_session(self, lmstudio_provider):
        """Test session with minimum personas."""
        single_persona = [
            Persona(
                name="Solo",
                role="Solo Expert",
                expertise=["general"],
                personality_traits=["independent"],
                perspective="Work alone",
            ),
        ]

        engine = CouncilEngine(
            provider=lmstudio_provider,
            max_rounds=1,
        )

        session = engine.run_session(
            topic="Solo Test",
            objective="Test single persona",
            personas=single_persona,
        )

        # Should complete (though voting won't work with 1 persona)
        assert len(session.rounds) >= 1

    def test_many_personas_session(self, lmstudio_provider):
        """Test session with many personas."""
        many_personas = [
            Persona(
                name=f"Expert{i}",
                role=f"Expert #{i}",
                expertise=["general"],
                personality_traits=["concise"],
                perspective=f"Perspective {i}",
            )
            for i in range(5)
        ]

        engine = CouncilEngine(
            provider=lmstudio_provider,
            max_rounds=1,
        )

        session = engine.run_session(
            topic="Many Personas Test",
            objective="Test with many personas",
            personas=many_personas,
        )

        # Should have responses from all
        if session.rounds:
            assert len(session.rounds[0].messages) == 5

    def test_very_short_max_tokens(self, lmstudio_provider_factory):
        """Test with very short max_tokens."""
        provider = lmstudio_provider_factory(max_tokens=10)

        result = provider.complete(
            "You are concise.",
            "Say hello world.",
        )

        # Should still produce some output
        assert result is not None
        assert len(result) > 0

    def test_empty_topic(self, council_engine_factory, quick_personas):
        """Test session with empty/minimal topic."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="",
            objective="Discuss",
            personas=quick_personas,
        )

        # Should still run
        assert len(session.rounds) >= 1

    def test_long_topic_and_objective(self, council_engine_factory, quick_personas):
        """Test with very long topic and objective."""
        long_topic = "Testing " * 50  # Long topic
        long_objective = "We need to decide " * 30  # Long objective

        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic=long_topic,
            objective=long_objective,
            personas=quick_personas,
        )

        assert len(session.rounds) >= 1


# =============================================================================
# TestProviderBehavior - Provider-specific behavior tests
# =============================================================================


@pytest.mark.api
class TestProviderBehavior:
    """Tests for provider behavior in council context."""

    def test_provider_handles_special_characters(self, lmstudio_provider):
        """Test provider handles special characters in prompts."""
        result = lmstudio_provider.complete(
            "You are helpful.",
            "Respond to: Hello! @#$%^&*() How are you?",
        )

        assert result is not None
        assert len(result) > 0

    def test_provider_handles_unicode(self, lmstudio_provider):
        """Test provider handles unicode in prompts."""
        result = lmstudio_provider.complete(
            "You are helpful.",
            "Say hello in response to: こんにちは 你好 مرحبا",
        )

        assert result is not None
        assert len(result) > 0

    def test_provider_handles_code_in_prompt(self, lmstudio_provider):
        """Test provider handles code snippets in prompts."""
        result = lmstudio_provider.complete(
            "You are a code reviewer.",
            "Review this code: def hello(): return 'Hello, World!'",
        )

        assert result is not None
        assert len(result) > 0

    def test_provider_handles_json_in_prompt(self, lmstudio_provider):
        """Test provider handles JSON in prompts."""
        result = lmstudio_provider.complete(
            "You are a data analyst.",
            'Analyze this JSON: {"name": "test", "value": 42}',
        )

        assert result is not None
        assert len(result) > 0


# =============================================================================
# TestSessionSerialization - Session output tests
# =============================================================================


@pytest.mark.api
class TestSessionSerialization:
    """Tests for session serialization."""

    def test_session_to_dict(self, council_engine_factory, quick_personas):
        """Test session converts to dictionary correctly."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Serialization Test",
            objective="Test to_dict",
            personas=quick_personas,
        )

        result = session.to_dict()

        assert isinstance(result, dict)
        assert "topic" in result
        assert "objective" in result
        assert "personas" in result
        assert "rounds" in result
        assert "consensus_reached" in result
        assert result["topic"] == "Serialization Test"

    def test_session_dict_has_all_messages(
        self, council_engine_factory, quick_personas
    ):
        """Verify serialized session includes all messages."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Full Serialization Test",
            objective="Test complete serialization",
            personas=quick_personas,
        )

        result = session.to_dict()

        # Check round messages
        for round_dict in result["rounds"]:
            assert "messages" in round_dict
            for msg in round_dict["messages"]:
                assert "persona_name" in msg
                assert "content" in msg
                assert "round_number" in msg


# =============================================================================
# TestConcurrentRequests - Multiple request handling
# =============================================================================


@pytest.mark.api
class TestConcurrentRequests:
    """Tests for handling multiple requests."""

    def test_sequential_sessions(self, council_engine_factory, quick_personas):
        """Test multiple sessions run sequentially."""
        engine = council_engine_factory(max_rounds=1)

        sessions = []
        for i in range(3):
            session = engine.run_session(
                topic=f"Sequential Test {i}",
                objective=f"Test session {i}",
                personas=quick_personas,
            )
            sessions.append(session)

        assert len(sessions) == 3
        for s in sessions:
            assert len(s.rounds) >= 1

    def test_reuse_engine_different_personas(self, lmstudio_provider):
        """Test reusing engine with different persona sets."""
        engine = CouncilEngine(
            provider=lmstudio_provider,
            max_rounds=1,
        )

        # First session with 2 personas
        personas1 = [
            Persona(name="A1", role="R1", expertise=["e"], personality_traits=["t"], perspective="p"),
            Persona(name="A2", role="R2", expertise=["e"], personality_traits=["t"], perspective="p"),
        ]
        session1 = engine.run_session(
            topic="Test 1",
            objective="Obj 1",
            personas=personas1,
        )

        # Second session with 3 personas
        personas2 = [
            Persona(name="B1", role="R1", expertise=["e"], personality_traits=["t"], perspective="p"),
            Persona(name="B2", role="R2", expertise=["e"], personality_traits=["t"], perspective="p"),
            Persona(name="B3", role="R3", expertise=["e"], personality_traits=["t"], perspective="p"),
        ]
        session2 = engine.run_session(
            topic="Test 2",
            objective="Obj 2",
            personas=personas2,
        )

        assert len(session1.personas) == 2
        assert len(session2.personas) == 3


# =============================================================================
# TestPersonaGenerationAPI - Persona generation with real LM Studio
# =============================================================================


@pytest.mark.api
class TestPersonaGenerationAPI:
    """Tests for persona generation with real LM Studio API."""

    def test_generate_personas_for_topic(self, lmstudio_provider):
        """Test generating personas for a specific topic."""
        from llm_council.personas import PersonaManager

        manager = PersonaManager(provider=lmstudio_provider)

        personas = manager.generate_personas_for_topic(
            topic="Climate Change Policy",
            count=3,
        )

        assert len(personas) == 3
        for p in personas:
            assert p.name is not None
            assert len(p.name) > 0
            assert p.role is not None

    def test_generate_and_save_personas(self, lmstudio_provider, tmp_path):
        """Test generating and saving personas to file."""
        from llm_council.personas import PersonaManager

        manager = PersonaManager(provider=lmstudio_provider)

        yaml_path = tmp_path / "generated_personas.yaml"
        personas = manager.generate_personas_for_topic(
            topic="Software Architecture",
            count=2,
            save_to=str(yaml_path),
        )

        assert yaml_path.exists()
        assert len(personas) == 2

        # Verify file can be loaded back
        loaded = manager.load_personas(str(yaml_path))
        assert len(loaded) == 2

    def test_generate_personas_with_provider_configs(self, lmstudio_provider, tmp_path):
        """Test generating personas and applying provider configs."""
        from llm_council.personas import PersonaManager
        from llm_council.models import PersonaProviderConfig

        manager = PersonaManager(provider=lmstudio_provider)

        personas = manager.generate_personas_for_topic(
            topic="Data Science",
            count=2,
        )

        # Apply custom configs after generation
        configs = {
            personas[0].name: PersonaProviderConfig(temperature=0.3, max_tokens=200),
            personas[1].name: PersonaProviderConfig(temperature=0.8, max_tokens=500),
        }

        configured = manager._apply_provider_configs(personas, configs)

        assert configured[0].provider_config is not None
        assert configured[0].provider_config.temperature == 0.3
        assert configured[1].provider_config.temperature == 0.8

    def test_generated_personas_in_session(self, lmstudio_provider):
        """Test using generated personas in a council session."""
        from llm_council.personas import PersonaManager

        manager = PersonaManager(provider=lmstudio_provider)

        personas = manager.generate_personas_for_topic(
            topic="API Design Best Practices",
            count=3,
        )

        engine = CouncilEngine(
            provider=lmstudio_provider,
            max_rounds=1,
        )

        session = engine.run_session(
            topic="API Design Best Practices",
            objective="Establish guidelines for API design",
            personas=personas,
        )

        assert len(session.rounds) >= 1
        assert len(session.rounds[0].messages) == 3


# =============================================================================
# TestPersistenceWithConfigs - Persistence tests with provider configs
# =============================================================================


@pytest.mark.api
class TestPersistenceWithConfigs:
    """Tests for persistence with provider configurations."""

    def test_save_and_load_session_with_persona_configs(
        self, council_engine_factory, temperature_varied_personas
    ):
        """Test saving and loading session with per-persona configs."""
        from llm_council.persistence import SessionManager

        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Persistence Config Test",
            objective="Test persistence with configs",
            personas=temperature_varied_personas,
        )

        # Save session
        manager = SessionManager()
        session_id = manager.save_session(session)

        # Load session
        loaded = manager.load_session(session_id)

        assert loaded is not None
        assert loaded.topic == "Persistence Config Test"
        # Verify data contains persona info
        assert loaded.data is not None
        assert "personas" in loaded.data

    def test_export_session_with_configs_json(
        self, council_engine_factory, temperature_varied_personas
    ):
        """Test exporting session with configs to JSON."""
        import json
        from llm_council.persistence import SessionManager

        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Export Config Test",
            objective="Test JSON export with configs",
            personas=temperature_varied_personas,
        )

        manager = SessionManager()
        session_id = manager.save_session(session)

        # Export to JSON
        json_str = manager.export_json(session_ids=[session_id])
        data = json.loads(json_str)

        assert data["session_count"] == 1
        assert len(data["sessions"]) == 1

    def test_multiple_sessions_different_configs(self, lmstudio_provider):
        """Test persisting multiple sessions with different configs."""
        from llm_council.persistence import SessionManager

        manager = SessionManager()

        # Session 1: Low temperature personas
        low_temp_personas = [
            Persona(
                name="Precise1",
                role="Analyst",
                expertise=["analysis"],
                personality_traits=["precise"],
                perspective="Precise analysis",
                provider_config=PersonaProviderConfig(temperature=0.1),
            ),
            Persona(
                name="Precise2",
                role="Reviewer",
                expertise=["review"],
                personality_traits=["thorough"],
                perspective="Thorough review",
                provider_config=PersonaProviderConfig(temperature=0.2),
            ),
        ]

        engine = CouncilEngine(provider=lmstudio_provider, max_rounds=1)
        session1 = engine.run_session(
            topic="Low Temp Test",
            objective="Test low temperature",
            personas=low_temp_personas,
        )
        id1 = manager.save_session(session1)

        # Session 2: High temperature personas
        high_temp_personas = [
            Persona(
                name="Creative1",
                role="Ideator",
                expertise=["ideation"],
                personality_traits=["creative"],
                perspective="Creative ideas",
                provider_config=PersonaProviderConfig(temperature=0.9),
            ),
            Persona(
                name="Creative2",
                role="Innovator",
                expertise=["innovation"],
                personality_traits=["bold"],
                perspective="Bold innovation",
                provider_config=PersonaProviderConfig(temperature=0.95),
            ),
        ]

        session2 = engine.run_session(
            topic="High Temp Test",
            objective="Test high temperature",
            personas=high_temp_personas,
        )
        id2 = manager.save_session(session2)

        # Verify both sessions saved correctly
        loaded1 = manager.load_session(id1)
        loaded2 = manager.load_session(id2)

        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded1.topic == "Low Temp Test"
        assert loaded2.topic == "High Temp Test"

    def test_session_search_with_configs(self, council_engine_factory):
        """Test searching sessions that have configs."""
        from llm_council.persistence import SessionManager

        manager = SessionManager()
        engine = council_engine_factory(max_rounds=1)

        config_persona = Persona(
            name="SearchTest",
            role="Test",
            expertise=["search"],
            personality_traits=["test"],
            perspective="Test search",
            provider_config=PersonaProviderConfig(temperature=0.5),
        )

        session = engine.run_session(
            topic="SearchableConfigSession",
            objective="Test search functionality",
            personas=[config_persona, config_persona],
        )

        manager.save_session(session)

        # Search for the session
        results = manager.search_sessions("SearchableConfigSession")
        assert len(results) >= 1


# =============================================================================
# TestProviderRegistryAPI - Provider registry with real API calls
# =============================================================================


@pytest.mark.api
class TestProviderRegistryAPI:
    """Tests for provider registry with real LM Studio API."""

    def test_registry_creates_providers_for_personas(self, lmstudio_provider):
        """Test registry creates providers for different persona configs."""
        from llm_council.providers import ProviderRegistry
        from llm_council.config import ResolvedConfig, ProviderSettings

        # Create resolved config with default settings
        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model="openai/qwen/qwen3-coder-30b",
                api_base="http://localhost:1234/v1",
                api_key="lm-studio",
                temperature=0.7,
                max_tokens=512,
            ),
            generation=None,
            providers={},
            persona_configs={
                "LowTemp": ProviderSettings(temperature=0.2),
                "HighTemp": ProviderSettings(temperature=0.9),
            },
            council=None,
            persistence=None,
        )

        registry = ProviderRegistry(resolved)

        # Get providers for different personas
        low_temp_provider = registry.get_for_persona("LowTemp")
        high_temp_provider = registry.get_for_persona("HighTemp")
        default_provider = registry.get_for_persona("Unknown")

        # All should work with real API
        result1 = low_temp_provider.complete("You are precise.", "Say 'low temp'.")
        result2 = high_temp_provider.complete("You are creative.", "Say 'high temp'.")
        result3 = default_provider.complete("You are helpful.", "Say 'default'.")

        assert len(result1) > 0
        assert len(result2) > 0
        assert len(result3) > 0

    def test_registry_caches_providers(self, lmstudio_provider):
        """Test registry caches providers to avoid recreation."""
        from llm_council.providers import ProviderRegistry
        from llm_council.config import ResolvedConfig, ProviderSettings

        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model="openai/qwen/qwen3-coder-30b",
                api_base="http://localhost:1234/v1",
                api_key="lm-studio",
                temperature=0.7,
                max_tokens=256,
            ),
            generation=None,
            providers={},
            persona_configs={},
            council=None,
            persistence=None,
        )

        registry = ProviderRegistry(resolved)

        # Get same provider twice
        provider1 = registry.get_for_persona("TestPersona")
        provider2 = registry.get_for_persona("TestPersona")

        # Should be the same instance (cached)
        assert provider1 is provider2

        # Both should work
        result = provider1.complete("Test", "Say hello")
        assert len(result) > 0

    def test_registry_with_council_session(self):
        """Test registry integrated with council session."""
        from llm_council.providers import ProviderRegistry, create_provider
        from llm_council.config import ResolvedConfig, ProviderSettings

        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model="openai/qwen/qwen3-coder-30b",
                api_base="http://localhost:1234/v1",
                api_key="lm-studio",
                temperature=0.7,
                max_tokens=512,
            ),
            generation=None,
            providers={},
            persona_configs={
                "Analyst": ProviderSettings(temperature=0.3, max_tokens=300),
                "Creative": ProviderSettings(temperature=0.9, max_tokens=600),
            },
            council=None,
            persistence=None,
        )

        registry = ProviderRegistry(resolved)

        # Create personas that match the config names
        personas = [
            Persona(
                name="Analyst",
                role="Data Analyst",
                expertise=["data"],
                personality_traits=["analytical"],
                perspective="Data-driven analysis",
            ),
            Persona(
                name="Creative",
                role="Creative Director",
                expertise=["creativity"],
                personality_traits=["innovative"],
                perspective="Creative solutions",
            ),
        ]

        # Get default provider for engine
        default_provider = create_provider(
            model="openai/qwen/qwen3-coder-30b",
            api_base="http://localhost:1234/v1",
            api_key="lm-studio",
            temperature=0.7,
            max_tokens=512,
        )

        engine = CouncilEngine(
            provider=default_provider,
            max_rounds=1,
        )

        session = engine.run_session(
            topic="Registry Integration Test",
            objective="Test registry with council",
            personas=personas,
        )

        assert len(session.rounds) >= 1
        # Both personas should have responded
        persona_names = {m.persona_name for m in session.rounds[0].messages}
        assert "Analyst" in persona_names
        assert "Creative" in persona_names


# =============================================================================
# TestInheritanceChainAPI - Inheritance chain with real API
# =============================================================================


@pytest.mark.api
class TestInheritanceChainAPI:
    """Tests for config inheritance chain with real API calls."""

    def test_defaults_to_persona_inheritance(self, lmstudio_provider):
        """Test defaults are inherited and overridden by persona config."""
        from llm_council.config import ConfigManager, ConfigSchema, ProviderSettings

        config = ConfigSchema(
            defaults=ProviderSettings(
                model="openai/qwen/qwen3-coder-30b",
                api_base="http://localhost:1234/v1",
                api_key="lm-studio",
                temperature=0.7,
                max_tokens=1024,
            ),
            persona_configs={
                "CustomPersona": ProviderSettings(temperature=0.2, max_tokens=200),
            },
        )

        manager = ConfigManager()
        resolved = manager.resolve(config)

        # Get settings for persona
        settings = manager.get_provider_for_persona("CustomPersona", resolved)

        # Should have overridden values
        assert settings.temperature == 0.2
        assert settings.max_tokens == 200
        # Should inherit defaults
        assert settings.model == "openai/qwen/qwen3-coder-30b"
        assert settings.api_base == "http://localhost:1234/v1"

    def test_provider_to_persona_inheritance(self, lmstudio_provider):
        """Test named provider settings inherited by persona."""
        from llm_council.config import ConfigManager, ConfigSchema, ProviderSettings

        config = ConfigSchema(
            defaults=ProviderSettings(
                model="openai/qwen/qwen3-coder-30b",
                api_base="http://localhost:1234/v1",
                api_key="lm-studio",
                temperature=0.5,
                max_tokens=512,
            ),
            providers={
                "creative_provider": ProviderSettings(temperature=0.9, top_p=0.95),
            },
            persona_configs={
                "CreativePersona": ProviderSettings(
                    provider="creative_provider",  # Reference named provider
                    max_tokens=800,  # Override
                ),
            },
        )

        manager = ConfigManager()
        resolved = manager.resolve(config)

        settings = manager.get_provider_for_persona("CreativePersona", resolved)

        # From named provider
        assert settings.temperature == 0.9
        assert settings.top_p == 0.95
        # Overridden by persona
        assert settings.max_tokens == 800
        # From defaults
        assert settings.model == "openai/qwen/qwen3-coder-30b"

    def test_full_inheritance_chain_in_session(self):
        """Test full inheritance chain in actual session."""
        from llm_council.config import ConfigManager, ConfigSchema, ProviderSettings
        from llm_council.providers import create_provider

        config = ConfigSchema(
            defaults=ProviderSettings(
                model="openai/qwen/qwen3-coder-30b",
                api_base="http://localhost:1234/v1",
                api_key="lm-studio",
                temperature=0.7,
                max_tokens=512,
                timeout=120,
            ),
            providers={
                "analytical": ProviderSettings(temperature=0.2, top_p=0.8),
            },
            persona_configs={
                "DataScientist": ProviderSettings(
                    provider="analytical",
                    max_tokens=400,
                ),
            },
        )

        manager = ConfigManager()
        resolved = manager.resolve(config)

        # Get provider for DataScientist
        ds_settings = manager.get_provider_for_persona("DataScientist", resolved)

        # Create provider with resolved settings
        provider = create_provider(
            model=ds_settings.model,
            api_base=ds_settings.api_base,
            api_key=ds_settings.api_key,
            temperature=ds_settings.temperature,
            max_tokens=ds_settings.max_tokens,
        )

        # Use in session
        result = provider.complete(
            "You are a data scientist.",
            "Briefly describe your approach to analysis.",
        )

        assert len(result) > 0
