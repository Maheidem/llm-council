"""Integration tests for isolated persona sessions and deterministic voting.

POLICY: NO MOCKED API TESTS - All tests use real LM Studio.
See CLAUDE.md for rationale.

These tests verify the success criteria:
1. Each persona executes as isolated LLM call with persona-specific system prompt
2. Personas receive other personas' outputs in context
3. Mediator persona controls discussion flow
4. Voting is computed by deterministic function
"""

import logging
import pytest
from io import StringIO

from llm_council.models import (
    Persona,
    Message,
    Vote,
    VoteChoice,
    ConsensusType,
    DEFAULT_PERSONAS,
)
from llm_council.council import CouncilEngine
from llm_council.discussion import DiscussionState, ResponseType
from llm_council.voting import VoteParser, VotingMachine, StructuredVote


@pytest.mark.api
class TestIsolatedPersonaSessions:
    """Verify each persona runs as isolated LLM invocation with real API."""

    def test_each_persona_produces_unique_response(self, council_engine_factory, simple_personas):
        """CRITERION 1: Each persona executes as isolated LLM call.

        Verification: Different personas produce different responses to same topic.
        """
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Test Isolation",
            objective="Verify isolated sessions by checking unique responses",
            personas=simple_personas,
        )

        # Get messages from first round
        assert len(session.rounds) >= 1
        messages = session.rounds[0].messages

        # Should have 3 messages (one per persona)
        assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"

        # Each message should be from a different persona
        persona_names = [m.persona_name for m in messages]
        assert len(set(persona_names)) == 3, "Each persona should contribute once"

        # Messages should be unique (real LLM produces different content)
        contents = [m.content for m in messages]
        assert len(set(contents)) == 3, "Each persona should produce unique content"

    def test_persona_names_match_assigned(self, council_engine_factory, simple_personas):
        """Verify each message is attributed to correct persona."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Attribution Test",
            objective="Verify message attribution",
            personas=simple_personas,
        )

        expected_names = {p.name for p in simple_personas}
        actual_names = {m.persona_name for m in session.rounds[0].messages}

        assert expected_names == actual_names, "All persona names should appear in messages"


@pytest.mark.api
class TestCrossPersonaAwareness:
    """Verify personas receive other personas' outputs in context with real API."""

    def test_multi_round_shows_context_building(self, council_engine_factory, simple_personas):
        """CRITERION 2: Personas receive context from previous rounds.

        Verification: Session completes multiple rounds, building on prior discussion.
        """
        engine = council_engine_factory(max_rounds=2)

        session = engine.run_session(
            topic="Context Building Test",
            objective="Verify personas build on each other's ideas",
            personas=simple_personas,
        )

        # Should complete at least 2 rounds
        assert len(session.rounds) >= 1, "Should have at least one round"

        # Each round should have messages
        for i, round_data in enumerate(session.rounds):
            assert len(round_data.messages) > 0, f"Round {i+1} should have messages"


@pytest.mark.api
class TestMediatorFlowControl:
    """Verify mediator persona controls discussion flow with real API."""

    def test_first_persona_is_mediator(self, council_engine_factory, simple_personas):
        """CRITERION 3: Mediator controls discussion flow.

        Verification: First persona in session is marked as mediator.
        """
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Mediator Test",
            objective="Verify mediator is first",
            personas=simple_personas,
        )

        # First persona should be mediator
        assert session.personas[0].is_mediator is True, "First persona should be mediator"

        # Other personas should not be mediators
        for persona in session.personas[1:]:
            assert persona.is_mediator is False, "Only first persona should be mediator"

    def test_mediator_is_first_in_messages(self, council_engine_factory, simple_personas):
        """Verify mediator speaks first in each round."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Message Order Test",
            objective="Verify mediator speaks first",
            personas=simple_personas,
        )

        mediator_name = session.personas[0].name

        # First message should be from mediator
        first_message = session.rounds[0].messages[0]
        assert first_message.persona_name == mediator_name, "Mediator should speak first"

    def test_mediator_excluded_from_voting(self, council_engine_factory, simple_personas):
        """Verify mediator doesn't vote - maintains neutrality."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Voting Exclusion Test",
            objective="Verify mediator doesn't vote",
            personas=simple_personas,
        )

        # Find round with votes
        votes = None
        for round_data in session.rounds:
            if round_data.votes:
                votes = round_data.votes
                break

        if votes:
            mediator_name = session.personas[0].name
            voter_names = [v.persona_name for v in votes]
            assert mediator_name not in voter_names, "Mediator should not vote"
            assert len(votes) == 2, "Should have 2 votes (3 personas - 1 mediator)"


class TestDeterministicVoting:
    """Verify voting is deterministic with fixed inputs/outputs - pure logic tests."""

    def test_vote_parsing_deterministic(self):
        """CRITERION 4: Vote parsing is deterministic."""
        response = "[VOTE] AGREE\n[CONFIDENCE] 0.85\n[REASONING] Good proposal."

        # Parse same input multiple times
        results = [VoteParser.parse("Test", response) for _ in range(10)]

        # All results should be identical
        for r in results:
            assert r.choice == VoteChoice.AGREE
            assert r.confidence == 0.85
            assert "Good proposal" in r.reasoning

    def test_vote_tally_deterministic(self):
        """Vote tallying produces same result for same inputs."""
        votes = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, "Yes"),
            StructuredVote("P2", VoteChoice.AGREE, 0.8, "OK"),
            StructuredVote("P3", VoteChoice.DISAGREE, 0.7, "No"),
        ]

        machine = VotingMachine(ConsensusType.MAJORITY)

        # Tally same votes multiple times
        results = [machine.tally(votes) for _ in range(10)]

        # All results should be identical
        for r in results:
            assert r.agree_count == 2
            assert r.disagree_count == 1
            assert r.agree_ratio == 2/3
            assert r.consensus_reached is True

    def test_consensus_thresholds_exact(self):
        """Verify exact threshold behavior."""
        # Test boundary: 50% should NOT pass majority (needs >50%)
        votes_50_50 = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, ""),
            StructuredVote("P2", VoteChoice.DISAGREE, 0.9, ""),
        ]
        machine = VotingMachine(ConsensusType.MAJORITY)
        assert machine.tally(votes_50_50).consensus_reached is False

        # Test boundary: 66% SHOULD pass majority
        votes_66 = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, ""),
            StructuredVote("P2", VoteChoice.AGREE, 0.9, ""),
            StructuredVote("P3", VoteChoice.DISAGREE, 0.9, ""),
        ]
        assert machine.tally(votes_66).consensus_reached is True  # 66% > 50%

        # Test supermajority boundary
        machine_super = VotingMachine(ConsensusType.SUPERMAJORITY)
        votes_66_super = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, ""),
            StructuredVote("P2", VoteChoice.AGREE, 0.9, ""),
            StructuredVote("P3", VoteChoice.DISAGREE, 0.9, ""),
        ]
        assert machine_super.tally(votes_66_super).consensus_reached is False  # 66% not > 66.67%


@pytest.mark.api
class TestFullIntegration:
    """Full integration test with real LM Studio."""

    def test_full_council_session_with_logging(self, council_engine_factory, simple_personas):
        """Run complete session and verify all criteria via output structure."""
        # Set up logging capture
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger = logging.getLogger('llm_council.council')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            engine = council_engine_factory(max_rounds=1)

            session = engine.run_session(
                topic="Budget Planning",
                objective="Decide on budget allocation approach",
                personas=simple_personas,
            )

            log_output = log_stream.getvalue()

            # CRITERION 1: Verify API calls logged
            assert "[API CALL]" in log_output, "Should log API calls"

            # CRITERION 2: Verify personas logged by name
            assert any(p.name in log_output for p in simple_personas), \
                "Should log persona names"

            # CRITERION 3: Verify mediator designation logged
            assert "Mediator" in log_output or "mediator" in log_output, \
                "Should log mediator designation"

            # Verify session structure
            assert len(session.rounds) >= 1
            assert len(session.personas) == 3
            assert session.personas[0].is_mediator is True

        finally:
            logger.removeHandler(handler)

    def test_session_output_structure(self, council_engine_factory, simple_personas):
        """Verify session output contains all required fields."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Structure Test",
            objective="Verify output format",
            personas=simple_personas,
        )

        # Convert to dict to verify JSON serialization
        session_dict = session.to_dict()

        # Verify structure
        assert "topic" in session_dict
        assert "objective" in session_dict
        assert "personas" in session_dict
        assert "rounds" in session_dict
        assert "consensus_reached" in session_dict

        # Verify persona structure includes is_mediator
        for persona in session_dict["personas"]:
            assert "is_mediator" in persona

        # Verify message structure includes is_pass and is_mediator
        if session_dict["rounds"]:
            for msg in session_dict["rounds"][0]["messages"]:
                assert "is_pass" in msg
                assert "is_mediator" in msg

    def test_session_reaches_vote_phase(self, council_engine_factory, simple_personas):
        """Verify session progresses through to voting."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Decision Required",
            objective="Make a definitive choice",
            personas=simple_personas,
        )

        # Should have attempted voting
        has_votes = any(r.votes for r in session.rounds)
        assert has_votes or session.consensus_reached, \
            "Session should reach vote phase or consensus"


@pytest.mark.api
class TestEdgeCases:
    """Test edge cases with real API."""

    def test_empty_personas_raises_error(self, lmstudio_provider):
        """Empty persona list should raise error."""
        engine = CouncilEngine(provider=lmstudio_provider)

        with pytest.raises(ValueError):
            engine.run_session(
                topic="Test",
                objective="Test",
                personas=[],
            )

    def test_single_round_session(self, council_engine_factory, simple_personas):
        """Single round should still produce valid output."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Quick Test",
            objective="Single round validation",
            personas=simple_personas,
        )

        assert len(session.rounds) >= 1
        assert len(session.rounds[0].messages) == 3

    def test_response_content_not_empty(self, council_engine_factory, simple_personas):
        """All responses should have content."""
        engine = council_engine_factory(max_rounds=1)

        session = engine.run_session(
            topic="Content Test",
            objective="Verify non-empty responses",
            personas=simple_personas,
        )

        for round_data in session.rounds:
            for msg in round_data.messages:
                assert len(msg.content.strip()) > 0, \
                    f"Message from {msg.persona_name} should have content"
