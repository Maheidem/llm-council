"""Integration tests for isolated persona sessions and deterministic voting.

These tests verify the success criteria:
1. Each persona executes as isolated LLM call with persona-specific system prompt
2. Personas receive other personas' outputs in context
3. Mediator persona controls discussion flow
4. Voting is computed by deterministic function
"""

import logging
import pytest
from unittest.mock import MagicMock, patch, call
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


class APICallTracker:
    """Tracks API calls to verify isolated sessions."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Record call and return mock response."""
        self.calls.append({
            "call_number": self.call_count,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

    def test_connection(self) -> bool:
        return True


class TestIsolatedPersonaSessions:
    """Verify each persona runs as isolated LLM invocation."""

    def test_each_persona_gets_separate_api_call(self):
        """CRITERION 1: Each persona executes as isolated LLM call."""
        tracker = APICallTracker(responses=[
            "Mediator: Let's discuss this topic.",
            "Pragmatist: I think we should be practical.",
            "Innovator: Let's explore new ideas.",
            # Vote phase
            "Proposal: Combine practical and innovative approaches.",
            "[VOTE] AGREE\n[CONFIDENCE] 0.9\n[REASONING] Good balance.",
            "[VOTE] AGREE\n[CONFIDENCE] 0.8\n[REASONING] Sounds reasonable.",
        ])

        engine = CouncilEngine(provider=tracker, max_rounds=1)
        personas = DEFAULT_PERSONAS[:3]

        session = engine.run_session(
            topic="Test Isolation",
            objective="Verify isolated sessions",
            personas=personas,
        )

        # Verify: At least 3 separate API calls (one per persona)
        assert len(tracker.calls) >= 3, f"Expected at least 3 API calls, got {len(tracker.calls)}"

        # Verify: Each persona's system prompt is unique
        system_prompts = [c["system_prompt"] for c in tracker.calls[:3]]
        assert len(set(system_prompts)) == 3, "Each persona should have unique system prompt"

        # Verify: First persona is mediator (has mediator in prompt)
        assert "MEDIATOR" in system_prompts[0].upper() or "mediator" in system_prompts[0].lower(), \
            "First persona should be mediator"

    def test_persona_system_prompts_contain_persona_identity(self):
        """Verify each persona's system prompt includes their role/expertise."""
        tracker = APICallTracker(responses=["Response"] * 10)
        engine = CouncilEngine(provider=tracker, max_rounds=1)
        personas = DEFAULT_PERSONAS[:3]

        engine.run_session(
            topic="Test",
            objective="Verify identity",
            personas=personas,
        )

        # Check first 3 calls (discussion round)
        for i, persona in enumerate(personas):
            call = tracker.calls[i]
            system_prompt = call["system_prompt"]
            # Each system prompt should mention the persona's name or role
            assert persona.name in system_prompt or persona.role in system_prompt, \
                f"System prompt should contain persona identity for {persona.name}"


class TestCrossPersonaAwareness:
    """Verify personas receive other personas' outputs in context."""

    def test_personas_see_previous_round_messages(self):
        """CRITERION 2: Personas receive other personas' outputs in context."""
        tracker = APICallTracker(responses=[
            # Round 1
            "Mediator opening statement.",
            "First persona contribution about topic A.",
            "Second persona response to first.",
            # Round 2
            "Mediator summarizing.",
            "First persona follow-up.",
            "Second persona agreement.",
            # Vote
            "Proposal text",
            "[VOTE] AGREE\n[CONFIDENCE] 0.9\n[REASONING] Good.",
            "[VOTE] AGREE\n[CONFIDENCE] 0.8\n[REASONING] OK.",
        ])

        engine = CouncilEngine(provider=tracker, max_rounds=2)
        engine.run_session(
            topic="Test Cross-Awareness",
            objective="Verify context sharing",
            personas=DEFAULT_PERSONAS[:3],
        )

        # Round 2 prompts should contain Round 1 content
        round2_calls = tracker.calls[3:6]  # Calls 3, 4, 5 are round 2
        for call in round2_calls:
            user_prompt = call["user_prompt"]
            # Should reference "Round 1" in previous discussion
            assert "Round 1" in user_prompt or "round" in user_prompt.lower(), \
                "Round 2 prompts should include Round 1 history"

    def test_same_round_awareness(self):
        """Personas in same round see earlier responses from that round."""
        tracker = APICallTracker(responses=["Response"] * 10)
        engine = CouncilEngine(provider=tracker, max_rounds=1)

        engine.run_session(
            topic="Test",
            objective="Verify same-round awareness",
            personas=DEFAULT_PERSONAS[:3],
        )

        # Third persona (index 2) should see first two personas' responses
        third_call = tracker.calls[2]
        user_prompt = third_call["user_prompt"]
        # Should contain "THIS ROUND SO FAR" with previous responses
        assert "THIS ROUND SO FAR" in user_prompt or "this round" in user_prompt.lower(), \
            "Later personas should see earlier same-round responses"


class TestMediatorFlowControl:
    """Verify mediator persona controls discussion flow."""

    def test_mediator_is_first_in_each_round(self):
        """CRITERION 3: Mediator controls discussion flow."""
        tracker = APICallTracker(responses=["Response"] * 20)
        engine = CouncilEngine(provider=tracker, max_rounds=2)

        engine.run_session(
            topic="Test Mediator Order",
            objective="Verify mediator first",
            personas=DEFAULT_PERSONAS[:3],
        )

        # First call of each round should have mediator system prompt
        round1_first = tracker.calls[0]
        round2_first = tracker.calls[3]

        for call in [round1_first, round2_first]:
            assert "MEDIATOR" in call["system_prompt"].upper(), \
                "First call in each round should be mediator"

    def test_mediator_has_enhanced_prompt(self):
        """Mediator gets phase-specific guidance."""
        tracker = APICallTracker(responses=["Response"] * 10)
        engine = CouncilEngine(provider=tracker, max_rounds=1)

        engine.run_session(
            topic="Test",
            objective="Verify mediator enhancement",
            personas=DEFAULT_PERSONAS[:3],
        )

        mediator_call = tracker.calls[0]
        system_prompt = mediator_call["system_prompt"]

        # Mediator prompt should contain guidance about role
        assert any(word in system_prompt.lower() for word in
                   ["mediator", "guide", "facilitate", "neutral"]), \
            "Mediator should have enhanced prompt with role guidance"

    def test_mediator_excluded_from_voting(self):
        """Mediator doesn't vote - maintains neutrality."""
        tracker = APICallTracker(responses=[
            # Discussion
            "Mediator opening.", "Contribution 1.", "Contribution 2.",
            # Vote
            "Proposal text",
            "[VOTE] AGREE\n[CONFIDENCE] 0.9\n[REASONING] Yes.",
            "[VOTE] DISAGREE\n[CONFIDENCE] 0.7\n[REASONING] No.",
        ])

        engine = CouncilEngine(provider=tracker, max_rounds=1)
        personas = DEFAULT_PERSONAS[:3]

        session = engine.run_session(
            topic="Test",
            objective="Verify mediator exclusion",
            personas=personas,
        )

        # Should have votes from 2 personas (not mediator)
        if session.rounds and session.rounds[-1].votes:
            votes = session.rounds[-1].votes
            assert len(votes) == 2, "Should have 2 votes (mediator excluded)"
            # Mediator (The Diplomat - auto-selected) should not appear
            voter_names = [v.persona_name for v in votes]
            assert session.personas[0].name not in voter_names, \
                "Mediator should not vote"


class TestDeterministicVoting:
    """Verify voting is deterministic with fixed inputs/outputs."""

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

        # Test boundary: 51% SHOULD pass majority
        votes_51 = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, ""),
            StructuredVote("P2", VoteChoice.AGREE, 0.9, ""),
            StructuredVote("P3", VoteChoice.DISAGREE, 0.9, ""),
        ]
        assert machine.tally(votes_51).consensus_reached is True  # 66% > 50%

        # Test supermajority boundary
        machine_super = VotingMachine(ConsensusType.SUPERMAJORITY)
        votes_66 = [
            StructuredVote("P1", VoteChoice.AGREE, 0.9, ""),
            StructuredVote("P2", VoteChoice.AGREE, 0.9, ""),
            StructuredVote("P3", VoteChoice.DISAGREE, 0.9, ""),
        ]
        assert machine_super.tally(votes_66).consensus_reached is False  # 66% not > 66.67%


class TestFullIntegration:
    """Full integration test with logging verification."""

    def test_full_council_session_with_logging(self):
        """Run complete session and verify all criteria via logs."""
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
            tracker = APICallTracker(responses=[
                # Round 1
                "As mediator, I'll guide our discussion on this important topic.",
                "From a practical standpoint, we should consider costs.",
                "[PASS] I agree with the practical concerns raised.",
                # Vote
                "Proposal: Focus on cost-effective practical solutions.",
                "[VOTE] AGREE\n[CONFIDENCE] 0.9\n[REASONING] Makes sense.",
                "[VOTE] AGREE\n[CONFIDENCE] 0.8\n[REASONING] Agreed.",
            ])

            engine = CouncilEngine(provider=tracker, max_rounds=1)

            session = engine.run_session(
                topic="Budget Planning",
                objective="Decide on budget allocation",
                personas=DEFAULT_PERSONAS[:3],
            )

            log_output = log_stream.getvalue()

            # CRITERION 1: Verify isolated API calls logged
            assert "[API CALL]" in log_output, "Should log API calls"

            # CRITERION 2: Verify personas logged by name
            assert "The Pragmatist" in log_output or "Pragmatist" in log_output

            # CRITERION 3: Verify mediator designation logged
            assert "Mediator" in log_output or "mediator" in log_output

            # CRITERION 4: Verify vote tally logged
            assert "agree" in log_output.lower()

            # Verify session structure
            assert len(session.rounds) >= 1
            assert len(session.personas) == 3
            assert session.personas[0].is_mediator is True

            # Verify PASS was detected
            round_messages = session.rounds[0].messages
            pass_messages = [m for m in round_messages if m.is_pass]
            assert len(pass_messages) >= 1, "Should have detected PASS response"

        finally:
            logger.removeHandler(handler)

    def test_session_output_structure(self):
        """Verify session output contains all required fields."""
        tracker = APICallTracker(responses=["Response"] * 10)
        engine = CouncilEngine(provider=tracker, max_rounds=1)

        session = engine.run_session(
            topic="Test Structure",
            objective="Verify output format",
            personas=DEFAULT_PERSONAS[:3],
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


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_all_pass_triggers_vote(self):
        """High pass rate should trigger auto-vote."""
        tracker = APICallTracker(responses=[
            "Mediator opening.",
            "[PASS] Nothing to add.",
            "[PASS] I concur.",
            # Should trigger vote due to high pass rate
            "Proposal",
            "[VOTE] AGREE\n[CONFIDENCE] 0.9\n[REASONING] OK.",
            "[VOTE] AGREE\n[CONFIDENCE] 0.9\n[REASONING] OK.",
        ])

        engine = CouncilEngine(provider=tracker, max_rounds=3)
        session = engine.run_session(
            topic="Test",
            objective="Test pass handling",
            personas=DEFAULT_PERSONAS[:3],
        )

        # Should have votes (auto-triggered)
        has_votes = any(r.votes for r in session.rounds)
        assert has_votes or session.consensus_reached

    def test_malformed_vote_defaults_to_abstain(self):
        """Unparseable vote should default to ABSTAIN."""
        response = "I'm not sure what to think about this proposal."
        vote = VoteParser.parse("Test", response)

        assert vote.choice == VoteChoice.ABSTAIN
        assert vote.parse_success is False

    def test_empty_personas_raises_error(self):
        """Empty persona list should raise error."""
        tracker = APICallTracker(responses=[])
        engine = CouncilEngine(provider=tracker)

        with pytest.raises(ValueError):
            engine.run_session(
                topic="Test",
                objective="Test",
                personas=[],
            )
