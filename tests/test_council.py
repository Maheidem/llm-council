"""Tests for council engine."""

import pytest
from unittest.mock import MagicMock, patch

from llm_council.models import (
    Persona,
    Message,
    Vote,
    VoteChoice,
    ConsensusType,
    DEFAULT_PERSONAS,
)
from llm_council.council import CouncilEngine


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.responses:
            response = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            return response
        return "This is a mock response."

    def test_connection(self) -> bool:
        return True


class TestCouncilEngine:
    """Tests for CouncilEngine."""

    def test_engine_creation(self):
        provider = MockProvider()
        engine = CouncilEngine(
            provider=provider,
            consensus_type=ConsensusType.MAJORITY,
            max_rounds=5,
        )
        assert engine.consensus_type == ConsensusType.MAJORITY
        assert engine.max_rounds == 5

    def test_conduct_round(self):
        provider = MockProvider(responses=[
            "This is my first perspective.",
            "I agree with the first point.",
            "Let me add my thoughts.",
        ])
        engine = CouncilEngine(provider=provider)
        personas = DEFAULT_PERSONAS[:3]

        result = engine._conduct_round(
            round_num=1,
            topic="Test topic",
            objective="Make a decision",
            personas=personas,
            history=[],
        )

        assert result.round_number == 1
        assert len(result.messages) == 3
        for msg in result.messages:
            assert msg.round_number == 1

    def test_format_history(self):
        engine = CouncilEngine(provider=MockProvider())
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

    def test_check_consensus_parses_json(self):
        provider = MockProvider(responses=[
            '{"reached": true, "position": "We should proceed", "summary": "Agreement reached"}'
        ])
        engine = CouncilEngine(provider=provider)

        result = engine._check_consensus(
            topic="Test",
            objective="Decide",
            history=[Message("Expert", "I agree", 1)],
            personas=DEFAULT_PERSONAS[:1],
        )

        assert result["reached"] is True
        assert result["position"] == "We should proceed"

    def test_check_consensus_handles_invalid_json(self):
        provider = MockProvider(responses=["This is not JSON"])
        engine = CouncilEngine(provider=provider)

        result = engine._check_consensus(
            topic="Test",
            objective="Decide",
            history=[Message("Expert", "Discussion", 1)],
            personas=DEFAULT_PERSONAS[:1],
        )

        assert result["reached"] is False

    def test_get_vote_agree(self):
        provider = MockProvider(responses=[
            "VOTE: AGREE\nREASON: This makes complete sense."
        ])
        engine = CouncilEngine(provider=provider)
        persona = DEFAULT_PERSONAS[0]

        vote = engine._get_vote(
            persona=persona,
            topic="Test",
            objective="Decide",
            proposal="We should do X",
            history_text="Previous discussion...",
        )

        assert vote.choice == VoteChoice.AGREE
        assert vote.persona_name == persona.name

    def test_get_vote_disagree(self):
        provider = MockProvider(responses=[
            "VOTE: DISAGREE\nREASON: I have concerns about this approach."
        ])
        engine = CouncilEngine(provider=provider)
        persona = DEFAULT_PERSONAS[0]

        vote = engine._get_vote(
            persona=persona,
            topic="Test",
            objective="Decide",
            proposal="We should do X",
            history_text="Discussion...",
        )

        assert vote.choice == VoteChoice.DISAGREE

    def test_conduct_vote_majority(self):
        # 2 agree, 1 disagree = majority reached
        provider = MockProvider(responses=[
            "Proposal: Do X",  # synthesize_proposal
            "VOTE: AGREE\nREASON: Good idea",
            "VOTE: AGREE\nREASON: Sounds reasonable",
            "VOTE: DISAGREE\nREASON: Not convinced",
        ])
        engine = CouncilEngine(
            provider=provider,
            consensus_type=ConsensusType.MAJORITY,
        )

        result = engine._conduct_vote(
            topic="Test",
            objective="Decide",
            personas=DEFAULT_PERSONAS[:3],
            history=[],
        )

        assert result["consensus_reached"] is True
        assert len(result["votes"]) == 3
        assert result["agree_count"] == 2

    def test_run_session_reaches_consensus(self):
        responses = [
            # Round 1 discussions (3 personas)
            "I think we should go with option A.",
            "I agree, option A seems best.",
            "Option A has my support too.",
            # Consensus check
            '{"reached": true, "position": "Option A is the best choice", "summary": "All agree"}',
        ]
        provider = MockProvider(responses=responses)
        engine = CouncilEngine(provider=provider, max_rounds=3)

        session = engine.run_session(
            topic="Choose an option",
            objective="Pick the best option",
            personas=DEFAULT_PERSONAS[:3],
        )

        assert session.consensus_reached is True
        assert session.final_consensus is not None
        assert len(session.rounds) >= 1

    def test_run_session_forces_vote_on_stalemate(self):
        # Simulate stalemate then vote
        responses = [
            # Round 1 discussions
            "I prefer A", "I prefer B", "I prefer C",
            # Check 1
            '{"reached": false, "summary": "No agreement"}',
            # Round 2 discussions
            "Still prefer A", "Still prefer B", "Still prefer C",
            # Check 2 - same summary triggers stalemate counter
            '{"reached": false, "summary": "No agreement"}',
            # Round 3 discussions
            "A is best", "B is best", "C is best",
            # Check 3 - same again, triggers vote
            '{"reached": false, "summary": "No agreement"}',
            # Vote phase
            "Proposal: Go with majority preference",
            "VOTE: AGREE\nREASON: Fine",
            "VOTE: AGREE\nREASON: OK",
            "VOTE: DISAGREE\nREASON: No",
        ]
        provider = MockProvider(responses=responses)
        engine = CouncilEngine(
            provider=provider,
            max_rounds=5,
            stalemate_threshold=2,
        )

        session = engine.run_session(
            topic="Debate",
            objective="Reach agreement",
            personas=DEFAULT_PERSONAS[:3],
        )

        # Should have votes in at least one round
        has_votes = any(r.votes for r in session.rounds)
        assert has_votes or session.consensus_reached
