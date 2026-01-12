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

    def test_engine_creation(self, lmstudio_provider):
        """Test engine instantiation."""
        engine = CouncilEngine(
            provider=lmstudio_provider,
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

    def test_format_history(self, lmstudio_provider):
        """Test history formatting - pure logic, no API."""
        engine = CouncilEngine(provider=lmstudio_provider)
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
