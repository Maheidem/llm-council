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
from llm_council.discussion import DiscussionState
from llm_council.voting import VoteParser, VotingMachine, StructuredVote


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
            "As mediator, let's begin the discussion.",
            "This is my first perspective.",
            "I agree with the first point.",
        ])
        engine = CouncilEngine(provider=provider)
        personas = DEFAULT_PERSONAS[:3]
        discussion_state = DiscussionState()
        discussion_state.advance_round()

        result = engine._conduct_round(
            round_num=1,
            topic="Test topic",
            objective="Make a decision",
            personas=personas,
            history=[],
            initial_context=None,
            discussion_state=discussion_state,
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

    def test_conduct_vote_majority(self):
        # First response is synthesize_proposal, then votes from non-mediator personas
        # With 3 personas and first marked as mediator, we get 2 votes
        provider = MockProvider(responses=[
            "Proposal: Do X",  # synthesize_proposal
            "[VOTE] AGREE\n[CONFIDENCE] 0.9\n[REASONING] Good idea",
            "[VOTE] DISAGREE\n[CONFIDENCE] 0.7\n[REASONING] Not convinced",
        ])
        engine = CouncilEngine(
            provider=provider,
            consensus_type=ConsensusType.MAJORITY,
        )

        # Mark first persona as mediator (as run_session does)
        personas = DEFAULT_PERSONAS[:3].copy()
        personas[0] = Persona(
            name=personas[0].name,
            role=personas[0].role,
            expertise=personas[0].expertise,
            personality_traits=personas[0].personality_traits,
            perspective=personas[0].perspective,
            is_mediator=True,
        )

        result = engine._conduct_vote(
            topic="Test",
            objective="Decide",
            personas=personas,
            history=[],
        )

        # With majority (>50%), 1 agree vs 1 disagree = 50% = not reached
        assert result["consensus_reached"] is False
        # 2 votes (mediator excluded)
        assert len(result["votes"]) == 2

    def test_run_session_reaches_consensus(self):
        # Each round: 3 personas respond (mediator + 2 others)
        # Then vote: proposal + 2 votes (mediator excluded)
        responses = [
            # Round 1 discussions (3 personas)
            "As mediator, let's discuss option A.",
            "I think we should go with option A.",
            "I agree, option A seems best.",
            # Vote triggered after max rounds or stalemate
            "Proposal: Option A is the best choice",
            "[VOTE] AGREE\n[CONFIDENCE] 0.9\n[REASONING] Makes sense",
            "[VOTE] AGREE\n[CONFIDENCE] 0.8\n[REASONING] I support this",
        ]
        provider = MockProvider(responses=responses)
        engine = CouncilEngine(provider=provider, max_rounds=1)

        session = engine.run_session(
            topic="Choose an option",
            objective="Pick the best option",
            personas=DEFAULT_PERSONAS[:3],
        )

        # Should reach consensus via vote
        assert len(session.rounds) >= 1

    def test_run_session_forces_vote_on_stalemate(self):
        # Simulate discussion then vote
        responses = [
            # Round 1 discussions (3 personas)
            "As mediator, let's start.", "I prefer A", "I prefer B",
            # Round 2 discussions
            "Let's continue.", "Still prefer A", "Still prefer B",
            # Round 3 - same positions trigger stalemate
            "Let's try to agree.", "A is best", "B is best",
            # Vote phase
            "Proposal: Go with majority preference",
            "[VOTE] AGREE\n[CONFIDENCE] 0.8\n[REASONING] Fine",
            "[VOTE] AGREE\n[CONFIDENCE] 0.7\n[REASONING] OK",
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


class TestVoteParser:
    """Tests for deterministic vote parsing."""

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
    """Tests for deterministic vote tallying."""

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
