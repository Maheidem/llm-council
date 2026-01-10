"""Tests for data models."""

import pytest

from llm_council.models import (
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
