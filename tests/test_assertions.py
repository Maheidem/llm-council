"""Tests for US-01: Test Assertion API."""

import pytest
from llm_council.models import CouncilSession, RoundResult, Message, Persona
from llm_council.assertions import assert_council, CouncilAssertions, ValidationResult, AssertionReport


@pytest.fixture
def sample_personas():
    """Create sample personas."""
    return [
        Persona(
            name="TestPersona1",
            role="Tester",
            expertise=["testing"],
            personality_traits=["analytical"],
            perspective="Testing perspective"
        ),
        Persona(
            name="TestPersona2",
            role="Reviewer",
            expertise=["review"],
            personality_traits=["critical"],
            perspective="Review perspective"
        ),
    ]


@pytest.fixture
def sample_session_consensus(sample_personas):
    """Create a session that reached consensus."""
    return CouncilSession(
        topic="Framework Choice",
        objective="Choose between React or Vue",
        personas=sample_personas,
        rounds=[
            RoundResult(
                round_number=1,
                messages=[
                    Message(persona_name="TestPersona1", content="I recommend React for its ecosystem.", round_number=1),
                    Message(persona_name="TestPersona2", content="Vue is simpler but React scales better.", round_number=1),
                ],
                consensus_reached=False,
            ),
            RoundResult(
                round_number=2,
                messages=[
                    Message(persona_name="TestPersona1", content="Agreed on React.", round_number=2),
                    Message(persona_name="TestPersona2", content="Yes, React it is.", round_number=2),
                ],
                consensus_reached=True,
                consensus_position="React",
            ),
        ],
        consensus_reached=True,
        final_consensus="The council recommends React for its mature ecosystem and scalability.",
    )


@pytest.fixture
def sample_session_no_consensus(sample_personas):
    """Create a session that did NOT reach consensus."""
    return CouncilSession(
        topic="Database Choice",
        objective="Choose between PostgreSQL or MongoDB",
        personas=sample_personas,
        rounds=[
            RoundResult(
                round_number=1,
                messages=[
                    Message(persona_name="TestPersona1", content="PostgreSQL is better.", round_number=1),
                    Message(persona_name="TestPersona2", content="MongoDB is better.", round_number=1),
                ],
                consensus_reached=False,
            ),
        ],
        consensus_reached=False,
        final_consensus=None,
    )


class TestConsensusValidator:
    """Tests for consensus validation constraints."""

    def test_assert_consensus_reached_pass(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_consensus_reached().report()
        assert result.passed
        assert result.results[0].constraint_name == "consensus_reached"

    def test_assert_consensus_reached_fail(self, sample_session_no_consensus):
        result = assert_council(sample_session_no_consensus).assert_consensus_reached().report()
        assert not result.passed
        assert "NOT reached" in result.results[0].message

    def test_assert_consensus_not_reached_pass(self, sample_session_no_consensus):
        result = assert_council(sample_session_no_consensus).assert_consensus_not_reached().report()
        assert result.passed

    def test_assert_consensus_not_reached_fail(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_consensus_not_reached().report()
        assert not result.passed


class TestRoundCountValidator:
    """Tests for round count validation constraints."""

    def test_assert_max_rounds_pass(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_max_rounds(3).report()
        assert result.passed
        assert result.results[0].actual_value == 2

    def test_assert_max_rounds_fail(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_max_rounds(1).report()
        assert not result.passed
        assert "exceeded" in result.results[0].message

    def test_assert_min_rounds_pass(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_min_rounds(2).report()
        assert result.passed

    def test_assert_min_rounds_fail(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_min_rounds(5).report()
        assert not result.passed


class TestContentValidator:
    """Tests for content validation constraints."""

    def test_assert_consensus_contains_pass(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_consensus_contains("React").report()
        assert result.passed

    def test_assert_consensus_contains_case_insensitive(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_consensus_contains("react").report()
        assert result.passed

    def test_assert_consensus_contains_fail(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_consensus_contains("Angular").report()
        assert not result.passed

    def test_assert_consensus_matches_pass(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_consensus_matches(r"React.*ecosystem").report()
        assert result.passed

    def test_assert_consensus_matches_fail(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_consensus_matches(r"^Vue").report()
        assert not result.passed


class TestDecisionBoundaryValidator:
    """Tests for decision boundary validation constraints."""

    def test_assert_decision_in_options_pass(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_decision_in_options(["React", "Vue", "Angular"]).report()
        assert result.passed
        assert "React" in result.results[0].actual_value

    def test_assert_decision_in_options_fail(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_decision_in_options(["Svelte", "Solid"]).report()
        assert not result.passed


class TestPersonaValidator:
    """Tests for persona participation validation."""

    def test_assert_persona_count_pass(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_persona_count(2).report()
        assert result.passed

    def test_assert_persona_count_fail(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_persona_count(5).report()
        assert not result.passed

    def test_assert_min_personas_pass(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_min_personas(1).report()
        assert result.passed

    def test_assert_min_personas_fail(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_min_personas(10).report()
        assert not result.passed


class TestCustomValidator:
    """Tests for custom validation constraints."""

    def test_custom_validator_pass(self, sample_session_consensus):
        result = assert_council(sample_session_consensus).assert_custom(
            name="topic_check",
            validator=lambda s: "Framework" in s.topic,
            message_pass="Topic is about frameworks",
            message_fail="Topic is NOT about frameworks",
        ).report()
        assert result.passed

    def test_custom_validator_fail(self, sample_session_no_consensus):
        result = assert_council(sample_session_no_consensus).assert_custom(
            name="topic_check",
            validator=lambda s: "Framework" in s.topic,
        ).report()
        assert not result.passed


class TestChaining:
    """Tests for assertion chaining."""

    def test_chained_assertions_all_pass(self, sample_session_consensus):
        report = (
            assert_council(sample_session_consensus)
            .assert_consensus_reached()
            .assert_max_rounds(5)
            .assert_consensus_contains("React")
            .report()
        )
        assert report.passed
        assert len(report.results) == 3
        assert report.failed_count == 0

    def test_chained_assertions_some_fail(self, sample_session_consensus):
        report = (
            assert_council(sample_session_consensus)
            .assert_consensus_reached()
            .assert_max_rounds(1)  # This should fail
            .assert_consensus_contains("Angular")  # This should fail
            .report()
        )
        assert not report.passed
        assert report.failed_count == 2


class TestAssertionReport:
    """Tests for assertion report functionality."""

    def test_report_to_dict(self, sample_session_consensus):
        report = (
            assert_council(sample_session_consensus)
            .assert_consensus_reached()
            .assert_max_rounds(3)
            .report()
        )
        d = report.to_dict()
        assert d["passed"] is True
        assert d["total"] == 2
        assert d["failed"] == 0
        assert len(d["results"]) == 2

    def test_validate_method(self, sample_session_consensus):
        assertions = assert_council(sample_session_consensus).assert_consensus_reached()
        assert assertions.validate() is True

    def test_raise_on_failure(self, sample_session_consensus):
        assertions = (
            assert_council(sample_session_consensus)
            .assert_consensus_reached()
            .assert_max_rounds(1)  # Will fail
        )
        with pytest.raises(AssertionError) as exc_info:
            assertions.raise_on_failure()
        assert "max_rounds" in str(exc_info.value)
