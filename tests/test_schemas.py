"""Tests for US-02: Schema validation with Pydantic."""

import pytest
from llm_council.schemas import (
    DiscussionRequestSchema,
    PersonaTemplateSchema,
    SessionOutputSchema,
    SchemaValidationError,
    ValidationErrors,
    ValidationErrorCode,
    validate_discussion_request,
    validate_persona_template,
    SchemaValidator,
)


class TestDiscussionRequestSchema:
    """Tests for discussion request validation."""

    def test_valid_minimal_request(self):
        data = {"topic": "Test Topic", "objective": "Test Objective"}
        schema = DiscussionRequestSchema(**data)
        assert schema.topic == "Test Topic"
        assert schema.objective == "Test Objective"
        assert schema.personas == 3  # default
        assert schema.max_rounds == 3  # default
        assert schema.consensus_type == "majority"  # default

    def test_valid_full_request(self):
        data = {
            "topic": "Framework Choice",
            "objective": "Choose between React or Vue",
            "context": "Building a new web app",
            "personas": 5,
            "max_rounds": 4,
            "consensus_type": "supermajority",
        }
        schema = DiscussionRequestSchema(**data)
        assert schema.personas == 5
        assert schema.max_rounds == 4
        assert schema.consensus_type == "supermajority"

    def test_missing_required_field(self):
        data = {"topic": "Test Topic"}  # missing objective
        with pytest.raises(Exception) as exc_info:
            DiscussionRequestSchema(**data)
        assert "objective" in str(exc_info.value).lower()

    def test_empty_topic_fails(self):
        data = {"topic": "", "objective": "Test Objective"}
        with pytest.raises(Exception):
            DiscussionRequestSchema(**data)

    def test_whitespace_topic_fails(self):
        data = {"topic": "   ", "objective": "Test Objective"}
        with pytest.raises(Exception):
            DiscussionRequestSchema(**data)

    def test_invalid_consensus_type(self):
        data = {"topic": "Test", "objective": "Test", "consensus_type": "invalid"}
        with pytest.raises(Exception) as exc_info:
            DiscussionRequestSchema(**data)
        assert "consensus_type" in str(exc_info.value).lower()

    def test_personas_out_of_range_low(self):
        data = {"topic": "Test", "objective": "Test", "personas": 1}
        with pytest.raises(Exception):
            DiscussionRequestSchema(**data)

    def test_personas_out_of_range_high(self):
        data = {"topic": "Test", "objective": "Test", "personas": 20}
        with pytest.raises(Exception):
            DiscussionRequestSchema(**data)

    def test_max_rounds_out_of_range(self):
        data = {"topic": "Test", "objective": "Test", "max_rounds": 0}
        with pytest.raises(Exception):
            DiscussionRequestSchema(**data)


class TestPersonaTemplateSchema:
    """Tests for persona template validation."""

    def test_valid_persona(self):
        data = {
            "name": "Test Persona",
            "role": "Test Role",
            "expertise": ["testing", "validation"],
            "personality_traits": ["analytical", "thorough"],
            "perspective": "Testing perspective",
        }
        schema = PersonaTemplateSchema(**data)
        assert schema.name == "Test Persona"
        assert len(schema.expertise) == 2

    def test_missing_expertise(self):
        data = {
            "name": "Test Persona",
            "role": "Test Role",
            "personality_traits": ["analytical"],
            "perspective": "Testing perspective",
        }
        with pytest.raises(Exception) as exc_info:
            PersonaTemplateSchema(**data)
        assert "expertise" in str(exc_info.value).lower()

    def test_empty_expertise_list(self):
        data = {
            "name": "Test",
            "role": "Test",
            "expertise": [],
            "personality_traits": ["analytical"],
            "perspective": "Test",
        }
        with pytest.raises(Exception):
            PersonaTemplateSchema(**data)


class TestSchemaValidationError:
    """Tests for structured error format."""

    def test_error_to_dict(self):
        error = SchemaValidationError(
            field="discussion_request.topic",
            code=ValidationErrorCode.REQUIRED_FIELD,
            message="Field is required",
            severity="error",
        )
        d = error.to_dict()
        assert d["field"] == "discussion_request.topic"
        assert d["error_code"] == "required_field"
        assert d["message"] == "Field is required"
        assert d["severity"] == "error"

    def test_error_string_format(self):
        error = SchemaValidationError(
            field="topic",
            code=ValidationErrorCode.INVALID_VALUE,
            message="Value is invalid",
        )
        assert "[invalid_value]" in str(error)
        assert "topic" in str(error)


class TestValidationErrors:
    """Tests for validation errors collection."""

    def test_multiple_errors_to_dict(self):
        errors = ValidationErrors([
            SchemaValidationError("field1", ValidationErrorCode.REQUIRED_FIELD, "Required"),
            SchemaValidationError("field2", ValidationErrorCode.INVALID_TYPE, "Wrong type"),
        ])
        d = errors.to_dict()
        assert d["valid"] is False
        assert d["error_count"] == 2
        assert len(d["errors"]) == 2


class TestValidateDiscussionRequest:
    """Tests for validate_discussion_request helper."""

    def test_valid_request(self):
        data = {"topic": "Test", "objective": "Test"}
        result = validate_discussion_request(data)
        assert isinstance(result, DiscussionRequestSchema)

    def test_invalid_request_raises_validation_errors(self):
        data = {"topic": "Test"}  # missing objective
        with pytest.raises(ValidationErrors) as exc_info:
            validate_discussion_request(data)
        errors = exc_info.value.to_dict()
        assert errors["valid"] is False
        assert errors["error_count"] >= 1


class TestSchemaValidator:
    """Tests for fluent schema validator."""

    def test_valid_request_is_valid(self):
        validator = SchemaValidator().validate_request({"topic": "Test", "objective": "Test"})
        assert validator.is_valid
        assert len(validator.errors) == 0
        assert validator.data is not None

    def test_invalid_request_not_valid(self):
        validator = SchemaValidator().validate_request({"topic": "Test"})  # missing objective
        assert not validator.is_valid
        assert len(validator.errors) > 0

    def test_validator_to_dict(self):
        validator = SchemaValidator().validate_request({"topic": "Test", "objective": "Test"})
        d = validator.to_dict()
        assert d["valid"] is True
        assert d["error_count"] == 0

    def test_validator_to_dict_with_errors(self):
        validator = SchemaValidator().validate_request({})
        d = validator.to_dict()
        assert d["valid"] is False
        assert d["error_count"] > 0

    def test_raise_on_errors(self):
        validator = SchemaValidator().validate_request({})
        with pytest.raises(ValidationErrors):
            validator.raise_on_errors()

    def test_no_raise_when_valid(self):
        validator = SchemaValidator().validate_request({"topic": "Test", "objective": "Test"})
        validator.raise_on_errors()  # Should not raise


class TestSessionOutputSchema:
    """Tests for session output validation."""

    def test_valid_session_output(self):
        data = {
            "topic": "Test",
            "objective": "Test",
            "personas": [
                {
                    "name": "Persona1",
                    "role": "Role1",
                    "expertise": ["exp"],
                    "personality_traits": ["trait"],
                    "perspective": "perspective",
                }
            ],
            "rounds": [
                {
                    "round_number": 1,
                    "messages": [
                        {
                            "persona_name": "Persona1",
                            "content": "Test message",
                            "round_number": 1,
                            "message_type": "discussion",
                        }
                    ],
                    "consensus_reached": True,
                    "consensus_position": "Agreed",
                    "votes": [],
                }
            ],
            "final_consensus": "Final agreement",
            "consensus_reached": True,
        }
        schema = SessionOutputSchema(**data)
        assert schema.consensus_reached is True
        assert len(schema.rounds) == 1
