"""US-02: Schema validation with Pydantic for input/output validation.

Provides schema validation with clear error messages including field, code, and message.
"""

from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class ValidationErrorCode(str, Enum):
    """Error codes for schema validation."""
    REQUIRED_FIELD = "required_field"
    INVALID_TYPE = "invalid_type"
    INVALID_VALUE = "invalid_value"
    OUT_OF_RANGE = "out_of_range"
    INVALID_FORMAT = "invalid_format"
    CONSTRAINT_VIOLATION = "constraint_violation"


class SchemaValidationError(Exception):
    """Exception with structured error information."""

    def __init__(self, field: str, code: ValidationErrorCode, message: str, severity: str = "error"):
        self.field = field
        self.code = code
        self.message = message
        self.severity = severity
        super().__init__(f"[{code.value}] {field}: {message}")

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "field": self.field,
            "error_code": self.code.value,
            "message": self.message,
            "severity": self.severity,
        }


class ValidationErrors(Exception):
    """Collection of validation errors."""

    def __init__(self, errors: list[SchemaValidationError]):
        self.errors = errors
        messages = "; ".join(str(e) for e in errors)
        super().__init__(f"Validation failed: {messages}")

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "valid": False,
            "error_count": len(self.errors),
            "errors": [e.to_dict() for e in self.errors],
        }


# Input Schemas

class DiscussionRequestSchema(BaseModel):
    """Schema for council discussion requests."""
    model_config = ConfigDict(strict=True)

    topic: str = Field(..., min_length=1, max_length=500, description="The topic to discuss")
    objective: str = Field(..., min_length=1, max_length=1000, description="The goal or decision to reach")
    context: Optional[str] = Field(None, max_length=5000, description="Additional context")
    personas: int = Field(default=3, ge=2, le=10, description="Number of personas (2-10)")
    max_rounds: int = Field(default=3, ge=1, le=10, description="Maximum discussion rounds (1-10)")
    consensus_type: str = Field(default="majority", description="Type of consensus required")

    @field_validator("consensus_type")
    @classmethod
    def validate_consensus_type(cls, v: str) -> str:
        valid_types = ["unanimous", "supermajority", "majority", "plurality"]
        if v.lower() not in valid_types:
            raise ValueError(f"consensus_type must be one of: {', '.join(valid_types)}")
        return v.lower()

    @field_validator("topic", "objective")
    @classmethod
    def validate_not_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty or whitespace only")
        return v.strip()


class PersonaTemplateSchema(BaseModel):
    """Schema for custom persona templates."""
    model_config = ConfigDict(strict=True)

    name: str = Field(..., min_length=1, max_length=100, description="Persona name")
    role: str = Field(..., min_length=1, max_length=200, description="Role description")
    expertise: list[str] = Field(..., min_length=1, max_length=10, description="Areas of expertise")
    personality_traits: list[str] = Field(..., min_length=1, max_length=10, description="Personality traits")
    perspective: str = Field(..., min_length=1, max_length=500, description="Perspective/viewpoint")

    @field_validator("expertise", "personality_traits")
    @classmethod
    def validate_list_items(cls, v: list[str], info) -> list[str]:
        if not v:
            raise ValueError(f"{info.field_name} must have at least one item")
        cleaned = [item.strip() for item in v if item.strip()]
        if not cleaned:
            raise ValueError(f"{info.field_name} items cannot be empty")
        return cleaned


# Output Schemas

class PersonaOutputSchema(BaseModel):
    """Schema for persona output."""
    name: str
    role: str
    expertise: list[str]
    personality_traits: list[str]
    perspective: str


class MessageOutputSchema(BaseModel):
    """Schema for message output."""
    persona_name: str
    content: str
    round_number: int
    message_type: str = "discussion"


class VoteOutputSchema(BaseModel):
    """Schema for vote output."""
    persona_name: str
    choice: str
    reasoning: str


class RoundOutputSchema(BaseModel):
    """Schema for round output."""
    round_number: int
    messages: list[MessageOutputSchema]
    consensus_reached: bool
    consensus_position: Optional[str] = None
    votes: list[VoteOutputSchema] = Field(default_factory=list)


class SessionOutputSchema(BaseModel):
    """Schema for session output."""
    topic: str
    objective: str
    personas: list[PersonaOutputSchema]
    rounds: list[RoundOutputSchema]
    final_consensus: Optional[str] = None
    consensus_reached: bool


# Validation helpers

def validate_discussion_request(data: dict) -> DiscussionRequestSchema:
    """Validate a discussion request and return schema or raise ValidationErrors."""
    try:
        return DiscussionRequestSchema(**data)
    except Exception as e:
        errors = _parse_pydantic_errors(e, "discussion_request")
        raise ValidationErrors(errors)


def validate_persona_template(data: dict) -> PersonaTemplateSchema:
    """Validate a persona template and return schema or raise ValidationErrors."""
    try:
        return PersonaTemplateSchema(**data)
    except Exception as e:
        errors = _parse_pydantic_errors(e, "persona_template")
        raise ValidationErrors(errors)


def validate_session_output(data: dict) -> SessionOutputSchema:
    """Validate a session output and return schema or raise ValidationErrors."""
    try:
        return SessionOutputSchema(**data)
    except Exception as e:
        errors = _parse_pydantic_errors(e, "session_output")
        raise ValidationErrors(errors)


def _parse_pydantic_errors(exc: Exception, schema_name: str) -> list[SchemaValidationError]:
    """Parse Pydantic validation errors into structured errors."""
    errors = []

    if hasattr(exc, "errors"):
        # Pydantic ValidationError
        for err in exc.errors():
            field = ".".join(str(loc) for loc in err.get("loc", ["unknown"]))
            error_type = err.get("type", "unknown")
            message = err.get("msg", str(err))

            # Map Pydantic error types to our codes
            code = _map_error_type(error_type)

            errors.append(SchemaValidationError(
                field=f"{schema_name}.{field}",
                code=code,
                message=message,
            ))
    else:
        # Generic exception
        errors.append(SchemaValidationError(
            field=schema_name,
            code=ValidationErrorCode.INVALID_VALUE,
            message=str(exc),
        ))

    return errors


def _map_error_type(pydantic_type: str) -> ValidationErrorCode:
    """Map Pydantic error types to ValidationErrorCode."""
    mapping = {
        "missing": ValidationErrorCode.REQUIRED_FIELD,
        "string_type": ValidationErrorCode.INVALID_TYPE,
        "int_type": ValidationErrorCode.INVALID_TYPE,
        "list_type": ValidationErrorCode.INVALID_TYPE,
        "string_too_short": ValidationErrorCode.OUT_OF_RANGE,
        "string_too_long": ValidationErrorCode.OUT_OF_RANGE,
        "greater_than_equal": ValidationErrorCode.OUT_OF_RANGE,
        "less_than_equal": ValidationErrorCode.OUT_OF_RANGE,
        "value_error": ValidationErrorCode.INVALID_VALUE,
        "enum": ValidationErrorCode.INVALID_VALUE,
    }
    return mapping.get(pydantic_type, ValidationErrorCode.CONSTRAINT_VIOLATION)


class SchemaValidator:
    """Fluent schema validator for validation pipelines."""

    def __init__(self):
        self._errors: list[SchemaValidationError] = []
        self._data: Optional[Any] = None

    def validate_request(self, data: dict) -> "SchemaValidator":
        """Validate discussion request data."""
        try:
            self._data = validate_discussion_request(data)
        except ValidationErrors as e:
            self._errors.extend(e.errors)
        return self

    def validate_persona(self, data: dict) -> "SchemaValidator":
        """Validate persona template data."""
        try:
            self._data = validate_persona_template(data)
        except ValidationErrors as e:
            self._errors.extend(e.errors)
        return self

    def validate_output(self, data: dict) -> "SchemaValidator":
        """Validate session output data."""
        try:
            self._data = validate_session_output(data)
        except ValidationErrors as e:
            self._errors.extend(e.errors)
        return self

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return len(self._errors) == 0

    @property
    def errors(self) -> list[SchemaValidationError]:
        """Get validation errors."""
        return self._errors

    @property
    def data(self) -> Optional[Any]:
        """Get validated data."""
        return self._data

    def raise_on_errors(self) -> None:
        """Raise ValidationErrors if any errors occurred."""
        if self._errors:
            raise ValidationErrors(self._errors)

    def to_dict(self) -> dict:
        """Get validation result as dictionary."""
        if self._errors:
            return {
                "valid": False,
                "error_count": len(self._errors),
                "errors": [e.to_dict() for e in self._errors],
            }
        return {
            "valid": True,
            "error_count": 0,
            "errors": [],
        }
