# LLM Council - Project Diary

## Project Overview
A CLI tool for LLM-based council/consensus decision making where multiple AI personas discuss toward defined objectives.

## Key Decisions

### 2026-01-10 - Project Initialization
- **Decision**: Use Python with LiteLLM as primary provider abstraction
- **Rationale**: LiteLLM provides unified interface to multiple LLM providers including local LM Studio
- **Decision**: Focus on LiteLLM + local LM Studio (qwen3-coder-30b, nemotron) as primary target
- **Decision**: OpenAI support as secondary provider option
- **Decision**: CLI-first, non-interactive design for agentic use

### Architecture Decisions
- **Pattern**: Strategy pattern for LLM providers (LiteLLMProvider abstract class)
- **Pattern**: Council engine orchestrates discussion rounds between personas
- **Consensus**: Configurable consensus mechanisms (unanimous, majority, supermajority, plurality)
- **Stalemate Resolution**: Automatic voting after configurable stalemate threshold
- **Data Models**: Dataclasses for Persona, Message, Vote, RoundResult, CouncilSession

### Implementation Decisions
- **Click**: Used for CLI framework (clean, well-tested)
- **Rich**: Used for terminal output formatting
- **Balanced bracket parsing**: Custom parser for LLM-generated persona definitions
- **No emojis**: Avoid Unicode symbols for Windows compatibility

## Requirements

### Functional
1. Define personas with roles, expertise, and personality traits
2. Auto-generate appropriate personas based on problem analysis
3. Conduct multi-round discussions between personas
4. Reach consensus through structured deliberation
5. Handle stalemates through voting mechanisms
6. Support both OpenAI and LiteLLM (local models)
7. Fully non-interactive CLI mode

### Non-Functional
1. No user interaction required during execution
2. Clear logging and progress indication
3. Configurable timeouts and retry logic
4. JSON output for programmatic consumption

## TODO
- [x] Initialize project structure
- [x] Implement core data models
- [x] Build LiteLLM integration
- [x] Create persona system
- [x] Implement discussion engine
- [x] Build consensus/voting system
- [x] Create CLI interface
- [x] Write tests (47 tests passing)
- [x] Test CLI with LM Studio connection
- [x] Create README documentation

## Achievements

### 2026-01-10
- Project initialized with git repository
- Created pyproject.toml with all dependencies (litellm, openai, pydantic, click, rich)
- Implemented core data models (models.py):
  - Persona, Message, Vote, RoundResult, CouncilSession dataclasses
  - ConsensusType and VoteChoice enums
  - 5 default personas with diverse perspectives
- Implemented LiteLLM provider layer (providers.py):
  - ProviderConfig dataclass
  - LiteLLMProvider class with complete/test_connection methods
  - Factory function and preset configurations
- Created persona management system (personas.py):
  - PersonaManager with default/custom persona support
  - LLM-based persona generation from topics
  - Robust parsing with balanced bracket matching
- Built council engine (council.py):
  - Multi-round discussion orchestration
  - Consensus detection via LLM
  - Automatic voting on stalemate
  - Multiple consensus types supported
- Created full CLI interface (cli.py):
  - discuss, test-connection, list-personas, run-config commands
  - JSON and text output formats
  - Preset support for quick configuration
- Wrote comprehensive test suite:
  - 47 tests covering all modules
  - Mock provider for unit testing
  - CLI tests with click.testing.CliRunner
- All tests passing

## File Structure
```
llm-council/
├── src/
│   └── llm_council/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── council.py
│       ├── models.py
│       ├── personas.py
│       └── providers.py
├── tests/
│   ├── __init__.py
│   ├── test_cli.py
│   ├── test_council.py
│   ├── test_models.py
│   ├── test_personas.py
│   └── test_providers.py
├── .gitignore
├── PROJECT_DIARY.md
├── README.md
└── pyproject.toml
```
