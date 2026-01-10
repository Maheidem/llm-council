# LLM Council

A CLI tool for LLM-based council/consensus decision making where multiple AI personas discuss toward defined objectives.

## Features

- **Multi-Persona Discussions**: Create councils with multiple AI personas, each with unique perspectives and expertise
- **Consensus Mechanisms**: Support for unanimous, supermajority, majority, and plurality voting
- **Stalemate Resolution**: Automatic voting when discussions reach an impasse
- **LiteLLM Integration**: Works with any LLM provider supported by LiteLLM, including local models via LM Studio
- **Non-Interactive Mode**: Fully automated for agentic use cases
- **JSON Output**: Programmatic output format for integration with other tools

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd llm-council

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install the package
pip install -e ".[dev]"
```

## Quick Start

### With Local LM Studio

1. Start LM Studio and load a model (e.g., qwen3-coder-30b or nemotron)
2. Enable the local server (usually http://localhost:1234)
3. Run a council session:

```bash
llm-council discuss \
    --topic "API Design" \
    --objective "Choose between REST and GraphQL for our new service" \
    --model "openai/qwen3-coder-30b" \
    --api-base "http://localhost:1234/v1"
```

### With OpenAI

```bash
export OPENAI_API_KEY="your-key-here"
llm-council discuss \
    --topic "Code Review" \
    --objective "Evaluate the proposed architecture changes" \
    --preset openai
```

## CLI Commands

### `discuss` - Run a Council Discussion

```bash
llm-council discuss [OPTIONS]

Options:
  -t, --topic TEXT            Discussion topic (required)
  -o, --objective TEXT        Goal/decision to reach (required)
  -c, --context TEXT          Additional context
  -m, --model TEXT            Model to use (default: openai/qwen3-coder-30b)
  -b, --api-base TEXT         API base URL (default: http://localhost:1234/v1)
  -k, --api-key TEXT          API key if required
  -p, --preset TEXT           Use a preset (lmstudio, openai, openai-mini)
  -n, --personas INTEGER      Number of personas (default: 3)
  --auto-personas             Auto-generate personas based on topic
  --consensus-type TEXT       Type required (unanimous, supermajority, majority, plurality)
  -r, --max-rounds INTEGER    Maximum discussion rounds (default: 5)
  -O, --output TEXT           Output format (text, json)
  -q, --quiet                 Minimal output for automation
```

### `test-connection` - Test LLM Provider Connection

```bash
llm-council test-connection --api-base "http://localhost:1234/v1"
```

### `list-personas` - Show Available Default Personas

```bash
llm-council list-personas
```

### `run-config` - Run from Configuration File

```bash
llm-council run-config config.json
```

Configuration file format:
```json
{
    "topic": "Discussion topic",
    "objective": "Goal to achieve",
    "context": "Optional context",
    "model": "openai/qwen3-coder-30b",
    "api_base": "http://localhost:1234/v1",
    "personas": 3,
    "auto_personas": false,
    "consensus_type": "majority",
    "max_rounds": 5,
    "output": "json"
}
```

## Default Personas

The tool includes 5 default personas designed to provide balanced perspectives:

1. **The Pragmatist** - Focus on achievable solutions with current resources
2. **The Innovator** - Push boundaries and explore novel approaches
3. **The Critic** - Identify weaknesses, risks, and potential failures
4. **The Diplomat** - Find common ground and ensure all viewpoints are heard
5. **The Specialist** - Ensure technical accuracy and adherence to standards

## Consensus Types

- **Unanimous**: All participants must agree
- **Supermajority**: 2/3 of participants must agree
- **Majority**: More than 50% must agree
- **Plurality**: The option with the most votes wins

## Programmatic Usage

```python
from llm_council.providers import create_provider
from llm_council.council import CouncilEngine
from llm_council.personas import PersonaManager
from llm_council.models import ConsensusType

# Create provider
provider = create_provider(
    model="openai/qwen3-coder-30b",
    api_base="http://localhost:1234/v1",
)

# Get personas
manager = PersonaManager()
personas = manager.get_default_personas(3)

# Create engine and run session
engine = CouncilEngine(
    provider=provider,
    consensus_type=ConsensusType.MAJORITY,
    max_rounds=5,
)

session = engine.run_session(
    topic="Architecture Decision",
    objective="Choose the best database for our use case",
    personas=personas,
)

print(f"Consensus reached: {session.consensus_reached}")
print(f"Final position: {session.final_consensus}")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=llm_council
```

## License

MIT
