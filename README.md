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

### Quick Start with UVX (Recommended)

Run directly without installation:

```bash
uvx llm-council-mcp
```

### Install as Tool

```bash
# Install UV first (if not already installed)
# Linux/Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install llm-council-mcp as a tool
uv tool install llm-council-mcp
```

### Traditional pip Install

```bash
pip install llm-council-mcp
```

## Claude Code MCP Integration

Add llm-council as an MCP server in your Claude Code configuration:

```json
{
  "mcpServers": {
    "llm-council": {
      "command": "uvx",
      "args": ["llm-council-mcp"]
    }
  }
}
```

Or if installed as a tool:

```json
{
  "mcpServers": {
    "llm-council": {
      "command": "llm-council-mcp"
    }
  }
}
```

With environment variables for API keys:

```json
{
  "mcpServers": {
    "llm-council": {
      "command": "uvx",
      "args": ["llm-council-mcp"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `council_discuss` | Run a council discussion with multiple AI personas to reach consensus |
| `config_init` | Initialize configuration with 3-step guided setup wizard |
| `config_get` | Get current configuration values |
| `config_set` | Set configuration values |
| `config_validate` | Validate provider connections |
| `providers_list` | List all configured LLM providers |
| `personas_generate` | Generate custom personas for a specific topic |

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
# Clone the repository
git clone https://github.com/Maheidem/llm-council.git
cd llm-council

# Install dependencies with UV
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=llm_council

# Run the CLI during development
uv run llm-council --help

# Run the MCP server during development
uv run llm-council-mcp
```

Alternative with pip:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
