# Plan Agent Handoff - MCP Server Config Integration & UV/UVX Packaging

**Agent**: plan
**Status**: SUCCESS
**Timestamp**: 2026-01-10
**Task**: Analyze MCP server and plan config system integration with UV/UVX packaging

---

## Executive Summary

Comprehensive analysis of the `llm-council` MCP server and planning for integration with the new configuration system. This plan covers:
1. Current MCP server structure analysis
2. New MCP tools for config/onboarding
3. UV/UVX packaging for easy installation
4. Docker containerization option
5. Migration path from current setup

---

## Task Context

**Original Request**:
Analyze the existing MCP server and plan how to integrate the new configuration system with interactive onboarding.

**Objective**:
Create a detailed technical plan for integrating config management tools into the MCP server, supporting interactive onboarding, and packaging with UV/UVX for easy distribution.

**Scope**:
- MCP server enhancement
- Config management via MCP tools
- Interactive onboarding workflow
- UV/UVX packaging
- Docker alternative
- Cross-platform support

---

## Discovery & Analysis

### Current Codebase Structure

```
src/llm_council/
    __init__.py          # Package exports
    __main__.py          # Module entry point (runs CLI)
    mcp_server.py        # Current MCP server implementation
    cli.py               # Click-based CLI with config commands
    config.py            # ConfigManager, ConfigSchema, ProviderSettings
    providers.py         # LiteLLMProvider, ProviderRegistry
    personas.py          # PersonaManager for persona handling
    council.py           # CouncilEngine for running discussions
    models.py            # Data models (Persona, ConsensusType, etc.)
    + other modules (assertions, contracts, templates, persistence, metrics, testing, schemas)
```

### Current MCP Server Analysis (`mcp_server.py`)

**Current Implementation**:
- Uses `mcp.server` SDK (low-level implementation)
- Single tool: `council_discuss`
- Hardcoded defaults: `openai/qwen/qwen3-coder-30b`, `http://localhost:1234/v1`
- No config integration - parameters passed inline
- No interactive capabilities
- stdio transport via `mcp.server.stdio.stdio_server()`

**Current Tool Schema**:
```python
Tool(
    name="council_discuss",
    inputSchema={
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "objective": {"type": "string"},
            "context": {"type": "string"},
            "personas": {"type": "integer", "default": 3},
            "max_rounds": {"type": "integer", "default": 3},
            "consensus_type": {"type": "string", "enum": [...]},
            "model": {"type": "string"},
            "api_base": {"type": "string"},
            "api_key": {"type": "string"}
        },
        "required": ["topic", "objective"]
    }
)
```

### Existing Config System (`config.py`)

**Already Implemented**:
- `ConfigSchema` - Pydantic model for config validation
- `ConfigManager` - Load, merge, resolve, save configs
- `ProviderSettings` - Per-provider configuration
- `ResolvedConfig` - Fully merged configuration
- Config file locations:
  - Windows: `%APPDATA%/llm-council/config.yaml`
  - Unix: `~/.config/llm-council/config.yaml`
  - Project: `./.llm-council.yaml`
- Environment variable resolution (`${ENV_VAR}` syntax)
- CLI commands: `config show`, `config set`, `config unset`, `config init`, `config validate`

### Dependencies Analysis (`pyproject.toml`)

**Current Build System**:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```

**Current Dependencies**:
- litellm>=1.40.0
- openai>=1.0.0
- pydantic>=2.0.0
- click>=8.0.0
- rich>=13.0.0
- pyyaml>=6.0.0

**Missing**:
- `mcp` package not in dependencies (should be `mcp[cli]>=1.0.0`)

---

## Requirements & Clarification

### User Requirements (Validated)

1. **Config Management via MCP Tools**
   - `get_config` - Read current configuration
   - `set_config` - Modify configuration values
   - `init_config` - Initialize configuration interactively

2. **Interactive Onboarding**
   - AI asks user questions to configure providers
   - Support for multiple providers (OpenAI, Anthropic, LM Studio, Ollama)
   - Per-persona provider configuration

3. **UV/UVX Integration**
   - Installable via `uvx llm-council`
   - Proper entry points in pyproject.toml
   - Cross-platform support

4. **Docker Alternative**
   - Containerized deployment option
   - Proper socket mounting for AI clients

### Assumptions Made

1. MCP SDK version compatibility is 1.0.0+
2. Interactive onboarding will use MCP prompts mechanism
3. Config changes should persist immediately
4. Default transport is stdio (standard for MCP)

### Constraints

- Windows, Linux, and macOS support required
- Cannot break existing CLI functionality
- Must work with Claude Desktop, VS Code, etc.

### Success Criteria

- [ ] MCP server exposes config management tools
- [ ] Interactive onboarding works through MCP prompts
- [ ] `uvx llm-council-mcp` runs the server without installation
- [ ] Docker image available as alternative
- [ ] Existing CLI commands continue to work

---

## Research & Validation

### Best Practices Researched

**Sources**:
- [UV Documentation](https://docs.astral.sh/uv/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [mcp-server-git pyproject.toml](https://github.com/modelcontextprotocol/servers/blob/main/src/git/pyproject.toml)
- [FastMCP Documentation](https://pypi.org/project/fastmcp/)
- [Docker MCP Servers](https://www.docker.com/blog/build-to-prod-mcp-servers-with-docker/)
- [MCP Prompts Specification](https://modelcontextprotocol.io/specification/2025-06-18/server/prompts)

### MCP SDK Architecture

**Two approaches available**:

1. **Low-level SDK** (current implementation):
   ```python
   from mcp.server import Server
   from mcp.server.stdio import stdio_server
   ```

2. **FastMCP** (recommended for new tools):
   ```python
   from mcp.server.fastmcp import FastMCP
   mcp = FastMCP("llm-council")

   @mcp.tool()
   def my_tool(param: str) -> str:
       """Tool description"""
       return result
   ```

### UV/UVX Entry Point Pattern

Based on official MCP servers:

```toml
[project.scripts]
mcp-server-git = "mcp_server_git:main"
```

```python
# mcp_server_git/__init__.py
def main():
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("git")
    # ... register tools
    mcp.run()
```

### MCP Prompts for Interactive Onboarding

MCP prompts are user-controlled templates that can:
- Present structured configuration options
- Collect user preferences
- Guide through setup workflows

```python
@mcp.prompt()
def onboarding_prompt(provider_type: str) -> list:
    """Interactive onboarding for provider configuration"""
    return [
        PromptMessage(role="user", content=TextContent(
            type="text",
            text=f"Configure {provider_type} provider settings..."
        ))
    ]
```

### Docker MCP Server Pattern

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install -e .
CMD ["python", "-m", "llm_council.mcp_server"]
```

Claude Desktop config:
```json
{
  "mcpServers": {
    "llm-council": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-v", "/path/to/config:/root/.config/llm-council", "llm-council-mcp:latest"]
    }
  }
}
```

---

## Plan & Architecture

### High-Level Architecture

```
                    +-------------------+
                    |   Claude Desktop  |
                    |    / VS Code      |
                    +--------+----------+
                             | stdio
                             v
+-------------------------------------------------------------------+
|                     MCP Server (llm-council-mcp)                   |
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  |  council_discuss |  |   config_*       |  |   prompts_*      |  |
|  |  (existing)      |  |   tools          |  |   (onboarding)   |  |
|  +--------+---------+  +--------+---------+  +--------+---------+  |
|           |                     |                     |            |
|           v                     v                     v            |
|  +------------------+  +------------------+  +------------------+  |
|  |  CouncilEngine   |  |  ConfigManager   |  |  PromptManager   |  |
|  +--------+---------+  +--------+---------+  +--------+---------+  |
+-----------|----------------------|----------------------|---------+
            |                      |                      |
            v                      v                      v
    +---------------+      +---------------+      +---------------+
    |  LLM Provider |      |  Config Files |      |  User Input   |
    |  (LiteLLM)    |      |  (YAML)       |      |  (via MCP)    |
    +---------------+      +---------------+      +---------------+
```

### New MCP Tools Design

#### 1. config_get
```python
@mcp.tool()
def config_get(key: Optional[str] = None, include_sources: bool = False) -> dict:
    """Get configuration values.

    Args:
        key: Specific config key (e.g., 'defaults.model'). If None, returns all.
        include_sources: Include source information (env, file, default)

    Returns:
        Configuration dictionary
    """
```

#### 2. config_set
```python
@mcp.tool()
def config_set(key: str, value: str, project: bool = False) -> dict:
    """Set a configuration value.

    Args:
        key: Config key path (e.g., 'defaults.model', 'providers.openai.api_key')
        value: Value to set (use ${ENV_VAR} for env variables)
        project: If True, save to project config instead of user config

    Returns:
        Updated configuration for the key
    """
```

#### 3. config_init
```python
@mcp.tool()
def config_init(preset: Optional[str] = None, project: bool = False) -> dict:
    """Initialize configuration with defaults or a preset.

    Args:
        preset: Preset name ('lmstudio', 'openai', 'anthropic', 'ollama')
        project: If True, create project config instead of user config

    Returns:
        Created configuration
    """
```

#### 4. config_validate
```python
@mcp.tool()
def config_validate(test_connection: bool = True) -> dict:
    """Validate current configuration and optionally test provider connections.

    Args:
        test_connection: If True, test API connections (slower)

    Returns:
        Validation results with any errors/warnings
    """
```

#### 5. providers_list
```python
@mcp.tool()
def providers_list() -> dict:
    """List available provider presets and configured providers.

    Returns:
        Dict with 'presets' and 'configured' providers
    """
```

### MCP Prompts for Onboarding

#### 1. onboarding_start
```python
@mcp.prompt()
def onboarding_start() -> list:
    """Start the interactive configuration wizard.

    This prompt guides users through setting up their LLM Council configuration,
    including provider selection, API keys, and model preferences.
    """
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text="""Welcome to LLM Council configuration wizard!

I'll help you set up your LLM providers. Please answer the following:

1. Which LLM provider(s) would you like to use?
   - lmstudio (local, no API key needed)
   - openai (requires API key)
   - anthropic (requires API key)
   - ollama (local, no API key needed)
   - custom (specify your own endpoint)

2. Do you have API keys ready for cloud providers?

3. What is your primary use case?
   - Development/testing (local models recommended)
   - Production (cloud models recommended)
   - Mixed (use local for development, cloud for production)

Please provide your answers and I'll configure your setup."""
            )
        )
    ]
```

#### 2. provider_setup_prompt
```python
@mcp.prompt()
def provider_setup_prompt(provider: str) -> list:
    """Configure a specific provider.

    Args:
        provider: Provider name (openai, anthropic, etc.)
    """
```

### pyproject.toml Changes

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llm-council"
version = "0.1.0"
description = "LLM-based council/consensus tool for multi-persona AI deliberation"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "mahei"}
]
keywords = ["llm", "consensus", "council", "ai", "deliberation", "mcp"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
dependencies = [
    "litellm>=1.40.0",
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "click>=8.0.0",
    "rich>=13.0.0",
    "pyyaml>=6.0.0",
    "mcp>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pyright>=1.1.389",
    "ruff>=0.7.3",
]

[project.scripts]
llm-council = "llm_council.cli:main"
llm-council-mcp = "llm_council.mcp_server:main"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_council"]

[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pyright>=1.1.389",
    "ruff>=0.7.3",
]
```

### Directory Structure After Implementation

```
llm-council/
    pyproject.toml                 # Updated with hatchling + MCP
    Dockerfile                     # New: Docker support
    docker-compose.yml             # New: Docker Compose
    README.md
    .gitignore
    src/
        llm_council/
            __init__.py
            __main__.py
            mcp_server.py          # MODIFIED: Enhanced with config tools
            cli.py                 # Unchanged
            config.py              # Minor additions for MCP support
            providers.py           # Unchanged
            personas.py            # Unchanged
            council.py             # Unchanged
            models.py              # Unchanged
            ...
    tests/
        ...
    .scratchpad/
        handoffs/
```

---

## Implementation Phases

### Phase 1: MCP Server Enhancement (Priority: High)

1. **Add MCP dependency to pyproject.toml**
2. **Migrate to FastMCP** (cleaner tool registration)
3. **Add config management tools**:
   - `config_get`
   - `config_set`
   - `config_init`
   - `config_validate`
   - `providers_list`
4. **Add onboarding prompts**

### Phase 2: UV/UVX Packaging (Priority: High)

1. **Migrate from setuptools to hatchling**
2. **Add `llm-council-mcp` entry point**
3. **Test with `uv run` and `uvx`**
4. **Update documentation**

### Phase 3: Docker Support (Priority: Medium)

1. **Create Dockerfile**
2. **Create docker-compose.yml**
3. **Add volume mounts for config persistence**
4. **Test with Claude Desktop**

### Phase 4: Documentation & Testing (Priority: Medium)

1. **Update README with installation options**
2. **Add MCP-specific documentation**
3. **Add integration tests for MCP tools**

---

## Task Breakdown

| Task | Description | Complexity | Agent | Dependencies |
|------|-------------|------------|-------|--------------|
| 1.1 | Add `mcp>=1.0.0` to dependencies | Simple | @general-programmer-agent | None |
| 1.2 | Refactor mcp_server.py to FastMCP | Moderate | @general-programmer-agent | 1.1 |
| 1.3 | Implement config_get tool | Simple | @general-programmer-agent | 1.2 |
| 1.4 | Implement config_set tool | Simple | @general-programmer-agent | 1.2 |
| 1.5 | Implement config_init tool | Simple | @general-programmer-agent | 1.2 |
| 1.6 | Implement config_validate tool | Moderate | @general-programmer-agent | 1.2 |
| 1.7 | Implement providers_list tool | Simple | @general-programmer-agent | 1.2 |
| 1.8 | Add onboarding prompts | Moderate | @general-programmer-agent | 1.2 |
| 2.1 | Migrate to hatchling build system | Moderate | @general-programmer-agent | None |
| 2.2 | Add llm-council-mcp entry point | Simple | @general-programmer-agent | 2.1 |
| 2.3 | Test uvx installation | Simple | @general-programmer-agent | 2.2 |
| 3.1 | Create Dockerfile | Simple | @general-programmer-agent | 2.1 |
| 3.2 | Create docker-compose.yml | Simple | @general-programmer-agent | 3.1 |
| 3.3 | Document Docker usage | Simple | @project-docs-writer | 3.2 |
| 4.1 | Update README | Moderate | @project-docs-writer | 2.3, 3.2 |
| 4.2 | Add MCP integration tests | Moderate | @general-programmer-agent | 1.8 |

---

## Risks & Mitigations

### Identified Risks

1. **Risk**: MCP SDK version incompatibility
   **Impact**: Medium
   **Mitigation**: Pin to specific version, test with multiple clients

2. **Risk**: Config file locking issues on Windows
   **Impact**: Low
   **Mitigation**: Use proper file handling with context managers

3. **Risk**: Interactive prompts not supported by all MCP clients
   **Impact**: Medium
   **Mitigation**: Provide fallback via config_init tool with presets

4. **Risk**: Docker socket permissions on different platforms
   **Impact**: Medium
   **Mitigation**: Document platform-specific setup, provide alternatives

### Security Considerations

1. **API Key Handling**: Always encourage `${ENV_VAR}` syntax, warn on plaintext keys
2. **Config File Permissions**: Document secure file permissions
3. **Docker**: Don't expose sensitive ports, use read-only mounts where possible

### Performance Considerations

1. **Config Loading**: Cache resolved config in MCP server
2. **Provider Testing**: Make connection tests optional
3. **Docker**: Use multi-stage builds for smaller images

---

## Deliverables

### Files to Create

| File | Purpose |
|------|---------|
| `Dockerfile` | Docker image build |
| `docker-compose.yml` | Docker Compose orchestration |
| `.dockerignore` | Exclude files from Docker build |

### Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Add mcp dependency, hatchling build, entry points |
| `src/llm_council/mcp_server.py` | Complete rewrite with FastMCP and new tools |
| `src/llm_council/config.py` | Minor: add helper methods for MCP tools |
| `README.md` | Add UV/UVX and Docker installation instructions |

### Agent Assignments

- **@general-programmer-agent**: All implementation tasks (1.x, 2.x, 3.1-3.2, 4.2)
- **@project-docs-writer**: Documentation (3.3, 4.1)

---

## Next Steps

### Immediate Actions

1. Create implementation specification for mcp_server.py rewrite
2. Update pyproject.toml with new dependencies and build system
3. Implement config management MCP tools
4. Test with Claude Desktop

### Questions for User

1. Should we support both the current `mcp.server` and FastMCP approaches, or fully migrate to FastMCP?
2. Are there specific MCP clients beyond Claude Desktop that need testing?
3. Should Docker be a first-class option or secondary to UV/UVX?

---

## Files Analyzed

### Read/Analyzed
- `C:\Users\mahei\Documents\llm-council\src\llm_council\mcp_server.py` - Current MCP server (151 lines, single tool)
- `C:\Users\mahei\Documents\llm-council\pyproject.toml` - Build config (needs mcp dependency)
- `C:\Users\mahei\Documents\llm-council\src\llm_council\config.py` - Full config system (470 lines)
- `C:\Users\mahei\Documents\llm-council\src\llm_council\providers.py` - Provider registry (294 lines)
- `C:\Users\mahei\Documents\llm-council\src\llm_council\cli.py` - CLI with config commands (770 lines)
- `C:\Users\mahei\Documents\llm-council\src\llm_council\personas.py` - Persona management (271 lines)
- `C:\Users\mahei\Documents\llm-council\src\llm_council\models.py` - Data models (226 lines)
- `C:\Users\mahei\Documents\llm-council\src\llm_council\__init__.py` - Package exports (224 lines)
- `C:\Users\mahei\Documents\llm-council\src\llm_council\__main__.py` - Entry point (7 lines)

---

## Installation Paths

### UV/UVX (Recommended)

**Windows**:
```powershell
# Install UV
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Run MCP server (no installation needed)
uvx llm-council-mcp

# Or install as tool
uv tool install llm-council
llm-council-mcp
```

**Linux/macOS**:
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run MCP server
uvx llm-council-mcp
```

### Claude Desktop Configuration

**Using uvx**:
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

**Using Docker**:
```json
{
  "mcpServers": {
    "llm-council": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "%APPDATA%/llm-council:/root/.config/llm-council",
        "llm-council-mcp:latest"
      ]
    }
  }
}
```

### VS Code / Copilot Configuration

```json
{
  "mcp": {
    "servers": {
      "llm-council": {
        "command": "uvx",
        "args": ["llm-council-mcp"]
      }
    }
  }
}
```

---

## Handoff Checklist

- [x] All requirements clarified and documented
- [x] Architecture designed and documented
- [x] Risks identified with mitigations
- [x] Tasks broken down with clear acceptance criteria
- [x] Agents assigned for implementation
- [x] Specifications created for all major components
- [ ] User alignment and plan approval (pending)
- [x] All artifacts saved in appropriate locations
- [x] Next steps clearly defined

---

**Prepared by**: plan agent
**Ready for**: Implementation by @general-programmer-agent
**Contact**: Escalate questions to user

---

## Appendix

### Research URLs

- [UV Documentation](https://docs.astral.sh/uv/) - Package manager installation and usage
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) - Official SDK reference
- [FastMCP](https://pypi.org/project/fastmcp/) - Simplified MCP server framework
- [MCP Prompts Specification](https://modelcontextprotocol.io/specification/2025-06-18/server/prompts) - Interactive prompts
- [Docker MCP Servers](https://www.docker.com/blog/build-to-prod-mcp-servers-with-docker/) - Containerization guide
- [mcp-server-git pyproject.toml](https://github.com/modelcontextprotocol/servers/blob/main/src/git/pyproject.toml) - Reference implementation

### Key Code Patterns

**FastMCP Tool Registration**:
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("llm-council")

@mcp.tool()
def my_tool(param: str) -> dict:
    """Tool description for LLM."""
    return {"result": "value"}

def main():
    mcp.run()
```

**MCP Prompt Registration**:
```python
from mcp.types import PromptMessage, TextContent

@mcp.prompt()
def my_prompt(arg: str) -> list[PromptMessage]:
    """Prompt description."""
    return [
        PromptMessage(
            role="user",
            content=TextContent(type="text", text="Prompt text")
        )
    ]
```

---

*End of Handoff Document*
