# Plan Agent Handoff - MCP Onboarding Flow Analysis & Improvement Plan

**Agent**: plan
**Status**: SUCCESS
**Timestamp**: 2026-01-10
**Task**: Analyze current config_init onboarding flow and propose comprehensive improvements

---

## Executive Summary

The current `config_init` MCP tool provides a basic two-step flow (select preset, save config) that is incomplete for a good onboarding experience. This analysis identifies the gaps and proposes a comprehensive multi-step wizard that guides users through all essential configuration while validating settings before saving.

**Key Findings**:
1. Current flow skips model selection for most presets
2. No connection validation before saving
3. Council settings (max_rounds, consensus_type) are never configured
4. Persona preferences are completely ignored
5. API key guidance for cloud providers is minimal
6. No feedback on what was configured or how to test it

---

## Task Context

**Original Request**:
Analyze the current MCP onboarding flow and identify what's missing from a good onboarding experience.

**Objective**:
Design an improved multi-step onboarding flow that guides users through essential settings, validates configuration before saving, and works well with AI agents.

---

## Current Flow Analysis

### Step-by-Step Breakdown of Current `config_init`

**Location**: `C:\Users\mahei\Documents\llm-council\src\llm_council\mcp_server.py` (lines 383-495)

#### Call 1: No Parameters (Discovery)

```python
# User/Agent calls: config_init()
# Response:
{
    "action": "onboarding_required",
    "message": "LLM Council configuration setup...",
    "questions": [
        {
            "id": "preset",
            "question": "Which LLM provider would you like to use?",
            "options": [
                {"value": "local", "label": "Local (LM Studio, Ollama, etc.)"},
                {"value": "openai", "label": "OpenAI"},
                {"value": "anthropic", "label": "Anthropic Claude"},
                {"value": "custom", "label": "Custom Provider"}
            ]
        }
    ],
    "next_step": "After user selects preset, call config_init with preset parameter..."
}
```

**What happens**: Returns a JSON structure describing available presets for the agent to present to the user.

#### Call 2: With Preset Selected

```python
# User/Agent calls: config_init(preset="local")
# Response (for non-custom presets):
{
    "success": True,
    "message": "Configuration initialized with 'local' preset",
    "config_path": "C:\\Users\\mahei\\AppData\\Roaming\\llm-council\\config.yaml",
    "config": {
        "defaults": {
            "model": "openai/qwen/qwen3-coder-30b",
            "api_base": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "temperature": 0.7,
            "max_tokens": 1024
        }
    },
    "next_step": "You can now use council_discuss..."
}
```

**What happens**:
1. Loads preset defaults (model, api_base, api_key)
2. Immediately saves to user config file
3. Returns success with the saved config

#### Call 2 (Custom Preset - Different Behavior)

```python
# User/Agent calls: config_init(preset="custom")
# Response:
{
    "action": "need_more_info",
    "preset": "custom",
    "questions": [
        {"id": "api_base", "question": "What is your API endpoint URL?", "required": True},
        {"id": "model", "question": "What is the model identifier?", "required": True},
        {"id": "api_key", "question": "What is your API key?", "sensitive": True}
    ],
    "message": "Please gather the following information from the user:"
}
```

**What happens**: Returns additional questions for custom preset only.

### Preset Configurations (Current Defaults)

| Preset | Model | API Base | API Key |
|--------|-------|----------|---------|
| local | openai/qwen/qwen3-coder-30b | http://localhost:1234/v1 | lm-studio |
| openai | gpt-4o | https://api.openai.com/v1 | ${OPENAI_API_KEY} |
| anthropic | anthropic/claude-3-5-sonnet-20241022 | https://api.anthropic.com | ${ANTHROPIC_API_KEY} |
| custom | (user provided) | (user provided) | (user provided) |

---

## Gap Analysis: What's Missing

### 1. Model Selection (Critical Gap)

**Problem**: Users cannot choose which model to use with their provider.

**Current behavior**:
- Local preset always uses `openai/qwen/qwen3-coder-30b` (assumes LM Studio with specific model)
- OpenAI preset always uses `gpt-4o` (no choice of gpt-4o-mini, gpt-4-turbo, etc.)
- Anthropic preset always uses `claude-3-5-sonnet-20241022` (no opus, haiku options)

**User expectations**:
- "I'm using Ollama, not LM Studio, and I have llama3:8b loaded"
- "I want to use gpt-4o-mini for cost savings"
- "I want to use claude-3-opus for complex discussions"

**Recommendation**: Add model selection step after provider selection with:
- Common models listed for each provider
- "Custom model name" option for unlisted models
- For local: option to detect running models via API

### 2. Connection Validation (Critical Gap)

**Problem**: Config is saved without verifying the provider is reachable.

**Current behavior**:
- Config is saved immediately after preset selection
- User doesn't know if it works until they try `council_discuss`
- If connection fails, user must troubleshoot manually

**User expectations**:
- "Test the connection before saving"
- "Tell me if my API key is valid"
- "Confirm my local LLM server is running"

**Recommendation**: Add validation step that:
- Tests API connectivity before saving
- For cloud providers: validates API key
- For local: checks if server is running
- Shows clear error messages with troubleshooting tips
- Allows user to retry or adjust settings

### 3. Council Settings (Moderate Gap)

**Problem**: Core council behavior settings are never configured.

**Missing settings from ConfigSchema**:
```python
class CouncilSettings(BaseModel):
    consensus_type: str = "majority"          # Never asked
    max_rounds: int = 5                       # Never asked
    stalemate_threshold: int = 2              # Never asked
    default_personas_count: int = 3           # Never asked
    auto_personas: bool = False               # Never asked
```

**User expectations**:
- "I want unanimous consensus for important decisions"
- "I want more discussion rounds for complex topics"
- "I want 5 personas instead of 3 by default"

**Recommendation**: Add optional council settings step:
- Show sensible defaults
- Allow customization
- Explain what each setting does
- Mark as "advanced" for users who want quick setup

### 4. API Key Guidance (Moderate Gap)

**Problem**: Cloud providers need API keys but guidance is minimal.

**Current behavior**:
- OpenAI preset uses `${OPENAI_API_KEY}` - assumes env var exists
- Anthropic preset uses `${ANTHROPIC_API_KEY}` - assumes env var exists
- No explanation of how to set env vars
- No link to where to get API keys

**User expectations**:
- "How do I get an OpenAI API key?"
- "How do I set an environment variable on Windows?"
- "Can I just paste my key here?"

**Recommendation**: Add API key guidance:
- Link to provider's API key page (openai.com/api-keys, console.anthropic.com)
- Explain environment variable syntax and why it's recommended
- Provide platform-specific instructions (Windows vs Unix)
- Allow inline key entry with security warning
- Test key validity immediately

### 5. Persona Preferences (Minor Gap)

**Problem**: Per-persona provider configuration is completely ignored.

**Current behavior**:
- `persona_configs` section in ConfigSchema exists but is never used in onboarding
- Users who want different models for different personas must manually edit config

**User expectations**:
- "I want my 'Devil's Advocate' persona to use a different model"
- "I want to use a cheaper model for some personas"

**Recommendation**: For initial onboarding, this is too advanced. Instead:
- Mention that per-persona configs are possible
- Point to documentation for advanced configuration
- Consider adding a separate `config_personas` tool for this

### 6. Generation Settings (Minor Gap)

**Problem**: Persona generation settings are not configured.

**Missing settings from ConfigSchema**:
```python
class GenerationSettings(BaseModel):
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: float = 0.8
    max_tokens: int = 2048
    prompt_template: Optional[str] = None
```

**User expectations**:
- "I want more creative persona generation"
- "I want to use a specific model for generating personas"

**Recommendation**: This is too advanced for initial onboarding. Mention in docs.

### 7. Post-Setup Guidance (Minor Gap)

**Problem**: After saving config, user doesn't know what to do next.

**Current behavior**:
- Returns `"next_step": "You can now use council_discuss..."`
- No example of how to use council_discuss
- No mention of config_validate for testing

**Recommendation**: Add post-setup summary:
- Confirm what was configured
- Provide example council_discuss call
- Suggest running config_validate to test
- Link to documentation for more options

---

## Proposed Improved Onboarding Flow

### Design Principles

1. **Progressive disclosure**: Start simple, allow drilling into advanced options
2. **Fail fast**: Validate early, don't save broken configs
3. **AI-agent friendly**: Return structured JSON for agents to present as conversation
4. **Resumable**: Allow partial configuration, save progress
5. **Sensible defaults**: Every setting has a good default

### Flow States

```
[START]
    |
    v
[STEP 1: Provider Selection]
    |
    v
[STEP 2: Model Selection] (varies by provider)
    |
    v
[STEP 3: API Key Setup] (cloud providers only)
    |
    v
[STEP 4: Connection Validation] (automatic)
    |
    +---> [FAIL] ---> [Troubleshooting] ---> [RETRY from STEP 1-3]
    |
    v
[STEP 5: Council Settings] (optional, show defaults)
    |
    v
[STEP 6: Save & Summary]
    |
    v
[END]
```

### New config_init Tool Design

#### Option A: Single Tool with State Machine

```python
@mcp.tool()
async def config_init(
    step: Optional[str] = None,
    preset: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_rounds: Optional[int] = None,
    consensus_type: Optional[str] = None,
    default_personas_count: Optional[int] = None,
    skip_validation: bool = False,
    skip_council_settings: bool = True,
) -> dict:
    """Initialize LLM Council configuration with guided setup.

    Call without parameters to start onboarding. Each response includes
    the current step, questions to ask, and what parameters to send next.

    Steps:
    1. provider_select - Choose provider preset
    2. model_select - Choose model for the provider
    3. api_key_setup - Configure API key (cloud providers only)
    4. validate - Test connection (automatic)
    5. council_settings - Configure council behavior (optional)
    6. save - Save configuration and show summary
    """
```

#### Option B: Separate Step Tools (Recommended)

```python
@mcp.tool()
async def onboard_start() -> dict:
    """Start LLM Council onboarding. Returns provider selection options."""

@mcp.tool()
async def onboard_provider(preset: str) -> dict:
    """Set provider preset. Returns model selection options for that provider."""

@mcp.tool()
async def onboard_model(model: str) -> dict:
    """Set model. Returns API key setup (if needed) or validation status."""

@mcp.tool()
async def onboard_api_key(api_key: str) -> dict:
    """Set API key. Returns validation status."""

@mcp.tool()
async def onboard_validate() -> dict:
    """Validate current configuration. Returns success or error with fixes."""

@mcp.tool()
async def onboard_council(
    max_rounds: int = 5,
    consensus_type: str = "majority",
    default_personas_count: int = 3
) -> dict:
    """Configure council settings. Returns summary."""

@mcp.tool()
async def onboard_save() -> dict:
    """Save configuration and return summary with next steps."""

@mcp.tool()
async def onboard_cancel() -> dict:
    """Cancel onboarding without saving."""
```

### Recommended Approach: Enhanced Single Tool

Keep the single `config_init` tool but enhance it with:

1. **Step parameter** to track progress
2. **Validation before save** (automatic)
3. **Model options per provider**
4. **Council settings as optional step**
5. **Rich error messages and troubleshooting**

---

## Detailed Step Specifications

### Step 1: Provider Selection

**Input**: `config_init()` (no params)

**Output**:
```json
{
    "step": "provider_select",
    "step_number": 1,
    "total_steps": 5,
    "message": "Welcome to LLM Council setup. Which LLM provider would you like to use?",
    "questions": [
        {
            "id": "preset",
            "type": "select",
            "question": "Select your LLM provider:",
            "options": [
                {
                    "value": "local",
                    "label": "Local Server",
                    "description": "LM Studio, Ollama, or other local LLM server",
                    "requires_api_key": false
                },
                {
                    "value": "openai",
                    "label": "OpenAI",
                    "description": "GPT-4o, GPT-4o-mini, and other OpenAI models",
                    "requires_api_key": true,
                    "api_key_url": "https://platform.openai.com/api-keys"
                },
                {
                    "value": "anthropic",
                    "label": "Anthropic Claude",
                    "description": "Claude Opus, Sonnet, and Haiku models",
                    "requires_api_key": true,
                    "api_key_url": "https://console.anthropic.com/settings/keys"
                },
                {
                    "value": "custom",
                    "label": "Custom Endpoint",
                    "description": "Any OpenAI-compatible API endpoint",
                    "requires_api_key": "optional"
                }
            ]
        }
    ],
    "next_action": "Call config_init(preset='<selected>') to continue"
}
```

### Step 2: Model Selection

**Input**: `config_init(preset="openai")`

**Output** (OpenAI example):
```json
{
    "step": "model_select",
    "step_number": 2,
    "total_steps": 5,
    "preset": "openai",
    "message": "Great! Now select which OpenAI model you'd like to use.",
    "questions": [
        {
            "id": "model",
            "type": "select",
            "question": "Select model:",
            "options": [
                {
                    "value": "gpt-4o",
                    "label": "GPT-4o",
                    "description": "Most capable model, best for complex discussions",
                    "recommended": true
                },
                {
                    "value": "gpt-4o-mini",
                    "label": "GPT-4o Mini",
                    "description": "Faster and cheaper, good for most use cases"
                },
                {
                    "value": "gpt-4-turbo",
                    "label": "GPT-4 Turbo",
                    "description": "Previous generation, still very capable"
                },
                {
                    "value": "custom",
                    "label": "Other model",
                    "description": "Enter a custom model name"
                }
            ],
            "allow_custom": true,
            "custom_hint": "Enter model name (e.g., gpt-4o-2024-11-20)"
        }
    ],
    "next_action": "Call config_init(preset='openai', model='<selected>') to continue"
}
```

**Output** (Local example):
```json
{
    "step": "model_select",
    "step_number": 2,
    "total_steps": 5,
    "preset": "local",
    "message": "Configure your local LLM server.",
    "questions": [
        {
            "id": "api_base",
            "type": "text",
            "question": "What is your local server URL?",
            "default": "http://localhost:1234/v1",
            "hint": "LM Studio default: http://localhost:1234/v1, Ollama: http://localhost:11434/v1"
        },
        {
            "id": "model",
            "type": "text",
            "question": "What model are you running?",
            "default": "openai/qwen/qwen3-coder-30b",
            "hint": "Use 'openai/' prefix for OpenAI-compatible endpoints"
        }
    ],
    "detect_models": {
        "supported": true,
        "endpoint": "{api_base}/models",
        "hint": "We can try to detect available models from your server"
    },
    "next_action": "Call config_init(preset='local', api_base='...', model='...') to continue"
}
```

### Step 3: API Key Setup (Cloud Providers Only)

**Input**: `config_init(preset="openai", model="gpt-4o")`

**Output**:
```json
{
    "step": "api_key_setup",
    "step_number": 3,
    "total_steps": 5,
    "preset": "openai",
    "model": "gpt-4o",
    "message": "OpenAI requires an API key. How would you like to configure it?",
    "questions": [
        {
            "id": "api_key",
            "type": "select_or_input",
            "question": "Configure API key:",
            "options": [
                {
                    "value": "${OPENAI_API_KEY}",
                    "label": "Use environment variable (recommended)",
                    "description": "References OPENAI_API_KEY environment variable",
                    "recommended": true,
                    "setup_instructions": {
                        "windows": "setx OPENAI_API_KEY \"sk-...\" (then restart terminal)",
                        "unix": "export OPENAI_API_KEY=\"sk-...\" (add to ~/.bashrc)"
                    }
                },
                {
                    "value": "paste",
                    "label": "Enter API key directly",
                    "description": "Key will be stored in config file",
                    "warning": "Less secure - key stored in plaintext"
                }
            ],
            "api_key_url": "https://platform.openai.com/api-keys",
            "api_key_hint": "Get your API key at platform.openai.com/api-keys"
        }
    ],
    "next_action": "Call config_init(preset='openai', model='gpt-4o', api_key='...') to continue"
}
```

### Step 4: Connection Validation (Automatic)

**Input**: `config_init(preset="openai", model="gpt-4o", api_key="${OPENAI_API_KEY}")`

**Output** (Success):
```json
{
    "step": "validate",
    "step_number": 4,
    "total_steps": 5,
    "validation_status": "success",
    "message": "Connection successful! Your configuration is working.",
    "validation_details": {
        "provider": "openai",
        "model": "gpt-4o",
        "api_base": "https://api.openai.com/v1",
        "connection_test": "passed",
        "response_time_ms": 450
    },
    "questions": [
        {
            "id": "configure_council",
            "type": "confirm",
            "question": "Would you like to customize council settings? (Default: 3 personas, majority consensus, 5 rounds max)",
            "default": false
        }
    ],
    "next_action": "Call config_init(..., skip_council_settings=true) to save, or config_init(..., skip_council_settings=false) to customize council settings"
}
```

**Output** (Failure):
```json
{
    "step": "validate",
    "step_number": 4,
    "total_steps": 5,
    "validation_status": "failed",
    "message": "Connection failed. Please check your configuration.",
    "error": {
        "type": "authentication_error",
        "message": "Invalid API key",
        "details": "The API key provided was rejected by OpenAI"
    },
    "troubleshooting": [
        "Verify your API key is correct (starts with 'sk-')",
        "Check if the API key has been revoked",
        "Ensure you have billing set up at platform.openai.com",
        "If using env var, restart your terminal after setting it"
    ],
    "retry_options": [
        {
            "action": "Change API key",
            "call": "config_init(preset='openai', model='gpt-4o', api_key='<new_key>')"
        },
        {
            "action": "Change provider",
            "call": "config_init()"
        },
        {
            "action": "Skip validation and save anyway",
            "call": "config_init(..., skip_validation=true)",
            "warning": "Not recommended - config may not work"
        }
    ]
}
```

### Step 5: Council Settings (Optional)

**Input**: `config_init(..., skip_council_settings=false)`

**Output**:
```json
{
    "step": "council_settings",
    "step_number": 5,
    "total_steps": 5,
    "message": "Customize your council settings.",
    "questions": [
        {
            "id": "consensus_type",
            "type": "select",
            "question": "What type of consensus should the council require?",
            "options": [
                {"value": "unanimous", "label": "Unanimous", "description": "All personas must agree"},
                {"value": "supermajority", "label": "Supermajority", "description": "2/3 must agree"},
                {"value": "majority", "label": "Majority", "description": "More than half must agree", "default": true},
                {"value": "plurality", "label": "Plurality", "description": "Most common position wins"}
            ],
            "default": "majority"
        },
        {
            "id": "max_rounds",
            "type": "number",
            "question": "Maximum discussion rounds before stopping?",
            "default": 5,
            "min": 1,
            "max": 20,
            "hint": "More rounds allow deeper discussion but take longer"
        },
        {
            "id": "default_personas_count",
            "type": "number",
            "question": "Default number of personas for discussions?",
            "default": 3,
            "min": 2,
            "max": 10,
            "hint": "More personas = more perspectives but longer discussions"
        }
    ],
    "next_action": "Call config_init(..., consensus_type='...', max_rounds=N, default_personas_count=N) to save"
}
```

### Step 6: Save & Summary

**Input**: All parameters provided

**Output**:
```json
{
    "step": "complete",
    "status": "success",
    "message": "Configuration saved successfully!",
    "config_path": "C:\\Users\\mahei\\AppData\\Roaming\\llm-council\\config.yaml",
    "summary": {
        "provider": "OpenAI",
        "model": "gpt-4o",
        "api_key": "${OPENAI_API_KEY} (from environment)",
        "council": {
            "consensus_type": "majority",
            "max_rounds": 5,
            "default_personas_count": 3
        }
    },
    "next_steps": [
        {
            "action": "Run a test discussion",
            "example": "council_discuss(topic='Test Topic', objective='Verify setup works')"
        },
        {
            "action": "Validate configuration",
            "example": "config_validate()"
        },
        {
            "action": "View full configuration",
            "example": "config_get(resolved=true)"
        }
    ],
    "documentation": {
        "advanced_config": "Edit config file at the path above for advanced options",
        "per_persona_models": "Add persona_configs section to use different models per persona",
        "project_config": "Create .llm-council.yaml in project root for project-specific settings"
    }
}
```

---

## Implementation Plan

### Task Breakdown

| Task ID | Description | Complexity | Dependencies |
|---------|-------------|------------|--------------|
| ON-01 | Add step tracking to config_init | Simple | None |
| ON-02 | Implement model selection per provider | Moderate | ON-01 |
| ON-03 | Add API key guidance with provider links | Simple | ON-01 |
| ON-04 | Implement connection validation | Moderate | ON-02, ON-03 |
| ON-05 | Add troubleshooting for validation failures | Moderate | ON-04 |
| ON-06 | Add council settings step | Simple | ON-04 |
| ON-07 | Implement save with summary | Simple | ON-06 |
| ON-08 | Add model detection for local servers | Moderate | ON-02 |
| ON-09 | Update tests for new flow | Moderate | ON-07 |

### Files to Modify

1. **`src/llm_council/mcp_server.py`**
   - Refactor `handle_config_init()` with step tracking
   - Add model options per provider
   - Add validation step
   - Add council settings step
   - Improve error messages and troubleshooting

2. **`src/llm_council/providers.py`** (minor)
   - Add model lists per provider type
   - Add model detection for local servers

3. **`src/llm_council/config.py`** (minor)
   - Add helper for partial config (onboarding state)

### Estimated Effort

- **Total**: 4-6 hours of implementation
- **Testing**: 2-3 hours
- **Documentation**: 1 hour

---

## Risks & Mitigations

### 1. State Management Between Calls

**Risk**: MCP tools are stateless; how to track onboarding progress?

**Options**:
- **Option A**: Pass all previous params on each call (recommended for simplicity)
- **Option B**: Store state in temp file with session ID
- **Option C**: Return state object that agent passes back

**Recommendation**: Option A - require all previous params on each call. The agent can accumulate them.

### 2. Model Detection for Local Servers

**Risk**: Not all local servers support `/models` endpoint.

**Mitigation**: Make detection optional with fallback to manual entry.

### 3. API Key Security

**Risk**: Users might paste sensitive keys into config.

**Mitigation**:
- Strong preference for env vars in UI
- Security warning for inline keys
- Mask key display after entry

---

## Acceptance Criteria

1. **Step tracking**: Each call returns current step and total steps
2. **Model selection**: User can choose from list or enter custom model
3. **Validation**: Connection is tested before config is saved
4. **Error handling**: Clear error messages with troubleshooting tips
5. **Council settings**: Optional step to customize council behavior
6. **Summary**: Final step shows what was configured and what to do next
7. **Backward compatible**: Existing `config_init(preset="local")` still works

---

## Comparison: Current vs Proposed

| Aspect | Current | Proposed |
|--------|---------|----------|
| Steps | 2 (provider, save) | 5-6 (provider, model, api_key, validate, council, save) |
| Model selection | Fixed per preset | User choice with suggestions |
| Validation | None before save | Automatic with retry |
| Council settings | Never | Optional step with defaults |
| API key guidance | Minimal | Links, env var instructions |
| Error handling | Basic | Detailed with troubleshooting |
| Summary | Minimal | Comprehensive with next steps |

---

## Agent Assignment

**Primary**: @general-programmer-agent

**Specification**: See Step Specifications section above for exact JSON schemas.

---

## Handoff Checklist

- [x] Current flow analyzed step-by-step
- [x] Gaps identified with severity
- [x] New flow designed with all steps
- [x] JSON schemas for each step defined
- [x] Implementation tasks broken down
- [x] Risks identified with mitigations
- [x] Acceptance criteria defined
- [x] Agent assigned

---

**Prepared by**: plan agent
**Ready for**: Implementation by @general-programmer-agent
**Reference files**:
- Current implementation: `C:\Users\mahei\Documents\llm-council\src\llm_council\mcp_server.py`
- Config schema: `C:\Users\mahei\Documents\llm-council\src\llm_council\config.py`
- Providers: `C:\Users\mahei\Documents\llm-council\src\llm_council\providers.py`

---

*End of Handoff Document*
