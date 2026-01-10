# LLM Council MCP Server Docker Image
# Build: docker build -t llm-council-mcp .
# Run: docker run -it --rm llm-council-mcp

FROM python:3.12-slim

LABEL maintainer="mahei"
LABEL description="LLM Council MCP Server - Multi-persona AI deliberation tool"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 council && \
    useradd --uid 1000 --gid council --shell /bin/bash --create-home council

# Set working directory
WORKDIR /app

# Install UV for faster package management (optional but recommended)
RUN pip install uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package
RUN uv pip install --system -e .

# Switch to non-root user
USER council

# Create config directory for the user
RUN mkdir -p /home/council/.config/llm-council

# Default command runs the MCP server
ENTRYPOINT ["python", "-m", "llm_council.mcp_server"]

# Health check - verify the module can be imported
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from llm_council.mcp_server import server; print('OK')" || exit 1
