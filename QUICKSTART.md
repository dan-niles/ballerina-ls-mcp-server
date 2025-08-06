# Quick Start Guide

## Installation

### Option 1: Install from source (Development)
```bash
# Clone and install in editable mode
git clone <repository-url>
cd java-mcp-server
uv pip install -e .
```

### Option 2: Install from wheel
```bash
# Build and install the wheel
uv build
uv pip install dist/ballerina_language_server_mcp-0.2.0-py3-none-any.whl
```

## Configuration

Set the required environment variable:
```bash
export BALLERINA_REPO_PATH="/path/to/ballerina-lang-repository"
```

Optional environment variables:
```bash
export BALLERINA_DB_PATH="custom_index.db"      # Default: ballerina_index.db
export BALLERINA_MAX_FILE_SIZE="20971520"       # Default: 10MB (in bytes)
```

## Running

### Option 1: Using uv (recommended for development)
```bash
uv run python run.py
```

### Option 2: Using the package module
```bash
uv run python -m ballerina_mcp.main
```

### Option 3: Direct execution (if dependencies are installed globally)
```bash
python3 run.py
```

## Verification

Test that the server starts correctly:
```bash
# Should show "Configuration error" if BALLERINA_REPO_PATH is not set
uv run python run.py

# With proper configuration, should start indexing
BALLERINA_REPO_PATH=/path/to/repo uv run python run.py
```

## Development

To modify the server:

1. Edit files in `ballerina_mcp/`
2. Test changes: `uv run python run.py`
3. Build package: `uv build`
4. Install updated package: `uv pip install -e .`

## Package Structure

```
java-mcp-server/
├── ballerina_mcp/          # Main package
│   ├── __init__.py         # Package metadata
│   └── main.py             # Core MCP server code
├── run.py                  # Convenience runner script
├── README.md               # Full documentation
├── pyproject.toml          # Package configuration
└── MANIFEST.in             # File inclusion rules
```
