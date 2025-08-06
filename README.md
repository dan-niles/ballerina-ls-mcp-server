# Ballerina Language Server MCP Server

An intelligent Model Context Protocol (MCP) server for querying and analyzing Java language server codebases, specifically designed for the Ballerina Language Server project.

## Features

### 🔍 **Advanced Code Search**
- **Basic Search**: Search through classes, methods, and fields by name or content
- **Fuzzy Search**: Enhanced search with relevance scoring and multi-word matching
- **Similarity Search**: Find methods with similar names using phonetic matching

### 📊 **Repository Analytics**
- Repository statistics (file count, class count, method count)
- Package distribution analysis
- Largest classes identification
- Code metrics and insights

### 🏗️ **LSP Protocol Discovery**
- Find LSP protocol method implementations
- Search for specific language server features
- Analyze protocol handler patterns

### ⚡ **Performance Optimized**
- SQLite-based indexing for fast queries
- Tree-sitter AST parsing for accurate code analysis
- Caching for frequently accessed queries
- Incremental re-indexing support

## Installation

### Prerequisites
- Python 3.13+
- Git (for cloning repositories)

### Setup
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install fastmcp tree-sitter tree-sitter-java
   ```

### Alternative: Using uv
```bash
uv pip install fastmcp tree-sitter tree-sitter-java
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

## Usage

### Running the Server
```bash
# Option 1: Using the installed script (after installation)
ballerina-mcp

# Option 2: Using the run script
python run.py

# Option 3: Using the module directly
python -m ballerina_mcp.main
```

### Available MCP Tools

#### 1. `search_code(query: str, limit: int = 10)`
Basic search through the codebase for classes, methods, and content.

**Example:**
```
Query: "completion"
Returns: Classes and methods related to code completion
```

#### 2. `advanced_search(query: str, search_type: str = "fuzzy", limit: int = 10)`
Enhanced search with different strategies:
- `fuzzy`: Multi-word matching with relevance scoring
- `exact`: Exact string matching

**Example:**
```
Query: "text document hover", Type: "fuzzy"
Returns: Ranked results for hover functionality
```

#### 3. `get_class_info(class_name: str)`
Get detailed information about a specific class including all methods and fields.

**Example:**
```
Query: "CompletionProvider"
Returns: Full class definition, methods, and context
```

#### 4. `get_repository_stats()`
Get comprehensive statistics about the indexed repository.

**Returns:**
- File count and distribution
- Top packages by class count
- Largest classes by line count
- Indexing health metrics

#### 5. `find_similar_methods(method_name: str, limit: int = 5)`
Find methods with similar names using phonetic matching algorithms.

**Example:**
```
Query: "getCompletion"
Returns: getCompletion, getCompletions, findCompletion, etc.
```

#### 6. `reindex_repository()`
Re-index the repository to pick up new changes.

#### 7. `find_lsp_protocol_implementations(protocol_method: str = "")`
Find implementations of specific LSP protocol methods.

**Example:**
```
Query: "textDocument/hover"
Returns: All hover-related implementations
```

## Architecture

### Components

1. **JavaCodeIndexer**: Core indexing engine using Tree-sitter
   - Parses Java files into AST
   - Extracts classes, methods, fields, and imports
   - Stores structured data in SQLite

2. **ServerConfig**: Configuration management
   - Environment variable handling
   - Default value management
   - Validation

3. **MCP Tools**: FastMCP-based tool implementations
   - Search functionality
   - Analytics and reporting
   - Repository management

### Database Schema

The server uses SQLite with the following main tables:
- `files`: File metadata and content
- `classes`: Class definitions and hierarchy
- `methods`: Method signatures and implementations
- `fields`: Field declarations
- `imports`: Import statements and dependencies

## Performance Considerations

### Indexing Performance
- Initial indexing time depends on repository size
- Incremental updates for changed files only
- Hash-based change detection

### Query Performance
- Indexed searches on names and content
- Relevance-based result ranking
- Configurable result limits
- Query caching for frequent searches

### Memory Usage
- SQLite provides efficient storage
- Tree-sitter uses minimal memory for parsing
- LRU cache for frequently accessed data

## Troubleshooting

### Common Issues

1. **"Repository not indexed" error**
   - Ensure `BALLERINA_REPO_PATH` is set correctly
   - Check that the path exists and contains Java files
   - Verify read permissions on the directory

2. **Slow indexing**
   - Large repositories may take several minutes to index initially
   - Consider excluding test directories if not needed
   - Monitor disk space for the SQLite database

3. **Missing search results**
   - Try re-indexing with `reindex_repository()`
   - Check if files have been recently modified
   - Verify file extensions are supported (.java)

### Logging
The server uses Python's standard logging module. Set log level:
```python
import logging
logging.getLogger("ballerina-mcp").setLevel(logging.DEBUG)
```

## Development

### Adding New Search Features
1. Extend the `JavaCodeIndexer` class with new methods
2. Add corresponding MCP tool functions
3. Update database schema if needed
4. Add tests for new functionality

### Extending Language Support
1. Add new Tree-sitter language parsers
2. Update `ServerConfig.supported_extensions`
3. Modify parsing logic in `JavaCodeIndexer`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[Add your license information here]

## Related Projects

- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [Tree-sitter](https://tree-sitter.github.io/) - Code parsing library
- [Ballerina Language Server](https://github.com/ballerina-platform/ballerina-lang) - Target language server
