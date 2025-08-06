# Ballerina Language Server MCP Server

An intelligent Model Context Protocol (MCP) server for querying and analyzing Java language server codebases, specifically designed for the Ballerina Language Server project.

## Features

### 🔍 **Advanced Code Search & Discovery**
- **Enhanced Fuzzy Search**: Multi-word matching with relevance scoring as the default search method
- **Similarity Search**: Find methods with similar names using phonetic matching
- **Class Information**: Detailed class analysis with methods, fields, and context
- **Method Hierarchy**: Explore method overrides and inheritance patterns

### 📊 **Repository Analytics & Insights**
- **Repository Statistics**: Comprehensive metrics (files, classes, methods, packages)
- **Package Analysis**: Distribution and organization insights
- **Code Complexity**: Method complexity analysis with control structure metrics
- **File Structure**: Repository organization and architecture overview
- **Dependency Analysis**: Class dependency mapping and usage patterns

### 🏗️ **LSP Protocol Specialized Tools**
- **Protocol Implementation Discovery**: Find specific LSP method implementations
- **Capability Analysis**: Analyze implemented LSP features and coverage
- **Protocol Handler Detection**: Identify message handlers and service providers
- **Language Server Architecture**: Understand LSP server structure and patterns

### 🎨 **Design Pattern Recognition**
- **Pattern Detection**: Identify common patterns (Factory, Observer, Singleton, etc.)
- **Architecture Analysis**: Understand design decisions and code organization
- **Configuration Discovery**: Find setup and configuration code
- **Error Handling**: Analyze exception patterns and error management

### ⚡ **Performance Optimized**
- **SQLite-based Indexing**: Fast queries with structured data storage
- **Regex-based Parsing**: Reliable Java code analysis with fallback strategies  
- **Output Length Management**: Pagination and limits to prevent overwhelming responses
- **Incremental Re-indexing**: Efficient updates for repository changes
- **Query Caching**: LRU cache for frequently accessed data

## Installation

### Prerequisites
- Claude Desktop
- Git (for cloning repositories)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Easier way to run Python scripts :)

### Setup
1. Clone this repository
2. Install dependencies: Using uv
```bash
uv sync
```
3. Update `claude_desktop_config.json` to include the MCP server path, to use this MCP server with Claude Desktop . It should be located at:
```
/Users/[your_username]/Library/Application Support/Claude/claude_desktop_config.json
```
4. Add the following entry in `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "ballerina-language-server": {
      "command": "uv",
      "args": [
        "--directory",
        "<PATH_TO_BALLERINA_LS_MCP_SERVER>",
        "run",
        "--active",
        "run.py"
      ],
      "env": {
        "BALLERINA_REPO_PATH": "<PATH_TO_BALLERINA_LANGUAGE_SERVER_REPO>"
      }
    }
  }
}
```
5. Restart Claude Desktop to apply the changes.

### Usage

After setting up, you can use the MCP server with Claude Desktop to query and analyze the Ballerina Language Server codebase.

### Available MCP Tools

#### Core Search & Analysis Tools

#### 1. `search_code(query: str, limit: int = 10)`
Enhanced fuzzy search through the codebase with relevance scoring and multi-word matching.

**Example:**
```
Query: "completion provider"
Returns: Ranked results for completion-related classes and methods
```

#### 2. `get_class_info(class_name: str)`
Get detailed information about a specific class including all methods and fields.

**Example:**
```
Query: "CompletionProvider"
Returns: Full class definition, methods, and context
```

#### 3. `get_repository_stats()`
Get comprehensive statistics about the indexed repository.

**Returns:**
- File count and distribution
- Top packages by class count
- Largest classes by line count
- Indexing health metrics

#### 4. `find_similar_methods(method_name: str, limit: int = 5)`
Find methods with similar names using phonetic matching algorithms.

**Example:**
```
Query: "getCompletion"
Returns: getCompletion, getCompletions, findCompletion, etc.
```

#### LSP Protocol Analysis Tools

#### 5. `find_lsp_protocol_implementations(protocol_method: str = "")`
Find implementations of specific LSP protocol methods or search for common LSP patterns.

**Example:**
```
Query: "textDocument/hover"
Returns: All hover-related implementations
```

#### 6. `analyze_lsp_capabilities(limit: int = 10)`
Analyze what LSP capabilities are implemented in the server.

**Returns:**
- List of implemented LSP features
- Corresponding implementation classes
- Coverage analysis

#### 7. `find_protocol_handlers(limit: int = 10)`
Find classes that handle LSP protocol messages (Handlers, Providers, Services, Managers).

#### Code Structure & Quality Tools

#### 8. `analyze_dependencies(class_name: str, limit: int = 15)`
Find what classes/methods a given class depends on and what depends on it.

**Example:**
```
Query: "DocumentSymbolProvider"
Returns: Classes that reference or use this provider
```

#### 9. `find_design_patterns(pattern_type: str = "", limit: int = 15)`
Identify common design patterns in the codebase.

**Supported Patterns:**
- `factory`: Factory, Creator, Builder patterns
- `observer`: Observer, Listener, Event, Handler patterns
- `singleton`: Singleton pattern implementations
- `adapter`: Adapter, Wrapper patterns
- `decorator`: Decorator patterns
- `visitor`: Visitor pattern implementations
- `strategy`: Strategy, Policy patterns
- `command`: Command, Action, Execute patterns

#### 10. `get_method_hierarchy(method_name: str)`
Find method overrides, implementations, and inheritance hierarchy.

#### 11. `analyze_configuration()`
Find configuration files, properties, and setup code.

#### 12. `get_file_structure_overview(limit: int = 15)`
Get an overview of the repository file structure and package organization.

#### 13. `analyze_code_complexity()`
Analyze code complexity metrics including method sizes and control structures.

#### 14. `find_error_handling_patterns()`
Find error handling patterns and exception usage throughout the codebase.

#### Repository Management Tools

#### 15. `reindex_repository()`
Re-index the repository to pick up new changes.

## Architecture

## Architecture

### Components

1. **JavaCodeIndexer**: Core indexing engine with hybrid parsing approach
   - **Primary**: Regex-based Java parsing for reliable code analysis
   - **Fallback**: Tree-sitter AST parsing for complex scenarios
   - Extracts classes, methods, fields, and imports with full metadata
   - Stores structured data in SQLite with proper indexing

2. **ServerConfig**: Configuration management system
   - Environment variable handling with validation
   - Default value management and type safety
   - Repository path and database configuration

3. **FastMCP Tools**: Comprehensive tool suite (15+ tools)
   - **Search Tools**: Enhanced fuzzy search with relevance scoring
   - **Analysis Tools**: Repository metrics, complexity analysis, dependency mapping
   - **LSP Tools**: Protocol discovery, capability analysis, handler detection
   - **Quality Tools**: Design pattern recognition, error handling analysis
   - **Management Tools**: Re-indexing and repository maintenance

4. **Output Management**: Smart response handling
   - Pagination for large result sets
   - Configurable limits to prevent overwhelming responses
   - Truncation indicators and summary statistics
   - Relevance-based result ranking

### Database Schema

The server uses SQLite with the following main tables:
- `files`: File metadata and content
- `classes`: Class definitions and hierarchy
- `methods`: Method signatures and implementations
- `fields`: Field declarations
- `imports`: Import statements and dependencies

## Performance Considerations

### Indexing Performance
- **Initial indexing time**: Depends on repository size (typically 30-60 seconds for large repos)
- **Incremental updates**: Only changed files are re-processed
- **Hash-based change detection**: Efficient file modification tracking
- **Database migration**: Automatic schema updates for compatibility

### Query Performance
- **Indexed searches**: Fast lookups on names, content, and metadata
- **Relevance-based ranking**: Multi-word fuzzy search with scoring
- **Configurable result limits**: Prevent overwhelming responses (default 10-15 results)
- **LRU query caching**: Frequently accessed data cached in memory
- **Pagination support**: Large result sets handled efficiently

### Memory Usage
- **SQLite storage**: Efficient disk-based indexing with minimal memory footprint
- **Regex parsing**: Lower memory usage compared to full AST parsing
- **Output truncation**: Long content automatically shortened with indicators
- **Streaming results**: Large queries processed incrementally

### Response Optimization
- **Output length management**: Automatic pagination and truncation
- **Smart summarization**: Key information highlighted in responses
- **Progress indicators**: Clear feedback on result completeness
- **Error boundaries**: Graceful handling of parsing and query errors

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

4. **Tool output too long**
   - All tools now have built-in pagination and output limits
   - Use `limit` parameters to control result count
   - Responses automatically truncate with clear indicators

5. **Parsing errors**
   - Server uses robust regex-based parsing as primary method
   - Automatic fallback handling for complex code structures
   - Database schema automatically migrates for compatibility

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

## Related Projects

- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [Tree-sitter](https://tree-sitter.github.io/) - Code parsing library
- [Ballerina Language Server](https://github.com/ballerina-platform/ballerina-lang) - Target language server
