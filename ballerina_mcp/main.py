#!/usr/bin/env python3
"""
Ballerina Language Server MCP Server using FastMCP
An MCP server that provides intelligent querying of Java language server code
"""

import os
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
import time

# FastMCP and Tree-sitter imports
try:
    from fastmcp import FastMCP
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser
except ImportError:
    print("Please install FastMCP and tree-sitter: pip install fastmcp tree-sitter tree-sitter-java")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ballerina-mcp")

@dataclass
class ServerConfig:
    """Configuration for the MCP server"""
    repo_path: str
    db_path: str = "ballerina_index.db"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    supported_extensions: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['.java']
    
    @classmethod
    def from_env(cls) -> 'ServerConfig':
        """Create configuration from environment variables"""
        repo_path = os.getenv("BALLERINA_REPO_PATH")
        if not repo_path:
            raise ValueError("BALLERINA_REPO_PATH environment variable is required")
        
        return cls(
            repo_path=repo_path,
            db_path=os.getenv("BALLERINA_DB_PATH", "ballerina_index.db"),
            max_file_size=int(os.getenv("BALLERINA_MAX_FILE_SIZE", "10485760")),  # 10MB
        )

class JavaCodeIndexer:
    """Indexes Java code using Tree-sitter for proper AST parsing"""
    
    def __init__(self, repo_path: str, db_path: str = "ballerina_index.db"):
        self.repo_path = Path(repo_path)
        self.db_path = db_path
        
        # Initialize Tree-sitter
        self.java_language = Language(tsjava.language())
        self.parser = Parser(self.java_language)
        
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for code indexing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                content TEXT,
                hash TEXT,
                last_modified REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY,
                file_id INTEGER,
                name TEXT,
                package TEXT,
                content TEXT,
                line_start INTEGER,
                line_end INTEGER,
                FOREIGN KEY (file_id) REFERENCES files (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS methods (
                id INTEGER PRIMARY KEY,
                class_id INTEGER,
                name TEXT,
                signature TEXT,
                return_type TEXT,
                parameters TEXT,
                content TEXT,
                line_start INTEGER,
                line_end INTEGER,
                visibility TEXT,
                is_static BOOLEAN,
                FOREIGN KEY (class_id) REFERENCES classes (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fields (
                id INTEGER PRIMARY KEY,
                class_id INTEGER,
                name TEXT,
                type TEXT,
                visibility TEXT,
                is_static BOOLEAN,
                line_number INTEGER,
                FOREIGN KEY (class_id) REFERENCES classes (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS imports (
                id INTEGER PRIMARY KEY,
                file_id INTEGER,
                import_path TEXT,
                is_static BOOLEAN,
                FOREIGN KEY (file_id) REFERENCES files (id)
            )
        ''')
        
        # Create indexes for faster searching
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_classes_name ON classes(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_methods_name ON methods(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fields_name ON fields(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_imports_path ON imports(import_path)')
        
        conn.commit()
        conn.close()
    
    def get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content for change detection"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return ""
    
    def parse_java_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a Java file using Tree-sitter and extract classes, methods, and fields"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return {"content": "", "package": "", "imports": [], "classes": []}
        
        # Parse with Tree-sitter
        tree = self.parser.parse(content.encode())
        root_node = tree.root_node
        
        # Extract package declaration
        package = self.extract_package(root_node, content)
        
        # Extract imports
        imports = self.extract_imports(root_node, content)
        
        # Extract classes and interfaces
        classes = self.extract_classes(root_node, content)
        
        return {
            "content": content,
            "package": package,
            "imports": imports,
            "classes": classes
        }
    
    def extract_package(self, root_node, content: str) -> str:
        """Extract package declaration using Tree-sitter"""
        query = self.java_language.query("""
            (package_declaration 
                (scoped_identifier) @package)
        """)
        
        captures = query.captures(root_node) # type: ignore
        for node, _ in captures:
            return content[node.start_byte:node.end_byte]
        return ""
    
    def extract_imports(self, root_node, content: str) -> List[Dict[str, Any]]:
        """Extract import statements using Tree-sitter"""
        query = self.java_language.query("""
            (import_declaration 
                (scoped_identifier) @import_path) @import
            (import_declaration 
                "static"
                (scoped_identifier) @static_import_path) @static_import
        """)
        
        imports = []
        captures = query.captures(root_node) # type: ignore
        for node, capture_name in captures:
            if capture_name in ["import_path", "static_import_path"]:
                import_path = content[node.start_byte:node.end_byte]
                is_static = capture_name == "static_import_path"
                imports.append({
                    "import_path": import_path,
                    "is_static": is_static
                })
        
        return imports
    
    def extract_classes(self, root_node, content: str) -> List[Dict[str, Any]]:
        """Extract classes and interfaces using Tree-sitter"""
        query = self.java_language.query("""
            (class_declaration 
                name: (identifier) @class_name) @class_body
            (interface_declaration 
                name: (identifier) @interface_name) @interface_body
            (enum_declaration 
                name: (identifier) @enum_name) @enum_body
        """)
        
        classes = []
        captures = query.captures(root_node) # type: ignore
        
        # Group captures by class/interface
        class_nodes = {}
        for node, capture_name in captures:
            if capture_name.endswith("_body"):
                class_type = capture_name.replace("_body", "")
                class_nodes[node] = {"type": class_type, "node": node}
            elif capture_name.endswith("_name"):
                # Find the parent class/interface node
                parent = node.parent
                while parent and parent not in class_nodes:
                    parent = parent.parent
                if parent:
                    class_nodes[parent]["name"] = content[node.start_byte:node.end_byte]
        
        for class_data in class_nodes.values():
            if "name" not in class_data:
                continue
                
            node = class_data["node"]
            class_name = class_data["name"]
            class_type = class_data["type"]
            
            # Get class content
            class_content = content[node.start_byte:node.end_byte]
            
            # Get line numbers
            lines_before = content[:node.start_byte].count('\n')
            lines_in_class = class_content.count('\n')
            
            # Extract methods and fields from this class
            methods = self.extract_methods(node, content)
            fields = self.extract_fields(node, content)
            
            classes.append({
                "name": class_name,
                "type": class_type,
                "content": class_content,
                "line_start": lines_before + 1,
                "line_end": lines_before + lines_in_class + 1,
                "methods": methods,
                "fields": fields
            })
        
        return classes
    
    def extract_methods(self, class_node, content: str) -> List[Dict[str, Any]]:
        """Extract methods from a class node using Tree-sitter"""
        query = self.java_language.query("""
            (method_declaration
                (modifiers)? @modifiers
                type: (_) @return_type
                name: (identifier) @method_name
                parameters: (formal_parameters) @parameters) @method_body
            (constructor_declaration
                (modifiers)? @constructor_modifiers
                name: (identifier) @constructor_name
                parameters: (formal_parameters) @constructor_parameters) @constructor_body
        """)
        
        methods = []
        captures = query.captures(class_node) # type: ignore
        
        # Group captures by method
        method_nodes = {}
        for node, capture_name in captures:
            if capture_name.endswith("_body"):
                method_type = "constructor" if "constructor" in capture_name else "method"
                method_nodes[node] = {"type": method_type, "node": node}
            else:
                # Find the parent method node
                parent = node.parent
                while parent and parent not in method_nodes:
                    parent = parent.parent
                if parent:
                    if capture_name.endswith("_name"):
                        method_nodes[parent]["name"] = content[node.start_byte:node.end_byte]
                    elif "return_type" in capture_name:
                        method_nodes[parent]["return_type"] = content[node.start_byte:node.end_byte]
                    elif "parameters" in capture_name:
                        method_nodes[parent]["parameters"] = content[node.start_byte:node.end_byte]
                    elif "modifiers" in capture_name:
                        method_nodes[parent]["modifiers"] = content[node.start_byte:node.end_byte]
        
        for method_data in method_nodes.values():
            if "name" not in method_data:
                continue
                
            node = method_data["node"]
            method_name = method_data["name"]
            return_type = method_data.get("return_type", "void" if method_data["type"] == "constructor" else "")
            parameters = method_data.get("parameters", "()")
            modifiers = method_data.get("modifiers", "")
            
            # Build signature
            signature = f"{modifiers} {return_type} {method_name}{parameters}".strip()
            
            # Get method content
            method_content = content[node.start_byte:node.end_byte]
            
            # Get line numbers
            lines_before = content[:node.start_byte].count('\n')
            lines_in_method = method_content.count('\n')
            
            # Parse visibility and static
            visibility = "package"
            is_static = False
            if "public" in modifiers:
                visibility = "public"
            elif "private" in modifiers:
                visibility = "private"
            elif "protected" in modifiers:
                visibility = "protected"
            
            if "static" in modifiers:
                is_static = True
            
            methods.append({
                "name": method_name,
                "signature": signature,
                "return_type": return_type,
                "parameters": parameters,
                "content": method_content,
                "line_start": lines_before + 1,
                "line_end": lines_before + lines_in_method + 1,
                "visibility": visibility,
                "is_static": is_static
            })
        
        return methods
    
    def extract_fields(self, class_node, content: str) -> List[Dict[str, Any]]:
        """Extract fields from a class node using Tree-sitter"""
        query = self.java_language.query("""
            (field_declaration
                (modifiers)? @modifiers
                type: (_) @field_type
                declarator: (variable_declarator
                    name: (identifier) @field_name))
        """)
        
        fields = []
        captures = query.captures(class_node) # type: ignore
        
        # Group captures by field
        field_data = {}
        for node, capture_name in captures:
            if capture_name == "field_name":
                field_name = content[node.start_byte:node.end_byte]
                line_num = content[:node.start_byte].count('\n') + 1
                field_data[node] = {"name": field_name, "line": line_num}
            elif capture_name == "field_type":
                # Find associated field name
                for name_node, data in field_data.items():
                    if abs(node.start_byte - name_node.start_byte) < 100:  # Rough proximity
                        data["type"] = content[node.start_byte:node.end_byte]
                        break
            elif capture_name == "modifiers":
                # Find associated field name
                for name_node, data in field_data.items():
                    if abs(node.start_byte - name_node.start_byte) < 100:  # Rough proximity
                        modifiers = content[node.start_byte:node.end_byte]
                        data["modifiers"] = modifiers
                        
                        # Parse visibility and static
                        if "public" in modifiers:
                            data["visibility"] = "public"
                        elif "private" in modifiers:
                            data["visibility"] = "private"
                        elif "protected" in modifiers:
                            data["visibility"] = "protected"
                        else:
                            data["visibility"] = "package"
                        
                        data["is_static"] = "static" in modifiers
                        break
        
        for data in field_data.values():
            fields.append({
                "name": data["name"],
                "type": data.get("type", "unknown"),
                "visibility": data.get("visibility", "package"),
                "is_static": data.get("is_static", False),
                "line_number": data["line"]
            })
        
        return fields
    
    def index_repository(self):
        """Index all Java files in the repository"""
        logger.info(f"Indexing repository: {self.repo_path}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        java_files = list(self.repo_path.rglob("*.java"))
        logger.info(f"Found {len(java_files)} Java files")
        
        for file_path in java_files:
            try:
                relative_path = str(file_path.relative_to(self.repo_path))
                file_hash = self.get_file_hash(file_path)
                last_modified = file_path.stat().st_mtime
                
                # Check if file needs updating
                cursor.execute('SELECT hash FROM files WHERE path = ?', (relative_path,))
                result = cursor.fetchone()
                
                if result and result[0] == file_hash:
                    continue  # File hasn't changed
                
                # Parse file
                parsed_data = self.parse_java_file(file_path)
                
                # Insert/update file
                cursor.execute('''
                    INSERT OR REPLACE INTO files (path, content, hash, last_modified)
                    VALUES (?, ?, ?, ?)
                ''', (relative_path, parsed_data["content"], file_hash, last_modified))
                
                file_id = cursor.lastrowid
                
                # Remove old data
                cursor.execute('DELETE FROM methods WHERE class_id IN (SELECT id FROM classes WHERE file_id = ?)', (file_id,))
                cursor.execute('DELETE FROM fields WHERE class_id IN (SELECT id FROM classes WHERE file_id = ?)', (file_id,))
                cursor.execute('DELETE FROM classes WHERE file_id = ?', (file_id,))
                cursor.execute('DELETE FROM imports WHERE file_id = ?', (file_id,))
                
                # Insert imports
                for import_data in parsed_data["imports"]:
                    cursor.execute('''
                        INSERT INTO imports (file_id, import_path, is_static)
                        VALUES (?, ?, ?)
                    ''', (file_id, import_data["import_path"], import_data["is_static"]))
                
                # Insert classes, methods, and fields
                for class_data in parsed_data["classes"]:
                    cursor.execute('''
                        INSERT INTO classes (file_id, name, package, content, line_start, line_end)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (file_id, class_data["name"], parsed_data["package"], 
                          class_data["content"], class_data["line_start"], class_data["line_end"]))
                    
                    class_id = cursor.lastrowid
                    
                    # Insert methods
                    for method_data in class_data["methods"]:
                        cursor.execute('''
                            INSERT INTO methods (class_id, name, signature, return_type, parameters, content, line_start, line_end, visibility, is_static)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (class_id, method_data["name"], method_data["signature"], method_data["return_type"],
                              method_data["parameters"], method_data["content"], method_data["line_start"], 
                              method_data["line_end"], method_data["visibility"], method_data["is_static"]))
                    
                    # Insert fields
                    for field_data in class_data["fields"]:
                        cursor.execute('''
                            INSERT INTO fields (class_id, name, type, visibility, is_static, line_number)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (class_id, field_data["name"], field_data["type"], field_data["visibility"],
                              field_data["is_static"], field_data["line_number"]))
                
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
        
        conn.commit()
        conn.close()
        logger.info("Indexing complete")
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for code elements matching the query"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            results = []
            
            # Sanitize query to prevent SQL injection issues
            safe_query = query.replace("'", "''").replace("%", "\\%").replace("_", "\\_")
            
            # Search in class names and content
            cursor.execute('''
                SELECT c.name as class_name, c.package, c.content, f.path, c.line_start, c.line_end
                FROM classes c
                JOIN files f ON c.file_id = f.id
                WHERE c.name LIKE ? ESCAPE '\\' OR c.content LIKE ? ESCAPE '\\'
                ORDER BY 
                    CASE WHEN c.name LIKE ? ESCAPE '\\' THEN 1 ELSE 2 END,
                    c.name
                LIMIT ?
            ''', (f'%{safe_query}%', f'%{safe_query}%', f'%{safe_query}%', limit))
            
            for row in cursor.fetchall():
                results.append({
                    "type": "class",
                    "name": row[0] or "Unknown",
                    "package": row[1] or "",
                    "content": (row[2][:500] + "...") if row[2] and len(row[2]) > 500 else (row[2] or ""),
                    "file": row[3] or "",
                    "line_start": row[4] or 0,
                    "line_end": row[5] or 0
                })
            
            # Search in method names and content
            remaining_limit = max(0, limit - len(results))
            if remaining_limit > 0:
                cursor.execute('''
                    SELECT m.name as method_name, m.signature, m.content, c.name as class_name, 
                           c.package, f.path, m.line_start, m.line_end
                    FROM methods m
                    JOIN classes c ON m.class_id = c.id
                    JOIN files f ON c.file_id = f.id
                    WHERE m.name LIKE ? ESCAPE '\\' OR m.content LIKE ? ESCAPE '\\'
                    ORDER BY 
                        CASE WHEN m.name LIKE ? ESCAPE '\\' THEN 1 ELSE 2 END,
                        m.name
                    LIMIT ?
                ''', (f'%{safe_query}%', f'%{safe_query}%', f'%{safe_query}%', remaining_limit))
                
                for row in cursor.fetchall():
                    results.append({
                        "type": "method",
                        "name": row[0] or "Unknown",
                        "signature": row[1] or "",
                        "content": (row[2][:300] + "...") if row[2] and len(row[2]) > 300 else (row[2] or ""),
                        "class": row[3] or "Unknown",
                        "package": row[4] or "",
                        "file": row[5] or "",
                        "line_start": row[6] or 0,
                        "line_end": row[7] or 0
                    })
            
            return results
        except sqlite3.Error as e:
            logger.error(f"Database error during search: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return []
        finally:
            try:
                if conn is not None:
                    conn.close()
            except:
                pass
    
    @lru_cache(maxsize=100)
    def search_cached(self, query: str, limit: int = 10) -> str:
        """Cached version of search for frequently accessed queries"""
        results = self.search(query, limit)
        return json.dumps(results)
    
    def fuzzy_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Enhanced search with fuzzy matching capabilities"""
        conn = None
        results: List[Dict[str, Any]] = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            query_words = query.lower().split()
            
            # Create a more sophisticated search query
            search_conditions = []
            params = []
            
            for word in query_words:
                search_conditions.append("(LOWER(c.name) LIKE ? OR LOWER(c.content) LIKE ?)")
                params.extend([f'%{word}%', f'%{word}%'])
            
            if search_conditions:
                where_clause = " AND ".join(search_conditions)
                
                # Search classes with relevance scoring
                cursor.execute(f'''
                    SELECT c.name, c.package, c.content, f.path, c.line_start, c.line_end,
                           (CASE WHEN LOWER(c.name) LIKE ? THEN 10 ELSE 0 END +
                            CASE WHEN LOWER(c.name) LIKE ? THEN 5 ELSE 0 END) as relevance
                    FROM classes c
                    JOIN files f ON c.file_id = f.id
                    WHERE {where_clause}
                    ORDER BY relevance DESC, c.name
                    LIMIT ?
                ''', [f'%{query.lower()}%', f'%{query_words[0]}%'] + params + [limit])
                
                for row in cursor.fetchall():
                    results.append({
                        "type": "class",
                        "name": row[0] or "Unknown",
                        "package": row[1] or "",
                        "content": (row[2][:500] + "...") if row[2] and len(row[2]) > 500 else (row[2] or ""),
                        "file": row[3] or "",
                        "line_start": row[4] or 0,
                        "line_end": row[5] or 0,
                        "relevance": row[6] if len(row) > 6 else 0
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error in fuzzy search: {e}")
            return results
        finally:
            try:
                if conn is not None:
                    conn.close()
            except:
                pass

# Initialize FastMCP
mcp = FastMCP("Ballerina Language Server MCP")

# Global indexer instance
indexer = None

def initialize_indexer():
    """Initialize the code indexer"""
    global indexer
    
    try:
        config = ServerConfig.from_env()
        logger.info(f"Initializing indexer with config: repo_path={config.repo_path}, db_path={config.db_path}")
        
        if not Path(config.repo_path).exists():
            logger.error(f"Repository path does not exist: {config.repo_path}")
            return False
        
        indexer = JavaCodeIndexer(config.repo_path, config.db_path)
        
        # Initial indexing
        logger.info("Performing initial indexing...")
        start_time = time.time()
        indexer.index_repository()
        end_time = time.time()
        logger.info(f"Initial indexing complete in {end_time - start_time:.2f} seconds")
        return True
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        return False

@mcp.tool()
def search_code(query: str, limit: int = 10) -> str:
    """Search through the Ballerina language server Java codebase"""
    if indexer is None:
        return "Error: Repository not indexed. Please set BALLERINA_REPO_PATH environment variable."
    
    results = indexer.search(query, limit)
    
    if not results:
        return f"No results found for query: '{query}'"
    
    response = f"Found {len(results)} results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        response += f"{i}. **{result['type'].title()}: {result['name']}**\n"
        response += f"   File: {result['file']} (lines {result['line_start']}-{result['line_end']})\n"
        
        if result['type'] == 'method':
            response += f"   Class: {result['class']}\n"
            response += f"   Signature: {result['signature']}\n"
        
        if result.get('package'):
            response += f"   Package: {result['package']}\n"
        
        response += f"   Content preview:\n```java\n{result['content']}\n```\n\n"
    
    return response

@mcp.tool()
def advanced_search(query: str, search_type: str = "fuzzy", limit: int = 10) -> str:
    """Advanced search with different search strategies (fuzzy, exact, regex)"""
    if indexer is None:
        return "Error: Repository not indexed."
    
    if search_type == "fuzzy":
        results = indexer.fuzzy_search(query, limit)
    elif search_type == "exact":
        results = indexer.search(query, limit)
    else:
        return f"Unsupported search type: {search_type}. Use 'fuzzy' or 'exact'."
    
    if not results:
        return f"No results found for {search_type} search: '{query}'"
    
    response = f"Advanced {search_type} search results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        response += f"{i}. **{result['type'].title()}: {result['name']}**\n"
        response += f"   File: {result['file']} (lines {result['line_start']}-{result['line_end']})\n"
        
        if result.get('relevance'):
            response += f"   Relevance Score: {result['relevance']}\n"
        
        if result['type'] == 'method':
            response += f"   Class: {result.get('class', 'Unknown')}\n"
            response += f"   Signature: {result.get('signature', 'N/A')}\n"
        
        if result.get('package'):
            response += f"   Package: {result['package']}\n"
        
        response += f"   Content preview:\n```java\n{result['content']}\n```\n\n"
    
    return response

@mcp.tool()
def get_repository_stats() -> str:
    """Get statistics about the indexed repository"""
    if indexer is None:
        return "Error: Repository not indexed."
    
    try:
        conn = sqlite3.connect(indexer.db_path)
        cursor = conn.cursor()
        
        # Get file count
        cursor.execute("SELECT COUNT(*) FROM files")
        file_count = cursor.fetchone()[0]
        
        # Get class count
        cursor.execute("SELECT COUNT(*) FROM classes")
        class_count = cursor.fetchone()[0]
        
        # Get method count
        cursor.execute("SELECT COUNT(*) FROM methods")
        method_count = cursor.fetchone()[0]
        
        # Get package distribution
        cursor.execute("""
            SELECT package, COUNT(*) as class_count 
            FROM classes 
            WHERE package != '' 
            GROUP BY package 
            ORDER BY class_count DESC 
            LIMIT 10
        """)
        top_packages = cursor.fetchall()
        
        # Get largest classes
        cursor.execute("""
            SELECT name, package, (line_end - line_start + 1) as lines
            FROM classes
            ORDER BY lines DESC
            LIMIT 5
        """)
        largest_classes = cursor.fetchall()
        
        conn.close()
        
        response = "**Repository Statistics**\n\n"
        response += f"📁 **Files indexed:** {file_count}\n"
        response += f"🏗️ **Classes:** {class_count}\n"
        response += f"⚙️ **Methods:** {method_count}\n\n"
        
        response += "**Top Packages by Class Count:**\n"
        for package, count in top_packages:
            response += f"- {package}: {count} classes\n"
        
        response += "\n**Largest Classes by Lines:**\n"
        for name, package, lines in largest_classes:
            response += f"- {name} ({package}): {lines} lines\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting repository stats: {e}")
        return f"Error retrieving repository statistics: {str(e)}"

@mcp.tool()
def find_similar_methods(method_name: str, limit: int = 5) -> str:
    """Find methods with similar names or signatures"""
    if indexer is None:
        return "Error: Repository not indexed."
    
    try:
        conn = sqlite3.connect(indexer.db_path)
        cursor = conn.cursor()
        
        # Find methods with similar names
        cursor.execute("""
            SELECT m.name, m.signature, c.name as class_name, c.package, f.path, m.line_start
            FROM methods m
            JOIN classes c ON m.class_id = c.id
            JOIN files f ON c.file_id = f.id
            WHERE m.name LIKE ? OR SOUNDEX(m.name) = SOUNDEX(?)
            ORDER BY 
                CASE WHEN m.name = ? THEN 1
                     WHEN m.name LIKE ? THEN 2
                     ELSE 3 END,
                m.name
            LIMIT ?
        """, (f'%{method_name}%', method_name, method_name, f'{method_name}%', limit))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return f"No similar methods found for: '{method_name}'"
        
        response = f"**Similar methods to '{method_name}':**\n\n"
        
        for i, (name, signature, class_name, package, file_path, line_start) in enumerate(results, 1):
            response += f"{i}. **{name}**\n"
            response += f"   Class: {class_name} ({package})\n"
            response += f"   File: {file_path} (line {line_start})\n"
            response += f"   Signature: `{signature}`\n\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error finding similar methods: {e}")
        return f"Error finding similar methods: {str(e)}"

@mcp.tool()
def get_class_info(class_name: str) -> str:
    """Get detailed information about a specific class"""
    if indexer is None:
        return "Error: Repository not indexed."
    
    results = indexer.search(class_name, limit=5)
    class_results = [r for r in results if r['type'] == 'class' and r['name'] == class_name]
    
    if not class_results:
        return f"Class '{class_name}' not found."
    
    class_info = class_results[0]
    response = f"**Class: {class_info['name']}**\n\n"
    response += f"Package: {class_info['package']}\n"
    response += f"File: {class_info['file']} (lines {class_info['line_start']}-{class_info['line_end']})\n\n"
    response += f"```java\n{class_info['content']}\n```"
    
    return response

@mcp.tool()
def reindex_repository() -> str:
    """Re-index the language server repository to pick up new changes"""
    if indexer is None:
        return "Error: Repository not indexed."
    
    indexer.index_repository()
    return "Repository re-indexed successfully."

@mcp.tool()
def find_lsp_protocol_implementations(protocol_method: str = "") -> str:
    """Find LSP protocol method implementations in the language server"""
    if indexer is None:
        return "Error: Repository not indexed."
    
    # Common LSP protocol methods to search for
    lsp_methods = [
        "textDocument/completion",
        "textDocument/hover",
        "textDocument/signatureHelp",
        "textDocument/definition",
        "textDocument/references",
        "textDocument/documentHighlight",
        "textDocument/documentSymbol",
        "textDocument/codeAction",
        "textDocument/codeLens",
        "textDocument/formatting",
        "textDocument/rangeFormatting",
        "textDocument/rename",
        "workspace/symbol"
    ]
    
    if protocol_method:
        # Search for specific protocol method
        search_terms = [protocol_method, protocol_method.replace("/", ""), protocol_method.split("/")[-1]]
        results = []
        for term in search_terms:
            results.extend(indexer.search(term, limit=5))
    else:
        # Search for common LSP implementations
        results = []
        for method in lsp_methods[:5]:  # Limit to first 5 to avoid too many results
            method_name = method.split("/")[-1]
            results.extend(indexer.search(method_name, limit=2))
    
    if not results:
        return f"No LSP protocol implementations found for: {protocol_method or 'common methods'}"
    
    response = f"LSP Protocol Implementations found:\n\n"
    
    seen = set()
    for result in results:
        key = f"{result['name']}-{result['file']}"
        if key in seen:
            continue
        seen.add(key)
        
        response += f"**{result['type'].title()}: {result['name']}**\n"
        response += f"File: {result['file']}\n"
        if result.get('class'):
            response += f"Class: {result['class']}\n"
        response += f"```java\n{result['content'][:200]}...\n```\n\n"
    
    return response

def main():
    """Main entry point for the MCP server"""
    # Initialize indexer before starting server
    if initialize_indexer():
        mcp.run()
    else:
        exit(1)

if __name__ == "__main__":
    main()
