# Knowledge Server

A Model Context Protocol (MCP) server that provides AI agents with access to a knowledge base built from PDF and Markdown documents using RAG (Retrieval-Augmented Generation).

## Features

- **Document Loading**: Supports PDF and Markdown files from local directories, Lark Docs, Lark Wikis, and Lark Wiki Spaces
- **Vector Storage**: Uses Milvus for efficient vector similarity search with full-text search support
- **Embeddings**: Configurable embeddings via Ollama
- **Text Chunking**: Recursive character text splitting with configurable chunk size and overlap
- **MCP Integration**: Exposes knowledge base queries through FastMCP server
- **Lark Integration**: Direct integration with Lark Suite for loading documents, wikis, and entire wiki spaces
- **Flexible Configuration**: YAML-based configuration for easy customization

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌────────────┐
│  Datasource │─────▶│    Loader    │─────▶│  Splitter  │
│   (YAML)    │      │  Directory   │      │            │
│             │      │  Lark Doc    │      │            │
│             │      │  Lark Wiki   │      │            │
│             │      │  Lark Space  │      │            │
└─────────────┘      └──────────────┘      └────────────┘
                                                   │
                                                   ▼
┌─────────────┐      ┌──────────────┐      ┌────────────┐
│  MCP Client │◀─────│  MCP Server  │◀─────│   Milvus   │
│  (AI Agent) │      │  (FastMCP)   │      │  (Vector)  │
└─────────────┘      └──────────────┘      └────────────┘
```

## Installation

1. Ensure Python 3.12+ is installed
2. Install dependencies using uv:
   ```bash
   uv sync
   ```

## Configuration

### 1. Create Configuration File

Copy the example configuration:
```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml`:
```yaml
log_level: DEBUG
vector_store:
  type: milvus
  url: "http://localhost:19530"
  collection_name: knowledge_base
  reset_collection: true
  enable_full_text_search: true
chunk_size: 1000
chunk_overlap: 200
embeddings:
  source: ollama
  model: embeddinggemma:latest
lark:
  domain: "https://open.larksuite.com"
  app_id: "your_app_id"
  app_secret: "your_app_secret"
```

**Configuration Options:**
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR) - applies to both application and Lark client
- `vector_store`: Milvus configuration
- `chunk_size`: Size of text chunks for splitting
- `chunk_overlap`: Overlap between chunks
- `embeddings`: Ollama embeddings configuration
- `lark`: Lark Suite API credentials (required only if using Lark datasources)

### 2. Configure Data Sources

Create `datasource.yaml` with one or more data sources:

**Local Directory (PDF and Markdown files):**
```yaml
datasource:
  - type: directory
    path: ../datasets/
```

**Lark Document:**
```yaml
datasource:
  - type: lark-doc
    id: "doc-id"
```

**Lark Wiki:**
```yaml
datasource:
  - type: lark-wiki
    id: "wiki-id"
```

**Lark Wiki Space (loads all documents in a space):**
```yaml
datasource:
  - type: lark-space
    id: "space-id"
```

**Multiple Sources:**
```yaml
datasource:
  - type: directory
    path: ../datasets/
  - type: lark-doc
    id: "doc-id"
  - type: lark-wiki
    id: "wiki-id"
  - type: lark-space
    id: "space-id"
```

**Supported Datasource Types:**
- `directory`: Load PDF and Markdown files from a local directory
- `lark-doc`: Load a single Lark document by ID
- `lark-wiki`: Load a single wiki page by ID
- `lark-space`: Load all documents from a Lark wiki space by space ID (recursively loads all child pages)

### 3. Start Milvus

Using Docker Compose:
```bash
docker-compose up -d
```

This will start Milvus on `http://localhost:19530`.

## Usage

### Running the Server

Using Python directly:
```bash
uv run python main.py
```

Or using the Makefile:
```bash
make run
```

The server will:
1. Load documents from configured datasources
2. Split documents into chunks
3. Generate embeddings using Ollama
4. Store vectors in Milvus
5. Start the MCP server on streamable-http transport

### Querying the Knowledge Base

The server exposes an MCP tool `query_knowledge_base`:

```python
query_knowledge_base(
    query: str,      # Search query
    top_k: int = 4   # Number of results to return
) -> list[str]
```

## Project Structure

```
.
├── config/
│   └── config.py           # Configuration loader
├── loader/
│   ├── factory.py          # Loader factory and datasource abstraction
│   ├── directory.py        # Directory loader (PDF/MD)
│   └── lark.py             # Lark Suite loaders (Doc/Wiki/Space)
├── model/
│   ├── factory.py          # Embeddings factory
│   └── model_garden.py     # Model configurations
├── vector_store/
│   └── milvus.py          # Milvus vector store implementation
├── main.py                 # Application entry point
├── config.yaml             # Runtime configuration
├── datasource.yaml         # Data source definitions
├── Makefile               # Development tasks
└── pyproject.toml         # Project dependencies
```

## Dependencies

- **langchain-community**: Document loaders and utilities
- **langchain-ollama**: Ollama embeddings integration
- **mcp**: Model Context Protocol server
- **pymilvus**: Milvus vector database client
- **pypdf**: PDF parsing
- **pyyaml**: YAML configuration parsing
- **lark-oapi**: Lark Suite Open API SDK

## Development

### Code Style

Run all checks:
```bash
make check
```

Or run individual tasks:
```bash
make lint      # Run ruff check --fix
make format    # Run ruff format
make type-check # Run ty check
```

Format code using Ruff:
```bash
uv run ruff format .
uv run ruff check --fix
```

### Type Checking

Type checking is configured with `ty` (ignored rules in `pyproject.toml`).

## Troubleshooting

### Import Errors

If you encounter `ImportError: cannot import name 'Blob'`, ensure you're using the correct import:
```python
from langchain_community.document_loaders.blob_loaders import Blob
```

### Lark API Issues

**Authentication Errors:**
- Verify `app_id` and `app_secret` in `config.yaml`
- Ensure your Lark app has the required permissions:
  - `docx:document` for document access
  - `wiki:wiki` for wiki access

**Document Not Found:**
- Verify the document/wiki ID is correct
- Check that your app has access to the document/wiki
- Ensure the document/wiki hasn't been deleted

**Getting Document IDs:**
- For Lark Docs: The ID is in the URL: `https://xxx.larksuite.com/docx/{document_id}`
- For Lark Wikis: The ID is in the URL: `https://xxx.larksuite.com/wiki/{wiki_id}`
- For Lark Spaces: The space ID can be found in wiki space settings or via the Lark API

### Milvus Connection Issues

Verify Milvus is running:
```bash
docker-compose ps
```

Check Milvus logs:
```bash
docker-compose logs milvus-standalone
```
