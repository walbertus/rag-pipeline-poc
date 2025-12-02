# Knowledge Server

A Model Context Protocol (MCP) server that provides AI agents with access to a knowledge base built from PDF and Markdown documents using RAG (Retrieval-Augmented Generation).

## Features

- **Document Loading**: Supports PDF and Markdown files from local directories
- **Vector Storage**: Uses Milvus for efficient vector similarity search with full-text search support
- **Embeddings**: Configurable embeddings via Ollama
- **Text Chunking**: Recursive character text splitting with configurable chunk size and overlap
- **MCP Integration**: Exposes knowledge base queries through FastMCP server
- **Flexible Configuration**: YAML-based configuration for easy customization

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌────────────┐
│  Datasource │─────▶│    Loader    │─────▶│  Splitter  │
│   (YAML)    │      │ (PDF/MD)     │      │            │
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
  provider: ollama
  model: nomic-embed-text
```

### 2. Configure Data Sources

Create `datasource.yaml`:
```yaml
datasource:
  - type: directory
    path: ../datasets/
```

### 3. Start Milvus

Using Docker Compose:
```bash
docker-compose up -d
```

This will start Milvus on `http://localhost:19530`.

## Usage

### Running the Server

```bash
uv run python main.py
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
│   ├── datasource.py       # Datasource abstraction
│   └── directory.py        # Directory loader (PDF/MD)
├── model/
│   ├── factory.py          # Embeddings factory
│   └── model_garden.py     # Model configurations
├── vector_store/
│   └── milvus.py          # Milvus vector store implementation
├── main.py                 # Application entry point
├── config.yaml             # Runtime configuration
├── datasource.yaml         # Data source definitions
└── pyproject.toml         # Project dependencies
```

## Dependencies

- **langchain-community**: Document loaders and utilities
- **langchain-ollama**: Ollama embeddings integration
- **mcp**: Model Context Protocol server
- **pymilvus**: Milvus vector database client
- **pypdf**: PDF parsing
- **pyyaml**: YAML configuration parsing

## Development

### Code Style

Format code using Ruff:
```bash
uv run ruff format .
uv run ruff check .
```

### Type Checking

Type checking is configured with `ty` (ignored rules in `pyproject.toml`).

## Troubleshooting

### Import Errors

If you encounter `ImportError: cannot import name 'Blob'`, ensure you're using the correct import:
```python
from langchain_community.document_loaders.blob_loaders import Blob
```

### Milvus Connection Issues

Verify Milvus is running:
```bash
docker-compose ps
```

Check Milvus logs:
```bash
docker-compose logs milvus-standalone
```
