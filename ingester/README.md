# Ingester

## Seed Milvus DB using datasets

```bash
source venv/bin/activate # if not exists create one by `uv venv`
cp .env.example .env # fill in the .env file with your settings
uv sync
uv run main.py
```
