# Analysis of Version v0.2.2 (Commit: `7aebf09`)

Tiny but welcome update.  

Two new toggles in `config.toml`:
- enable_quality_filter – turn off the chunk heuristics if you want everything indexed.
- custom ChromaDB path – handy for Docker or multiple vaults.

Startup now happens inside a FastAPI lifespan manager, so you’ll get `503` instead of a crash if the DB isn’t ready.

Prompt for the ChunkRewriteAgent got a glow-up too.

Still no actual rewriting happening, but at least the instructions are prettier.