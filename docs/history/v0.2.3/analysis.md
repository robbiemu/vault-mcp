# Analysis of Version v0.2.3 (Commit: `7a4ea70`)

### “maintenance drop”
- `asyncio.run()` is in; hand-rolled loops are out.
- global init now waits for CLI flags—no more “why is my config ignored?” moments.
- new markdown logs:    
  - `EXTERNAL_DEPENDENCY_WARNINGS.md` for noisy libs 
  - `future_changes.md` for brain dumps.
- embedding factory got a couple of micro-fixes.
- no shiny buttons, just a sturdier build.