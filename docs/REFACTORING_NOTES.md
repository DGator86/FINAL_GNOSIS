# Main.py Refactoring Notes

## Problem
The `main.py` file is **26,570 lines** - far too large for maintainability.

## Solution
Created modular CLI structure:

### New Structure
```
cli/
├── __init__.py              # Package exports
├── pipeline_builder.py      # Extracted build_pipeline() function
└── commands/                # Individual command modules (future)
    ├── __init__.py
    ├── run_once.py          # run-once command (to be extracted)
    ├── live_loop.py         # live-loop command (to be extracted)
    ├── scan.py              # scan-opportunities command (to be extracted)
    └── multi_symbol.py      # multi-symbol-loop command (to be extracted)
```

### Migration Strategy
1. ✅ Extracted `build_pipeline()` → `cli/pipeline_builder.py`
2. ⏳ Extract individual Typer commands to separate modules
3. ⏳ Update `main.py` to import from `cli.commands.*`
4. ⏳ Reduce `main.py` to <500 lines (just CLI app setup + imports)

### Benefits
- **Maintainability**: Each command in own file (~200-500 lines)
- **Testability**: Can test commands independently
- **Readability**: Clear separation of concerns
- **Collaboration**: Reduces merge conflicts

### Usage
Original import still works:
```python
from main import build_pipeline  # Still works
```

New modular import:
```python
from cli import build_pipeline  # Preferred
```

## TODO
- [ ] Extract `run_once()` → `cli/commands/run_once.py`
- [ ] Extract `live_loop()` → `cli/commands/live_loop.py`
- [ ] Extract `scan_opportunities()` → `cli/commands/scan.py`
- [ ] Extract `multi_symbol_loop()` → `cli/commands/multi_symbol.py`
- [ ] Update `main.py` to use new structure
- [ ] Add command registration in `cli/commands/__init__.py`
