# AGENTS.md - Development Guide

## Architecture Principles

### 1. Single Source of Truth
- **Types**: One place for all type definitions (`types.py`)
- **Constants**: One place for all configuration (`constants.py`)
- **Defaults**: One place for all default values (not scattered across CLI)

### 2. Separation of Concerns
```
Library Code (reusable)     CLI Code (user-facing)
├── hardware.py             ├── cli.py
├── calculator.py           └── validators.py
├── downloader.py
└── llama.py
```

Library code: No `sys.exit()`, no `print()`, raise exceptions
CLI code: Handle exceptions, format output, exit on errors

### 3. Data Flow
```
Input → Validation → Business Logic → Output
  │         │              │            │
CLI args  validators    modules      print/click
```

## Design Patterns

### Exception Hierarchy
```
LlamaTunerError (base)
├── HardwareNotDetectedError
├── LlamaNotInstalledError
├── ModelNotFoundError
└── BenchmarkFailedError
```

**Rule**: Library functions raise, CLI catches and exits.

### Serialization Pattern
```python
@dataclass
class Profile:
    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        """Reconstruct from cache."""
        ...
    
    def to_dict(self) -> dict:
        """Serialize for cache."""
        ...
```

**Rule**: All dataclasses that need persistence should have `from_dict()`.

### Validation Pattern
```python
def validate_xxx() -> SomeType:
    """Validate and return, or raise."""
    if not condition:
        raise XxxError("message")
    return result
```

**Rule**: Validators return the validated object or raise.

## Coding Standards

### Type Safety
```python
# Good
def calculate(hardware: HardwareProfile, quant: Quant) -> OptimalArgs:
    ...

# Bad
def calculate(hardware, quant):  # Missing type hints
    ...
```

**Rule**: All public functions must have type hints.

### Constants vs Magic Numbers
```python
# Good
if vram_mb > HIGH_VRAM_THRESHOLD_MB:
    return BATCH_SIZE_HIGH_VRAM

# Bad
if vram_mb > 8000:  # Magic number
    return 2048
```

**Rule**: Extract all thresholds, limits, and defaults to `constants.py`.

### Import Organization
```python
# Standard library
import re
import sys
from pathlib import Path

# Third-party
import click
from huggingface_hub import hf_hub_download

# Local
from llamacpp_tuner.types import Quant
from llamacpp_tuner.constants import DEFAULT_PORT
```

**Rule**: Three sections: stdlib, third-party, local (sorted alphabetically).

### Function Size
```python
# Good: Single responsibility, max 30 lines
def validate_model_path(repo_id: str, quant: Quant) -> Path:
    model_path = get_model_path(repo_id, quant)
    if not model_path:
        raise ModelNotFoundError(repo_id, quant)
    return model_path

# Bad: Multiple responsibilities
def process_model(repo_id: str, quant: str) -> None:
    # 50+ lines doing multiple things
    ...
```

**Rule**: One function = one job. Extract helpers if needed.

### Error Messages
```python
# Good: Actionable
raise HardwareNotDetectedError(
    "Hardware not detected. Run 'lct setup' first."
)

# Bad: Vague
raise Exception("Error occurred")
```

**Rule**: Tell user what to do next.

## File Organization

### Module Structure
```python
"""Module docstring."""

# Imports (3 sections)

# Constants (module-level)

# Public classes/functions

# Private helpers (prefixed with _)
```

### When to Create New Module
- New domain concept (e.g., `benchmark.py` for performance testing)
- Related functionality (e.g., `downloader.py` for all HuggingFace operations)
- Reusable utilities (e.g., `validators.py` for CLI validation)

## Naming Conventions

### Functions
- `get_xxx()` - retrieve data (never raises)
- `validate_xxx()` - check and return or raise
- `calculate_xxx()` - compute derived value
- `format_xxx()` - convert to string
- `_private_xxx()` - internal helper

### Classes
- Data: `XxxProfile`, `XxxInfo`, `XxxResult`
- Actions: `XxxError`, `XxxValidator`

### Variables
```python
# Good: Descriptive
vram_mb = mem.total // (1024 * 1024)
total_vram = sum(g.vram_mb for g in gpus)

# Bad: Cryptic
v = mem.total // (1024 * 1024)
t = sum(g.v for g in gpus)
```

## Testing Philosophy

### Test Structure
```python
class TestXxx:
    def test_happy_path(self):
        """Test normal operation."""
    
    def test_edge_case(self):
        """Test boundary conditions."""
    
    def test_error_case(self):
        """Test error handling."""
```

### Test Principles
1. **Independent**: Each test runs alone
2. **Fast**: Mock external dependencies (subprocess, network, file I/O)
3. **Readable**: Test name describes what it tests
4. **Deterministic**: Same input → same output

## Performance Guidelines

### Optimization Priority
1. Correctness
2. Readability
3. Performance (only if measured)

### Avoid Premature Optimization
```python
# Good: Clear and correct
for file in files:
    if pattern in file.name:
        return file

# Bad: Optimized before measuring
files_by_pattern = defaultdict(list)  # Complex caching
...
```

## Dependency Management

### Adding New Dependency
1. Check if truly needed (can we use stdlib?)
2. Add to `pyproject.toml` with version constraint
3. Import at top of module (not inside functions)
4. Handle `ImportError` gracefully if optional

### Import Inside Function (Only for Optional Dependencies)
```python
def detect_gpu():
    try:
        import pynvml  # Optional dependency
        pynvml.nvmlInit()
        ...
    except ImportError:
        return []  # Graceful fallback
```

## Configuration

### Configuration Hierarchy
1. **Constants** (`constants.py`) - hardcoded defaults
2. **Cache** (`cache.py`) - user-specific values
3. **CLI args** - runtime overrides

### When to Add New Constant
- Threshold (e.g., `HIGH_VRAM_THRESHOLD_MB`)
- Default value (e.g., `DEFAULT_PORT`)
- Timeout (e.g., `SERVER_TIMEOUT_SECONDS`)
- Limit (e.g., `MAX_SERVER_LOG_LINES`)

## Extending the Codebase

### Adding New Quantization Level
1. Add to `Quant` type in `types.py`
2. Add to `QUANT_CHOICES` in `types.py`
3. Add bytes per param in `constants.py`
4. Add to `BYTES_PER_PARAM` dict in `calculator.py`
5. Add pattern mapping in `downloader.py`

### Adding New CLI Command
1. Define command function in `cli.py`
2. Use existing validators
3. Use existing constants
4. Handle exceptions with `_handle_error()`

### Adding New Backend
1. Add to `Backend` type in `types.py`
2. Add detection logic in `hardware.py`
3. Add optimization logic in `calculator.py`

## Common Pitfalls

### Don't
```python
# Duplicate logic
def cmd1():
    cached = load_cache("hardware")
    hardware = _load_from_cache(cached)
    ...

def cmd2():
    cached = load_cache("hardware")
    hardware = _load_from_cache(cached)  # Duplicated!
    ...
```

### Do
```python
# Reusable validator
def cmd1():
    hardware = validate_hardware_cache()
    ...

def cmd2():
    hardware = validate_hardware_cache()  # Reused!
    ...
```

### Don't
```python
# Magic numbers scattered
if vram > 8000:
    batch = 2048
elif vram > 4000:
    batch = 1024
```

### Do
```python
# Constants in one place
if vram > HIGH_VRAM_THRESHOLD_MB:
    batch = BATCH_SIZE_HIGH_VRAM
elif vram > MED_VRAM_THRESHOLD_MB:
    batch = BATCH_SIZE_MED_VRAM
```

## Code Review Checklist

- [ ] Type hints on all public functions
- [ ] No magic numbers (use constants)
- [ ] No `sys.exit()` in library code
- [ ] No `print()` in library code (use exceptions)
- [ ] Single responsibility per function
- [ ] Descriptive variable names
- [ ] Imports sorted and grouped
- [ ] Tests for new functionality
- [ ] Error messages are actionable
