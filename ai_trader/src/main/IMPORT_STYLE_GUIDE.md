# AI Trader Import Style Guide

This guide defines the standard import patterns for the AI Trader codebase.

## Import Order

All imports should be organized into three groups, separated by blank lines:

1. **Standard library imports**
2. **Third-party imports**
3. **Local application imports**

Within each group, imports should be alphabetically sorted.

## Import Format

### Standard Library
```python
import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
```

### Third-party
```python
import numpy as np
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, Integer, String
import structlog
```

### Local Imports
```python
from main.config.config_manager import get_config
from main.data_pipeline.types import RawDataRecord
from main.utils.exceptions import DataFetchError, ValidationError
from main.utils.logging import get_logger
```

## Rules and Guidelines

### 1. Use Absolute Imports for Cross-Module References
✅ **Good:**
```python
from main.utils.cache import CacheManager
from main.data_pipeline.processing import DataProcessor
```

❌ **Bad:**
```python
from ai_trader.utils.cache import CacheManager  # Wrong module prefix
from ...utils.cache import CacheManager  # Too many relative imports
```

### 2. Use Relative Imports Only in Package __init__.py Files
✅ **Good (in __init__.py):**
```python
from .base_scanner import BaseScanner
from .layer0_static_universe import Layer0StaticUniverseScanner
from .layer1_liquidity_scanner import Layer1LiquidityScanner
```

❌ **Bad (in regular modules):**
```python
from . import utils  # Avoid in non-__init__ files
from ..config import settings  # Use absolute imports instead
```

### 3. Avoid Star Imports Except in __init__.py
✅ **Good (in __init__.py with __all__):**
```python
from .models import *

__all__ = ['Model1', 'Model2', 'Model3']
```

❌ **Bad (in regular modules):**
```python
from main.utils import *  # Too broad, unclear dependencies
```

### 4. Group Long Imports with Parentheses
✅ **Good:**
```python
from main.utils.exceptions import (
    APIError,
    DataFetchError,
    RateLimitError,
    ValidationError,
)
```

❌ **Bad:**
```python
from main.utils.exceptions import APIError, DataFetchError, \
    RateLimitError, ValidationError  # Avoid backslash continuation
```

### 5. Import What You Use
✅ **Good:**
```python
from main.utils.logging import get_logger
logger = get_logger(__name__)
```

❌ **Bad:**
```python
import main.utils.logging  # Import the specific function instead
logger = main.utils.logging.get_logger(__name__)
```

### 6. Handle Circular Imports
If you encounter circular imports:

1. **Use TYPE_CHECKING for type hints:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main.trading_engine.engine import TradingEngine
```

2. **Import inside functions:**
```python
def process_data():
    from main.data_pipeline.processor import DataProcessor
    processor = DataProcessor()
```

3. **Refactor to eliminate circular dependencies**

### 7. Standard Import Aliases
Use these standard aliases for common libraries:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
```

## Example: Complete Import Section

```python
"""Module docstring explaining the purpose of this module."""

# Standard library imports
import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from omegaconf import DictConfig
from pydantic import BaseModel, Field
import structlog

# Local imports
from main.config.config_manager import get_config
from main.data_pipeline.types import RawDataRecord
from main.utils.exceptions import (
    DataFetchError,
    RateLimitError,
    ValidationError,
)
from main.utils.logging import get_logger

# Module-level logger
logger = get_logger(__name__)
```

## Tools Configuration

### isort Configuration (.isort.cfg)
```ini
[settings]
profile = black
line_length = 88
multi_line_output = 3
include_trailing_comma = True
force_grid_wrap = 0
use_parentheses = True
ensure_newline_before_comments = True
known_first_party = main
known_third_party = alpaca,numpy,pandas,pydantic,omegaconf,sqlalchemy,structlog
sections = FUTURE,STDLIB,THIRDPARTY,FIRSTPARTY,LOCALFOLDER
import_heading_stdlib = Standard library imports
import_heading_thirdparty = Third-party imports
import_heading_firstparty = Local imports
```

### Pre-commit Hook (.pre-commit-config.yaml)
```yaml
repos:
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile', 'black']
```

## Migration Steps

1. **Run the import standardization script:**
   ```bash
   python tools/standardize_imports.py
   ```

2. **Review changes:**
   ```bash
   git diff
   ```

3. **Run tests to ensure nothing broke:**
   ```bash
   ./tools/run_tests.sh
   ```

4. **Commit the standardized imports:**
   ```bash
   git commit -m "refactor: standardize import patterns across codebase"
   ```

## Enforcement

- Use `isort` in pre-commit hooks
- Configure your IDE to auto-format imports
- Include import style in code reviews