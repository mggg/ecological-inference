# Migration Notes

This document describes breaking changes and migration instructions for the PyEI package.

## Version 1.2.0 - Code Refactoring and Type Annotations (Breaking Changes)

### Overview
This release includes significant internal refactoring to improve code organization and maintainability, plus comprehensive type annotations throughout the codebase. While the public API remains largely unchanged, some internal functions have been renamed and import patterns have been updated.

### Breaking Changes

#### 1. Private Function Renaming
Several internal helper functions have been renamed to follow Python conventions for private functions (leading underscore).

**Affected Functions:**
- `size_ticks()` → `_size_ticks()`
- `size_yticklabels()` → `_size_yticklabels()`
- `plot_single_ridgeplot()` → `_plot_single_ridgeplot()`
- `plot_single_histogram()` → `_plot_single_histogram()`
- `log_binom_sum()` → `_log_binom_sum()`
- `binom_conv_log_p()` → `_binom_conv_log_p()`
- `goodmans_er_bayes_model()` → `_goodmans_er_bayes_model()`
- `goodmans_er_bayes_pop_weighted_model()` → `_goodmans_er_bayes_pop_weighted_model()`
- `r_function()` → `_r_function()`
- `sample_low_to_high()` → `_sample_low_to_high()`
- `sample_high_to_low()` → `_sample_high_to_low()`
- `sample_Sigma()` → `_sample_Sigma()`
- `sample_mu()` → `_sample_mu()`
- `proposal_dist_generate_sample()` → `_proposal_dist_generate_sample()`
- `log_unnormalized_pdf()` → `_log_unnormalized_pdf()`
- `theta_to_omega()` → `_theta_to_omega()`
- `sample_theta()` → `_sample_theta()`
- `omega_to_theta()` → `_omega_to_theta()`
- `sample_internal_cell_counts()` → `_sample_internal_cell_counts()`
- `get_initial_internal_count_sample()` → `_get_initial_internal_count_sample()`

**Migration Required:** If you were directly importing or calling any of these functions, update your code to use the new names with leading underscores.

**Example:**
```python
# Before (v1.1.x)
from pyei.plot_utils import size_ticks
size_ticks(ax, "x")

# After (v1.2.0+)
from pyei.plot_utils import _size_ticks
_size_ticks(ax, "x")
```

#### 2. Import Pattern Changes
All relative imports within the package have been converted to absolute imports.

**Migration Required:** If you were copying import patterns from the PyEI source code, update them to use absolute imports.

**Example:**
```python
# Before (internal PyEI code)
from .plot_utils import plot_kdes
from .r_by_c_models import ei_multinom_dirichlet

# After (internal PyEI code)
from pyei.plot_utils import plot_kdes
from pyei.r_by_c_models import ei_multinom_dirichlet
```

#### 3. Removed `__all__` Specifiers
All `__all__` specifiers have been removed from module files.

**Impact:** This affects `from module import *` statements. While not recommended, if you were using wildcard imports, you may need to import specific functions explicitly.

**Example:**
```python
# Before (if using wildcard imports)
from pyei.plot_utils import *
# This would import only functions listed in __all__

# After (recommended approach)
from pyei.plot_utils import plot_kdes, plot_summary
# Import only what you need explicitly
```

#### 4. Type Annotations Added
Comprehensive type annotations have been added throughout the codebase using Python 3.11+ syntax.

**Changes:**
- All function parameters and return types are now annotated
- Class attributes are typed
- Use of modern Python syntax (`X | None` instead of `Optional[X]`)
- Proper typing for NumPy arrays, Matplotlib objects, and PyMC models
- Specific type annotations for ArviZ `InferenceData` objects
- Added `None` checks and type narrowing where appropriate
- Replaced `typing.Any` with specific types where possible
- Added `# noqa: ANN401` comments for necessary `Any` usage in `**kwargs` parameters

**Impact:** This improves code documentation and enables better IDE support and static type checking with `mypy`.

**Example:**
```python
# Before (no type annotations)
def plot_kdes(sampled_voting_prefs, group_names, candidate_names, plot_by="candidate", axes=None):
    return axes

# After (with type annotations)
def plot_kdes(
    sampled_voting_prefs: np.ndarray,
    group_names: list[str],
    candidate_names: list[str],
    plot_by: str = "candidate",
    axes: Axes | None = None
) -> Axes:
    return axes
```

**Type Safety Improvements:**
- Fixed matplotlib subplot array handling for better seaborn compatibility
- Improved type safety for plotting utilities and model parameters
- Resolved all `ANN401` linting errors (dynamically typed expressions)
- Enhanced type checking with zero `mypy` errors

### Non-Breaking Changes

#### 1. Public API Unchanged
All public classes and methods remain unchanged:
- `TwoByTwoEI`
- `TwoByTwoEIBaseBayes`
- `GoodmansER`
- `GoodmansERBayes`
- `RowByColumnEI`
- `Datasets`

#### 2. Functionality Preserved
All existing functionality works exactly the same way. Only internal implementation details have changed.

#### 3. Type Checking Support
The codebase now passes `mypy` type checking with zero errors, providing better development experience and code reliability.

#### 4. Linting Compliance
All `ANN401` errors (dynamically typed expressions) have been resolved by:
- Replacing `typing.Any` with specific types where possible
- Adding `# noqa: ANN401` comments for necessary `Any` usage in `**kwargs` parameters
- Ensuring type safety while maintaining flexibility for dynamic parameters

All `PD013` and `PD011` errors (pandas linting rules) have been resolved by:
- Adding `# noqa: PD013` comments for xarray `.stack()` calls (not pandas)
- Adding `# noqa: PD011` comments for xarray `.values` calls (not pandas)
- These are false positive warnings since the code uses xarray objects, not pandas DataFrames

### Migration Checklist

- [ ] Search your codebase for any direct imports of the renamed private functions
- [ ] Update function calls to use the new names with leading underscores
- [ ] Replace any wildcard imports (`from module import *`) with explicit imports
- [ ] Test your code to ensure it still works as expected
- [ ] Consider running `mypy` on your code to take advantage of the new type annotations

### Need Help?

If you encounter issues during migration:
1. Check this migration guide for the specific function you're using
2. Look at the updated PyEI source code for correct import patterns
3. Open an issue on GitHub if you need assistance

### Future Considerations

- Private functions (those with leading underscores) are considered internal implementation details
- Future versions may change or remove private functions without notice
- Always use public APIs when possible for better compatibility across versions
- Type annotations will continue to be maintained and improved in future releases
- Consider using `mypy` in your development workflow to catch type-related issues early



# Migration Notes: Docker → uv + Virtual Environments

This document explains the modernization changes made to the development workflow.

## Changes Made

### Replaced Docker with uv Virtual Environments
- **Removed**: `Dockerfile` and Docker-based workflow
- **Added**: `uv`-based virtual environment workflow
- **Updated**: Python requirement from 3.10 to 3.11

### Modernized Package Management
- **Removed**: `setup.py` (replaced with `pyproject.toml`)
- **Replaced**: `requirements.txt` and `requirements-dev.txt` → dependency groups in `pyproject.toml`
- **Updated**: Build system from `setuptools` to `hatchling`
- **Added**: `uv.lock` for exact reproducibility
- **Replaced**: `pip` → `uv` (faster, more reliable)
- **Replaced**: `black` + `pylint` → `ruff` (faster, unified)

### Consolidated CI/CD Workflow
- **Consolidated**: Multiple workflow files → single `ci.yml`
- **Updated**: Uses Makefile targets for consistency
- **New structure**:
  - Linting runs on all commits (push and PR)
  - Testing runs only on PRs (non-draft only)
  - Publishing runs only on pushes to main

## Command Equivalents

| Old Command | New Command |
|-------------|-------------|
| `make build` | `make setup` |
| `make test` (Docker) | `make test` (uv) |
| `make lint` (Docker) | `make lint` (uv) |
| `make bash` (Docker) | `source .venv/bin/activate` |
| `pip install -e .` | `uv pip install -e .` |
| `pip install -r requirements-dev.txt` | `uv pip install -e ".[dev]"` |

## For Contributors

### Development Setup
```bash
make setup          # Create venv and install dev dependencies
source .venv/bin/activate  # Activate virtual environment
make test           # Run tests
make lint           # Run linting
make format         # Format code
```

### Direct uv Commands
```bash
uv run pytest       # Run tests
uv run ruff check   # Lint code
uv run ruff format  # Format code
uv run mypy pyei    # Type checking
```

### Dependency Management
```bash
uv pip list         # Show installed packages
uv lock --upgrade   # Update lock file
```

## Publishing
To publish a new version:
1. Update the version in `pyproject.toml`
2. Commit and push to main
3. The CI/CD workflow automatically publishes to PyPI

## Backward Compatibility

- All functionality is preserved, just modernized
- Virtual environments ensure consistent environments across machines