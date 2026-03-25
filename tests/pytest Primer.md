## pytest: Crib Sheet / Daily Reference

### Installation & Setup

```bash
uv add --dev pytest pytest-cov pytest-xdist pytest-mock
```

**`pyproject.toml` config** (preferred):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
python_files = "test_*.py *_test.py"
python_classes = "Test*"
python_functions = "test_*"
```

Or `pytest.ini`:
```ini
[pytest]
testpaths = tests
addopts = -v --tb=short
```

### Test Discovery

pytest finds tests by scanning for:
- Files matching `test_*.py` or `*_test.py`
- Classes prefixed `Test` (no `__init__`)
- Functions/methods prefixed `test_`

```
tests/
├── conftest.py          # shared fixtures (auto-loaded)
├── test_auth.py
└── unit/
    ├── conftest.py      # scoped fixtures
    └── test_models.py
```

### Writing Tests

```python
# Basic assertions — use plain assert, pytest rewrites them
def test_addition():
    assert 1 + 1 == 2

def test_string():
    result = "hello world"
    assert "hello" in result
    assert result.startswith("hello")

def test_raises():
    with pytest.raises(ValueError, match="invalid"):
        raise ValueError("invalid input")

def test_raises_captures():
    with pytest.raises(KeyError) as exc_info:
        {}["missing"]
    assert exc_info.value.args[0] == "missing"

def test_approx():
    assert 0.1 + 0.2 == pytest.approx(0.3)
    assert [0.1, 0.2] == pytest.approx([0.1, 0.2], rel=1e-3)
```

### Fixtures

The core of pytest. Fixtures provide setup/teardown and dependency injection.

```python
import pytest

@pytest.fixture
def user():
    return {"name": "Anil", "role": "admin"}

def test_user_name(user):
    assert user["name"] == "Anil"
```

**Scopes** — control how often a fixture is created:

| Scope | Created once per... |
|---|---|
| `function` (default) | test function |
| `class` | test class |
| `module` | test file |
| `package` | package directory |
| `session` | entire test run |

```python
@pytest.fixture(scope="module")
def db_connection():
    conn = create_connection()
    yield conn          # code after yield = teardown
    conn.close()
```

**`yield` fixtures** — the yield is when the test runs:
```python
@pytest.fixture
def tmp_file(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("hello")
    yield f
    # cleanup runs here automatically (even on test failure)
```

**`conftest.py`** — fixtures defined here are available to all tests in the same directory and below, no import needed.

**`autouse`** — runs for every test in scope automatically:
```python
@pytest.fixture(autouse=True)
def reset_db():
    yield
    db.rollback()
```

**Parametrize a fixture:**
```python
@pytest.fixture(params=["sqlite", "postgres"])
def db(request):
    return connect(request.param)
```

**Built-in fixtures you'll use constantly:**

| Fixture | What it gives you |
|---|---|
| `tmp_path` | `pathlib.Path` to a per-test temp dir |
| `tmp_path_factory` | session-scoped temp dirs |
| `capsys` | capture stdout/stderr |
| `capfd` | capture file descriptors |
| `monkeypatch` | safely patch objects/env/imports |
| `request` | info about the requesting test |
| `mocker` | (pytest-mock) `unittest.mock` integration |

### Parametrize

Run one test with many inputs:

```python
@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_add(a, b, expected):
    assert a + b == expected
```

With IDs:
```python
@pytest.mark.parametrize("value", [
    pytest.param(None, id="none"),
    pytest.param("", id="empty"),
    pytest.param("hello", id="normal"),
])
def test_truthy(value):
    ...
```

Stacked (cartesian product):
```python
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [10, 20])
def test_product(x, y):   # runs 4 times
    ...
```

### Marks

```python
import pytest

@pytest.mark.skip(reason="not implemented yet")
def test_wip(): ...

@pytest.mark.skipif(sys.platform == "win32", reason="linux only")
def test_unix(): ...

@pytest.mark.xfail(reason="known bug", strict=True)
def test_known_broken(): ...
    # strict=True: fails the suite if it unexpectedly passes

@pytest.mark.slow
def test_big_model(): ...
```

Register custom marks in `pyproject.toml` to avoid warnings:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
]
```

Run/skip by mark:
```bash
pytest -m slow
pytest -m "not slow"
pytest -m "slow or integration"
```

### Mocking (pytest-mock)

```python
def test_api_call(mocker):
    mock_get = mocker.patch("mymodule.requests.get")
    mock_get.return_value.json.return_value = {"status": "ok"}

    result = mymodule.fetch_status()
    assert result == "ok"
    mock_get.assert_called_once_with("https://api.example.com/status")
```

```python
# Patch object attribute
mocker.patch.object(MyClass, "method", return_value=42)

# Patch environment variable
mocker.patch.dict(os.environ, {"API_KEY": "test-key"})

# Spy (calls through but tracks calls)
spy = mocker.spy(mymodule, "my_function")
mymodule.my_function(1, 2)
spy.assert_called_once_with(1, 2)
```

**monkeypatch** (no extra lib):
```python
def test_env(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.delenv("SECRET", raising=False)
    monkeypatch.setattr("mymodule.TIMEOUT", 0.1)
    monkeypatch.syspath_prepend("/some/path")
```

### capsys / capfd

```python
def test_output(capsys):
    print("hello")
    captured = capsys.readouterr()
    assert captured.out == "hello\n"
    assert captured.err == ""
```

### Running Tests — CLI Reference

```bash
pytest                          # all tests
pytest tests/test_auth.py       # specific file
pytest tests/test_auth.py::test_login  # specific test
pytest tests/test_auth.py::TestClass::test_method

pytest -v                       # verbose (show test names)
pytest -vv                      # even more verbose (show diffs)
pytest -s                       # don't capture stdout (see prints)
pytest -x                       # stop on first failure
pytest --lf                     # rerun only last-failed
pytest --ff                     # run failed first, then rest
pytest -k "login or auth"       # filter by name substring
pytest -k "not slow"
pytest --tb=short               # short tracebacks (default: auto)
pytest --tb=long
pytest --tb=no                  # no tracebacks
pytest -q                       # quiet output
pytest -n 4                     # parallel (pytest-xdist), 4 workers
pytest -n auto                  # use all CPU cores
pytest --co                     # collect only (dry run, show what would run)
pytest --durations=10           # show 10 slowest tests
```

### Coverage (pytest-cov)

```bash
pytest --cov=mypackage
pytest --cov=mypackage --cov-report=term-missing   # show uncovered lines
pytest --cov=mypackage --cov-report=html           # → htmlcov/index.html
pytest --cov=mypackage --cov-fail-under=80         # fail if < 80%
```

`pyproject.toml`:
```toml
[tool.coverage.run]
source = ["mypackage"]
omit = ["*/migrations/*", "*/tests/*"]

[tool.coverage.report]
exclude_lines = ["if TYPE_CHECKING:", "pass"]
```

### Class-Based Tests

```python
class TestAuthentication:
    def test_valid_login(self, user):
        assert login(user) is True

    def test_invalid_password(self):
        with pytest.raises(AuthError):
            login({"password": "wrong"})
```

No `__init__`, no inheritance from `unittest.TestCase` needed (though pytest supports both).

### Async Tests (pytest-asyncio)

```bash
pip install pytest-asyncio
```

```python
import pytest

@pytest.mark.asyncio
async def test_async_fetch():
    result = await fetch_data()
    assert result is not None
```

Or configure globally:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"   # marks all async tests automatically
```

### Useful Patterns

**Fixture factories:**
```python
@pytest.fixture
def make_user():
    def _make(name="test", role="user", **kwargs):
        return User(name=name, role=role, **kwargs)
    return _make

def test_admin(make_user):
    admin = make_user(role="admin")
    assert admin.can_delete()
```

**Shared data via `request.param` + indirect:**
```python
@pytest.fixture
def threshold(request):
    return request.param

@pytest.mark.parametrize("threshold", [0.5, 0.9], indirect=True)
def test_model_accuracy(threshold):
    assert accuracy() > threshold
```

**Skip entire module:**
```python
pytestmark = pytest.mark.skip("module not ready")
# or
pytestmark = pytest.mark.slow
```

**Custom assertion helpers:**
```python
# pytest rewrites assert statements for better diffs
# just use plain assert with clear expressions
assert result == expected, f"got {result!r}, wanted {expected!r}"
```

### Exit Codes

| Code | Meaning |
|---|---|
| 0 | All tests passed |
| 1 | Tests failed |
| 2 | Test execution interrupted |
| 3 | Internal error |
| 4 | CLI usage error |
| 5 | No tests collected |

### Quick Cheatsheet

```bash
# Most common daily invocations
pytest -x -v --tb=short          # dev: fail fast, verbose
pytest --lf -v                   # re-run failures only
pytest -k "audio" -v             # filter by name
pytest -n auto --dist=loadfile   # parallel, same file → same worker
pytest --co -q                   # what will run, no output noise
```

The most important things to internalize: 
- **fixtures over setUp/tearDown**
- **`yield` for teardown**
- **`conftest.py` for sharing fixtures**
- **`monkeypatch`/`mocker` for isolation**
