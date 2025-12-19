"""
Smoke test to verify pytest collection works correctly.

This test ensures that pytest can discover and run tests.
If this test fails, there's a configuration issue with pytest.
"""


def test_pytest_collects():
    """Verify pytest can collect and run tests."""
    assert True


def test_pytest_imports():
    """Verify basic imports work."""
    import sys
    from pathlib import Path
    
    # Verify repo root is on path
    repo_root = Path(__file__).resolve().parents[1]
    assert str(repo_root) in sys.path or any(str(repo_root) in p for p in sys.path)
    
    # Verify app module can be imported
    try:
        import app
        assert app is not None
    except ImportError:
        # If app is not a package, try importing a submodule
        from app import settings
        assert settings is not None

