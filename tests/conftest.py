import pytest

from tests.helpers import make_agents, make_models


@pytest.fixture
def agents():
    return make_agents()


@pytest.fixture
def models():
    return make_models()


@pytest.fixture
def register(monkeypatch):
    """Temporarily register a test-only class: ``register(REGISTRY, "name", cls)``."""

    def _register(registry, name, cls):
        monkeypatch.setitem(registry._classes, name, cls)
        monkeypatch.setattr(cls, "registry_name", name, raising=False)
        return cls

    return _register
