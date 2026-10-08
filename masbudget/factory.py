"""Generic name -> class registry used by every factory (policies, utilities, metrics, ...)."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a factory name (the string used in YAML configs) to a class.

    Usage::

        POLICIES = Registry("centralized policy")

        @POLICIES.register("optimal_mckp")
        class OptimalMCKPPolicy(BaseCentralizedPolicy): ...

        policy = POLICIES.create("optimal_mckp", **params)
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._classes: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._classes:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            cls.registry_name = name  # type: ignore[attr-defined]
            self._classes[name] = cls
            return cls

        return decorator

    def create(self, name: str, /, *args: Any, **params: Any) -> T:
        if name not in self._classes:
            raise ValueError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names()) or '(none)'}"
            )
        return self._classes[name](*args, **params)

    def names(self) -> list[str]:
        return sorted(self._classes)

    def __contains__(self, name: object) -> bool:
        return name in self._classes


def import_submodules(package_name: str) -> None:
    """Import every module of a package so their ``@register`` decorators run.

    Dropping a new file into a policy folder is therefore enough to make it available.
    """
    package = importlib.import_module(package_name)
    for info in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package_name}.{info.name}")
