"""Centralized allocation policies: a controller with global information assigns every agent a model."""

from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy, create_centralized_policy
from masbudget.factory import import_submodules

import_submodules(__name__)

__all__ = ["CENTRALIZED_POLICIES", "BaseCentralizedPolicy", "create_centralized_policy"]
