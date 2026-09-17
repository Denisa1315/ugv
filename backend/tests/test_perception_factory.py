import pytest

from app.modules.perception.factory import create_perception_module
from app.modules.perception.fallback import ClassicalPerception


def test_fallback_mode_returns_classical():
    module = create_perception_module("fallback")
    assert isinstance(module, ClassicalPerception)


def test_auto_mode_returns_a_ready_module():
    module = create_perception_module("auto")
    assert module.is_ready is True
    assert module.mode in ("real", "fallback")


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        create_perception_module("not_a_mode")
