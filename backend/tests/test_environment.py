import math

from app.modules.simulator.environment import Environment


def test_add_and_clear_obstacles():
    env = Environment()
    o = env.add_obstacle(2.0, 3.0, radius=0.5)
    assert len(env.obstacles) == 1
    assert o.id
    env.clear_obstacles()
    assert env.obstacles == []


def test_remove_obstacle_by_id():
    env = Environment()
    o = env.add_obstacle(1.0, 1.0)
    assert env.remove_obstacle(o.id) is True
    assert env.remove_obstacle("does-not-exist") is False


def test_distance_to_goal():
    env = Environment(goal=(3.0, 4.0))
    assert math.isclose(env.distance_to_goal(0.0, 0.0), 5.0)


def test_nearest_obstacle_distance():
    env = Environment()
    assert env.nearest_obstacle_distance(0, 0) is None
    env.add_obstacle(5.0, 0.0, radius=1.0)
    assert math.isclose(env.nearest_obstacle_distance(0.0, 0.0), 4.0)


def test_is_within_bounds():
    env = Environment(width=10, height=10)
    assert env.is_within_bounds(5, 5) is True
    assert env.is_within_bounds(-1, 5) is False
    assert env.is_within_bounds(5, 11) is False
