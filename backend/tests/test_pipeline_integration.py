import time

from app.modules.camera.demo_source import DemoFrameSource
from app.modules.depth.factory import create_depth_estimator
from app.modules.localization.factory import create_localizer
from app.modules.perception.factory import create_perception_module
from app.modules.pipeline.orchestrator import NavigationPipeline
from app.modules.simulator.environment import Environment
from app.modules.simulator.simulator import Simulator


def build_pipeline(env):
    camera = DemoFrameSource(width=160, height=120)
    perception = create_perception_module("fallback")
    depth = create_depth_estimator()
    localizer = create_localizer("orb")
    return NavigationPipeline(env, camera, perception, depth, localizer)


def test_pipeline_drives_vehicle_around_known_obstacle_to_goal():
    env = Environment(width=20, height=10, goal=(18.0, 5.0))
    env.add_obstacle(9.0, 5.0, radius=0.8)  # sits directly on the straight-line route

    sim = Simulator(environment=env, tick_hz=10.0)
    sim.reset(x=1.0, y=5.0, heading=0.0)

    pipeline = build_pipeline(env)
    sim.pre_tick_hook = pipeline.as_pre_tick_hook(sim)
    sim.start()

    dt = 1.0 / sim.tick_hz
    t0 = time.time()
    # The decision engine (phase 1k) genuinely slows/pauses the vehicle
    # under transient low confidence or proximity, per the safety rules —
    # a larger, still-generous tick budget accounts for that real caution
    # rather than the old, decision-engine-free pacing.
    for _ in range(1500):
        sim.tick(dt)
        if sim.goal_reached or sim.collided:
            break
    wall_time = time.time() - t0

    assert sim.collided is False, "pipeline-driven vehicle must not collide with a known static obstacle"
    assert sim.goal_reached is True, f"vehicle should reach the goal (final state: {sim.vehicle.state})"
    assert pipeline.last_cycle is not None
    assert wall_time < 60.0, f"pipeline cycles took {wall_time:.1f}s — too slow for real-time use"


def test_pipeline_replans_when_new_obstacle_appears_on_path():
    env = Environment(width=20, height=10, goal=(18.0, 5.0))  # initially open
    sim = Simulator(environment=env, tick_hz=10.0)
    sim.reset(x=1.0, y=5.0, heading=0.0)

    pipeline = build_pipeline(env)
    sim.pre_tick_hook = pipeline.as_pre_tick_hook(sim)
    sim.start()

    dt = 1.0 / sim.tick_hz
    for _ in range(20):
        sim.tick(dt)
    replans_before = pipeline.local_planner.replan_count
    assert replans_before >= 1, "an initial plan should have been generated"

    # Drop a new obstacle directly ahead of the vehicle's current heading.
    ahead_x = sim.vehicle.state.x + 3.0
    env.add_obstacle(ahead_x, sim.vehicle.state.y, radius=0.8)

    replanned_after_injection = False
    for _ in range(300):
        sim.tick(dt)
        if pipeline.local_planner.replan_count > replans_before:
            replanned_after_injection = True
        if sim.goal_reached or sim.collided:
            break

    assert sim.collided is False, "the newly-injected obstacle must not be driven through"
    assert replanned_after_injection, "the local planner must replan after a new obstacle appears on the active path"
