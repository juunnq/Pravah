"""Tests for World geometry utilities."""

import numpy as np
from sim.core.world import Wall, World, point_to_segment_distance, agents_to_walls


def test_perpendicular_distance():
    """Point (5,3), segment (0,0)->(10,0) -> dist=3, normal=(0,1)."""
    dist, closest, normal = point_to_segment_distance(
        np.array([5.0, 3.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0])
    )
    assert abs(dist - 3.0) < 1e-10
    np.testing.assert_allclose(closest, [5.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(normal, [0.0, 1.0], atol=1e-10)


def test_endpoint_distance():
    """Point (12,0), segment (0,0)->(10,0) -> closest=(10,0), dist=2."""
    dist, closest, normal = point_to_segment_distance(
        np.array([12.0, 0.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0])
    )
    assert abs(dist - 2.0) < 1e-10
    np.testing.assert_allclose(closest, [10.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(normal, [1.0, 0.0], atol=1e-10)


def test_point_on_segment():
    """Point (5,0), segment (0,0)->(10,0) -> dist ~= 0."""
    dist, closest, normal = point_to_segment_distance(
        np.array([5.0, 0.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0])
    )
    assert dist < 1e-7
    # Normal should be perpendicular to segment
    assert abs(normal[0]) < 1e-10
    assert abs(abs(normal[1]) - 1.0) < 1e-10


def test_degenerate_segment():
    """Start==end -> distance to that point."""
    dist, closest, normal = point_to_segment_distance(
        np.array([3.0, 4.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])
    )
    assert abs(dist - 5.0) < 1e-10
    np.testing.assert_allclose(closest, [0.0, 0.0], atol=1e-10)
    # Normal points from origin toward (3,4)
    np.testing.assert_allclose(normal, [0.6, 0.8], atol=1e-10)


def test_agents_to_walls_shape(box_world):
    """10 agents, 4 walls -> distances shape (10,4)."""
    positions = np.random.default_rng(42).uniform(1, 9, (10, 2))
    dist, normals = agents_to_walls(positions, box_world.walls)
    assert dist.shape == (10, 4)
    assert normals.shape == (10, 4, 2)


def test_agent_near_wall():
    """Agent at (5,0.1) -> small distance to bottom wall."""
    walls = [Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0]))]
    positions = np.array([[5.0, 0.1]])
    dist, normals = agents_to_walls(positions, walls)
    assert abs(dist[0, 0] - 0.1) < 1e-10
    np.testing.assert_allclose(normals[0, 0], [0.0, 1.0], atol=1e-10)


def test_normal_direction():
    """Normal always points from wall toward agent."""
    walls = [Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0]))]
    # Agent above wall
    pos_above = np.array([[5.0, 2.0]])
    _, normals_above = agents_to_walls(pos_above, walls)
    assert normals_above[0, 0, 1] > 0  # y-component positive (wall below, agent above)

    # Agent below wall
    pos_below = np.array([[5.0, -2.0]])
    _, normals_below = agents_to_walls(pos_below, walls)
    assert normals_below[0, 0, 1] < 0  # y-component negative (agent below wall)
