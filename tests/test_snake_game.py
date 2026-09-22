import json

import pytest

from laya_onnx.snake.game import DELTAS, DIRECTIONS, SnakeGame

DELTA_TO_DIR = {delta: name for name, delta in DELTAS.items()}


def neck_direction(game):
    (hx, hy), (nx, ny) = game.body[0], game.body[1]
    return DELTA_TO_DIR[(nx - hx, ny - hy)]


def test_init_validation():
    with pytest.raises(ValueError):
        SnakeGame(3, 10, 1, 4)  # width too small
    with pytest.raises(ValueError):
        SnakeGame(10, 3, 1, 4)  # height too small
    with pytest.raises(ValueError):
        SnakeGame(10, 10, 1, 1)  # initial_length too small
    with pytest.raises(ValueError):
        SnakeGame(10, 10, 1, 100)  # initial_length >= area


def test_body_cells_are_distinct_even_when_long():
    game = SnakeGame(4, 4, 1, 15)
    assert len(game.body) == 15
    assert len(set(game.body)) == 15


def test_snapshot_is_json_serializable():
    game = SnakeGame(12, 9, 3, 4)
    assert '"width": 12' in json.dumps(game.snapshot())


def test_same_seed_is_deterministic():
    a = SnakeGame(12, 10, seed=5, initial_length=4)
    b = SnakeGame(12, 10, seed=5, initial_length=4)
    assert a.food == b.food
    for _ in range(8):
        if not a.alive:
            break
        move = next(d for d in DIRECTIONS if a.legal(d))
        a.step(move)
        b.step(move)
        assert a.food == b.food
    assert a.snapshot() == b.snapshot()


def test_reverse_into_neck_is_illegal():
    game = SnakeGame(10, 10, 1, 4)
    assert not game.legal(neck_direction(game))
    assert any(game.legal(d) for d in DIRECTIONS)


def test_wall_death():
    game = SnakeGame(8, 8, 1, 4)
    forward = next(d for d in DIRECTIONS if game.legal(d) and d != neck_direction(game))
    for _ in range(64):
        if not game.alive:
            break
        game.step(forward)
    assert not game.alive


def test_eat_grows_and_scores():
    game = SnakeGame(10, 10, 1, 4)
    neck = neck_direction(game)
    direction = next(d for d in DIRECTIONS if game.legal(d) and d != neck)
    dx, dy = DELTAS[direction]
    hx, hy = game.body[0]
    game.food = (hx + dx, hy + dy)
    assert game.step(direction) == "eat"
    assert game.score == 1
    assert len(game.body) == 5


def test_win_when_board_filled():
    game = SnakeGame(4, 4, 1, 15)  # 15 cells + 1 food = full board
    assert game.food is not None
    neck = neck_direction(game)
    direction = next(d for d in DIRECTIONS if game.legal(d) and d != neck)
    game.food = tuple(
        game.body[0][axis] + DELTAS[direction][axis] for axis in (0, 1)
    )
    assert game.step(direction) == "eat"
    assert game.won
    assert len(game.body) == 16
    assert game.snapshot()["won"] is True


def test_invalid_direction_raises():
    game = SnakeGame(10, 10, 1, 4)
    with pytest.raises(ValueError):
        game.step("SIDEWAYS")
    assert set(DIRECTIONS) == {"UP", "DOWN", "LEFT", "RIGHT"}
