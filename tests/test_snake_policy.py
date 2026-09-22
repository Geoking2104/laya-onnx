from laya_onnx.snake.game import DELTAS, DIRECTIONS, SnakeGame
from laya_onnx.snake.policy import LayaPolicy

DELTA_TO_DIR = {delta: name for name, delta in DELTAS.items()}


def neck_direction(game):
    (hx, hy), (nx, ny) = game.body[0], game.body[1]
    return DELTA_TO_DIR[(nx - hx, ny - hy)]


class StubAgent:
    providers = "cpu"

    def __init__(self, choice, probabilities=None):
        self.choice = choice
        self.probabilities = probabilities or {}
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {
            "answers": {
                "move": {
                    "type": "choice",
                    "choice": self.choice,
                    "probabilities": self.probabilities,
                    "confidence": 0.5,
                }
            }
        }


def make_policy(agent, *, guarded=True, prompt="compact"):
    """Build a policy without touching the Hub (bypasses LayaPolicy.__init__)."""
    policy = LayaPolicy.__new__(LayaPolicy)
    policy.model_id = "stub"
    policy.guarded = guarded
    policy.prompt = prompt
    policy.optimize = False
    policy.agent = agent
    policy.metadata = {
        "hardware": "test",
        "model": "stub",
        "providers": "cpu",
        "prompt": prompt,
        "optimized": False,
    }
    return policy


def test_guarded_intervenes_on_illegal_move():
    game = SnakeGame(10, 10, 1, 4)
    neck = neck_direction(game)
    probabilities = {direction: 0.1 for direction in DIRECTIONS}
    probabilities[neck] = 0.9
    decision = make_policy(StubAgent(neck, probabilities), guarded=True).decide(game)
    assert decision.chosen == neck
    assert decision.executed != neck
    assert game.legal(decision.executed)
    assert decision.intervened == 1
    assert decision.to_dict()["intervened"] is True


def test_unassisted_passes_illegal_through():
    game = SnakeGame(10, 10, 1, 4)
    neck = neck_direction(game)
    decision = make_policy(StubAgent(neck, {neck: 1.0}), guarded=False).decide(game)
    assert decision.executed == neck
    assert decision.intervened == 0


def test_legal_choice_is_not_intervened():
    game = SnakeGame(10, 10, 1, 4)
    neck = neck_direction(game)
    safe = next(d for d in DIRECTIONS if game.legal(d) and d != neck)
    decision = make_policy(StubAgent(safe, {safe: 0.9}), guarded=True).decide(game)
    assert decision.executed == safe
    assert decision.intervened == 0


def test_questions_cover_all_directions():
    game = SnakeGame(10, 10, 1, 4)
    agent = StubAgent("UP")
    make_policy(agent).decide(game)
    state, questions = agent.calls[0]
    assert isinstance(state, str)
    assert questions["move"]["type"] == "choice"
    assert set(questions["move"]["criteria"]) == {"UP", "DOWN", "LEFT", "RIGHT"}


def test_to_dict_is_json_ready():
    import json

    game = SnakeGame(10, 10, 1, 4)
    agent = StubAgent("UP", {"UP": 0.7})
    payload = json.dumps(make_policy(agent).decide(game).to_dict())
    assert '"inference_ms"' in payload


def test_detailed_prompt_renders_grid():
    game = SnakeGame(6, 6, 1, 3)
    agent = StubAgent("UP")
    make_policy(agent, prompt="detailed").decide(game)
    state, _ = agent.calls[0]
    assert "\n" in state and "#" in state
