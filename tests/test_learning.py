import numpy as np
import pytest

from robot_dynamics.learning_data import encode_sample
from robot_dynamics.windows import HISTORY_FIELDS, FIXTURE_FIELDS, window


def test_horizon_input_excludes_later_actions_and_targets():
    episode = {k: np.ones((20, 3)) for k in HISTORY_FIELDS}
    episode.update({k: np.ones(3) for k in FIXTURE_FIELDS})
    episode["time"] = np.arange(20) * 0.05
    episode["actions"] = np.arange(140, dtype=float).reshape(20, 7)
    stats = {"position_center": [0, 0, 0], "position_scale": 2, "scalar_fields": {}}
    original = encode_sample(window(episode, 5, 4, 1), stats, 1)
    episode["actions"][6:] += 999
    episode["nut_position"][6:] += 999
    changed = encode_sample(window(episode, 5, 4, 1), stats, 1)
    np.testing.assert_array_equal(original[0], changed[0])
    np.testing.assert_array_equal(original[1], changed[1])
    assert not np.array_equal(original[2], changed[2])
    assert np.count_nonzero(original[1][-73:-3][7:]) == 0


@pytest.mark.parametrize("kind", ["current_mlp", "history_mlp", "gru"])
def test_models_have_finite_gradients_and_current_model_ignores_past(kind):
    torch = pytest.importorskip("torch")
    from robot_dynamics.learning_models import DynamicsPredictor

    torch.manual_seed(0)
    model = DynamicsPredictor(kind)
    state = torch.randn(8, 4, 42)
    context = torch.randn(8, 121)
    output = model(state, context)
    assert output.shape == (8, 6)
    output.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    if kind == "current_mlp":
        changed = state.clone()
        changed[:, :-1] += 100
        torch.testing.assert_close(model(changed, context), output, rtol=0, atol=0)
