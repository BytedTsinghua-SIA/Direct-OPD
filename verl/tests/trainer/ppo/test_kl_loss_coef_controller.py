import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[3] / "verl" / "trainer" / "ppo" / "kl_controller.py"
_SPEC = importlib.util.spec_from_file_location("kl_controller", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_KL_CONTROLLER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_KL_CONTROLLER)
update_kl_loss_coef_from_reward = _KL_CONTROLLER.update_kl_loss_coef_from_reward


def test_update_kl_loss_coef_from_reward_relaxes_on_negative_reward():
    new_coef = update_kl_loss_coef_from_reward(
        current_coef=2.0,
        reward_mean=-0.001,
        eps=0.02,
        min_coef=0.5,
        max_coef=2.0,
    )

    assert new_coef == 1.96


def test_update_kl_loss_coef_from_reward_tightens_on_positive_reward():
    new_coef = update_kl_loss_coef_from_reward(
        current_coef=1.0,
        reward_mean=0.001,
        eps=0.02,
        min_coef=0.5,
        max_coef=2.0,
    )

    assert new_coef == 1.02


def test_update_kl_loss_coef_from_reward_clamps_to_bounds():
    assert (
        update_kl_loss_coef_from_reward(
            current_coef=0.51,
            reward_mean=-0.001,
            eps=0.02,
            min_coef=0.5,
            max_coef=2.0,
        )
        == 0.5
    )
    assert (
        update_kl_loss_coef_from_reward(
            current_coef=1.99,
            reward_mean=0.001,
            eps=0.02,
            min_coef=0.5,
            max_coef=2.0,
        )
        == 2.0
    )
