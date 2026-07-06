def update_kl_loss_coef_from_reward(
    current_coef: float,
    reward_mean: float,
    eps: float,
    min_coef: float,
    max_coef: float,
) -> float:
    if reward_mean < 0:
        new_coef = current_coef * (1 - eps)
    elif reward_mean > 0:
        new_coef = current_coef * (1 + eps)
    else:
        new_coef = current_coef
    return max(min_coef, min(max_coef, new_coef))
