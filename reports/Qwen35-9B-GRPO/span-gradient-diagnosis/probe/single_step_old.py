"""Restrict current-forward PPO denominators to one accumulated policy update."""


SEPARATE_MODE = 'separate_pre_update_v1'
DETACHED_MODE = 'detached_current_single_step_v1'


def pre_update_probability_mode(requested, optimizer_phases):
    if requested not in (SEPARATE_MODE,DETACHED_MODE):
        raise ValueError(f'Unknown old probability mode: {requested}')
    if requested == DETACHED_MODE and tuple(optimizer_phases) != ('reward',):
        raise ValueError('Detached old probabilities require one reward optimizer step and no intervening update')
    return requested


def old_probability(current, frozen):
    return current.detach() if frozen is None else frozen
