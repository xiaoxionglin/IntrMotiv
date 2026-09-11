"""Keep physical final observations distinct from same-step reset outputs."""
import copy


def reset_with_final(env, observation, info):
    """Validity must be certified by the environment, never inferred from shape.

    DMLab historically returns the previous image after shutdown; that image
    can have the correct shape without representing the final physical state.
    """
    valid = bool(info.get('intrmotiv_final_observation_valid', False))
    final_observation = copy.deepcopy(observation) if valid else None
    final_info = copy.deepcopy(info)
    reset_observation, reset_info = env.reset()
    reset_info = dict(reset_info)
    reset_info.update(final_info=final_info, final_observation=final_observation,
                      final_observation_valid=valid)
    return reset_observation, reset_info
