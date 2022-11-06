from gym.envs.registration import register

register(
    id='foragym/ForaGym-v0',
    entry_point='foragym.envs:ForaGym'
)

register(
    id='foragym/ForaGym-v1',
    entry_point='foragym.envs:ForaGym_with_threat'
)