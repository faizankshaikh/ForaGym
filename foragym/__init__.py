from gym.envs.registration import register

register(
    id='foragym/ForaGym-v0',
    entry_point='foragym.envs:ForaGym'
)