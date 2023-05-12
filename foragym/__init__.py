from gymnasium.envs.registration import register

register(
    id='foragym/ForaGym-v0',
    entry_point='foragym.envs:ForaGym',
    max_episode_steps=300
)

register(
    id='foragym/ForaGym-v1',
    entry_point='foragym.envs:ForaGym_with_threat',
    max_episode_steps=300
)