from gym.envs.registration import register
register(
    id='cdtn-ASCEND2020-v0',
    entry_point='gym_cdtn.envs:CdtnEnvDiscrete',
)

register(
    id='cdtn-JAIS2021-v0',
    entry_point='gym_cdtn.envs:CdtnEnvContinuous',
)

register(
    id='cdtn-prioritiesRL-v0',
    entry_point='gym_cdtn.envs:CdtnEnvContinuousPrioritiesFullRL',
)

register(
    id='cdtn-prioritiesHybrid-v0',
    entry_point='gym_cdtn.envs:CdtnEnvContinuousPrioritiesHybrid',
)

register(
    id='cdtn-EO-v0',
    entry_point='gym_cdtn.envs:CdtnEnvEO',
)