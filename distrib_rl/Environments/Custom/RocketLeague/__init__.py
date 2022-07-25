from gym import register

id='RocketLeague-v0'
entry_point='distrib_rl.Environments.Custom.RocketLeague.RLGymEnvironment:RLGymEnvironment'

register(id=id, entry_point=entry_point)
