from gym import register
register(
    id='TwitchChatBetting-v0',
    entry_point='Environments.Custom.TwitchChatBetting.BettingEnv:BettingEnv',)
