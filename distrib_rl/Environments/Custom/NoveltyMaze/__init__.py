from gym import register
register(
    id='StanleyNovelty-v0',
    entry_point='Environments.Custom.NoveltyMaze.Environment.NoveltyMazeEnv:NoveltyMazeEnv',)
