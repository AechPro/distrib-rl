from .wordle import *
from .environment import Environment

from gym import register
register(
    id='Wordle-v0',
    entry_point='Environments.Custom.wordle.environment:Environment',)