from .Node import Node
from .TileMap import TileMap
from .Environment import Environment
from gym import register

register(
    id="CustomNovelty-v0",
    entry_point="Environments.Custom.Novelty.Environment:Environment",
)
