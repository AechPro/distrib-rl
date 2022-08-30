from .node import Node
from .tile_map import TileMap
from .environment import Environment
from gym import register

register(
    id="CustomNovelty-v0",
    entry_point="distrib_rl.environments.custom.novelty.environment:Environment",
)
