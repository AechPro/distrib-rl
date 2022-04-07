from Environments.Custom.Novelty import Node


class TileMap(object):
    def __init__(self):
        self.node_radius = 7
        self.nodes = []
        self.width = 0
        self.height = 0

    def build_links(self):
        for i in range(self.height):
            for j in range(self.width):
                target = self.nodes[i][j]
                for x in range(3):
                    for y in range(3):
                        tx = (j - 1) + x
                        ty = (i - 1) + y
                        if 0 <= tx < len(self.nodes[i]) and 0 <= ty < len(self.nodes):
                            n = self.nodes[ty][tx]
                            target.add_link(n)
                        else:
                            target.add_link(target)

    def load_map(self, file_path):
        self.nodes = []
        self.width = 0
        self.height = 0
        y = 0
        id = 0
        delim = " "

        with open(file_path, 'r') as f:
            for line in f:
                row = []
                tiles = line.split(delim)
                x = 0
                self.width = 0
                for tile in tiles:
                    walkable = tile == "0"
                    node = Node(x, y, id, walkable)
                    row.append(node)
                    id += 1
                    x += 1 * self.node_radius
                    self.width += 1

                self.nodes.append(row)
                y += 1 * self.node_radius
                self.height += 1

    def get_node(self, x, y):
        x = x // self.node_radius
        y = y // self.node_radius
        if len(self.nodes[0]) > x >= 0 and len(self.nodes) > y >= 0:
            return self.nodes[y][x]
        return None

    def cleanup(self):
        del self.nodes
        self.nodes = None
