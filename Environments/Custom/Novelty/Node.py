class Node(object):
    def __init__(self,x,y,id,walkable):
        self.x = x
        self.y = y
        self.id = id
        self.walkable = walkable
        self.links = []

    def take_action(self, action):
        if action >= len(self.links) or action < 0:
            return self
        if not self.links[action].walkable:
            return self
        return self.links[action]

    def add_link(self, other):
        self.links.append(other)