class EquationTreeNode:
    father = None
    node = None
    children = []

    def __init__(self, father, node, children):
        self.father = father
        self.node = node
        self.children = children