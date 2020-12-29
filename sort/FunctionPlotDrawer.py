from utils import EquationTreeNode as TreeNode
import math
import numpy as np
import matplotlib.pyplot as plt


def compute_node_value(node: TreeNode.EquationTreeNode, current_x_val: float):
    node_value = 0.
    number_of_children = len(node.children)

    if number_of_children == 1:
        # it is an operator

        # computing node child
        child_value = compute_node_value(node.children[0], current_x_val)

        # computing operation result
        if node.node == 'sqrt':
            node_value = math.sqrt(child_value)
        if node.node == 'log':
            node_value = math.log(child_value)
    elif number_of_children == 0:
        # it is a variable
        if node.node != 'x':
            node_value = float(node.node)
        else:
            node_value = current_x_val
    elif number_of_children == 2:
        # it is a binary operator
        child_1_val = compute_node_value(node.children[0], current_x_val)
        child_2_val = compute_node_value(node.children[1], current_x_val)

        if node.node == '+':
            node_value = child_1_val + child_2_val
        if node.node == '-':
            node_value = child_2_val - child_1_val
        if node.node == 'div':
            node_value = child_2_val / child_1_val
        if node.node == '*':
            node_value = child_1_val * child_2_val

    return node_value


def draw_equation(equation_tree):
    x_values = np.arange(0., 100.5, 0.5)
    y_values = []
    for x in x_values:
        equation_val = compute_node_value(equation_tree, x)
        y_values.append(equation_val)
    plt.plot(x_values, y_values)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    plt.show()