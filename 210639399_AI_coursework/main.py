# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import networkx as nx
import csv
from queue import PriorityQueue
import heapq

def read_csv_file():
    with open('tubedata.csv') as file:
        graph = nx.Graph()
        tube_map = csv.reader(file, skipinitialspace=True)
        for row_value in tube_map:
            graph.add_edge(row_value[0], row_value[1], weight=float(row_value[3]))
        print(graph)
        return graph


def DFS(nxobject, initial, goal, compute_exploration_cost=False, reverse=False):
    frontier = [{'label': initial, 'parent': None}]
    explored = {initial}
    number_of_explored_nodes = 1
    while frontier:
        node = frontier.pop()  # pop from the right of the list\n",
        number_of_explored_nodes += 1
        if node['label'] == goal:
            if compute_exploration_cost:
                print('Number of explorations for DFS = {}'.format(number_of_explored_nodes))
            return node
        neighbours = reversed(list(nxobject.neighbors(node['label']))) if reverse else nxobject.neighbors(node['label'])
        for child_label in neighbours:
            child = {'label': child_label, 'parent': node}
            if child_label not in explored:
                frontier.append(child)  # added to the right of the list, so it is a LIFO\n",
                explored.add(child_label)
    return None


def BFS(nxobject, initial, goal, compute_exploration_cost=False, reverse=False):
    if initial == goal:
        return None
    number_of_explored_nodes = 1
    frontier = [{'label': initial, 'parent': None}]
    explored = {initial}
    while frontier:
        node = frontier.pop()  # pop from the right of the list\n",
        neighbours = reversed(list(nxobject.neighbors(node['label']))) if reverse else nxobject.neighbors(node['label'])
        for child_label in neighbours:
            child = {'label': child_label, 'parent': node}
            if child_label == goal:
                if compute_exploration_cost:
                    print('Number of explorations for BFS = {}'.format(number_of_explored_nodes))
                    return child
            if child_label not in explored:
                frontier = [child] + frontier  # added to the left of the list, so a FIFO!
                number_of_explored_nodes += 1
                explored.add(child_label)
    return None


class Node:
    def __init__(self, label, path_cost, parent):
        self.label = label
        self.path_cost = path_cost
        self.parent = parent

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def repr(self):
        path = self.construct_path(self)
        return ('(%s, %s, %s)', (repr(self.label), repr(self.path_cost), repr(path)))


def construct_path(node):
    path_from_root = [node.label]
    while node.parent:
        node = node.parent
        path_from_root = [node.label] + path_from_root
    return path_from_root


def remove_node_with_higher_cost(new_node, frontier):
    removed = False
    frontier_list = frontier.queue
    for item in frontier_list:
        if item.label == new_node.label and item.path_cost > new_node.path_cost:
            removed_item = item
            frontier_list.remove(item)
            removed = True
            break
    if removed:
        print("frontier = frontier - {} + {} ".format(removed_item, new_node))
        new_queue = PriorityQueue()
        frontier_list.append(new_node)
        for item in frontier_list:
            new_queue.put(item)
            return new_queue
    else:
            return frontier


def in_frontier(new_node, frontier):
    frontier_list = frontier.queue
    for item in frontier_list:
        if item.label == new_node.label:
            return True
    return False


def UCS(nxobject, initial, goal):
    node = Node(initial, 0, None)
    # frontier is a priority queue
    frontier = PriorityQueue()
    # add the initial state to the priority queue
    frontier.put(node)
    # explored is a set\n",
    explored = set()
    #print("frontier = ", frontier.queue)
    #print("explored = ", explored)
    while not frontier.empty():
        # pop the first element from the priority queue (lowest cost node)\n",
        node = frontier.get()
        #print("frontier = frontier - ", node.label)
        # check if the node is the goal state then return node\n",
        if node.label == goal:
            return node
            # else add the node to the explored set\n",
        explored.add(node.label)
        #print("explored = explored + ", node.label),
        # get all the neighbours of the node
        neighbours = nxobject.neighbors(node.label)
        for child_label in neighbours:
            step_cost = nxobject.edges[(node.label, child_label)]['weight']
            child = Node(child_label, node.path_cost + step_cost, node)
            # check if the child node is already explored or not
            if child_label not in explored and not in_frontier(child, frontier):
                frontier.put(child)
                #print("frontier = frontier + ", child.label)
            elif in_frontier(child, frontier):
                frontier = remove_node_with_higher_cost(child, frontier)

def heuristic(node):
    with open('tubedata.csv') as file:
        tube_map = csv.reader(file, skipinitialspace=True)
        for row_value in tube_map:
            x = row_value[3]
    return abs(int(x))

def Astar(graph, origin, goal):
    admissible_heuristics = {}
    h = heuristic(origin)
    admissible_heuristics[origin] = h
    visited_nodes = {} # This will contain the data of how to get to any node\n",
    visited_nodes[origin] = (h, [origin])
    paths_to_explore = PriorityQueue()
    paths_to_explore.put((h, [origin], 0))
    while not paths_to_explore.empty():
        #print(paths_to_explore.get())
        _,path, total_cost = paths_to_explore.get()
        current_node = path[-1]
        neighbors = graph.neighbors(current_node)
        for neighbor in neighbors:
            edge_data = graph.get_edge_data(path[-1], neighbor)
            if "weight" in edge_data:
                cost_to_neighbor = edge_data["weight"]
            else:
                cost_to_neighbor = 1
            if neighbor in admissible_heuristics:
                h = admissible_heuristics[neighbor]
            else:
                h = heuristic(neighbor)
                admissible_heuristics[neighbor] = h
            new_cost = total_cost + cost_to_neighbor
            new_cost_plus_h = new_cost + h
            if (neighbor not in visited_nodes) or (visited_nodes[neighbor][0]>new_cost_plus_h):
                next_node = (new_cost_plus_h, path+[neighbor], new_cost)
                visited_nodes[neighbor] = next_node
                paths_to_explore.put(next_node)
    print(visited_nodes)
    return 0


df=pd.read_csv('tubedata_headers.csv')
tubes_dict=df.to_dict('records')

def findline(node, child):
    crntline = None
    for i in tubes_dict:
        if i['StartingStation'] == node and i['EndingStation'] == child:
            return i['TubeLine']

    if crntline == None:
        for i in tubes_dict:
            if i['EndingStation'] == node and i['StartingStation'] == child:
                return i['TubeLine']


def UCSwithextendedcost(tube_graph, initial, goal, reverse=False):
    frontier = []  # frontier used is list of nodes
    frontierDict = {}
    """
      node is a tuple
      contains structure as (Total_Time_Taken, 'EndStation', ['List_of_Station_explored_from_Initial_to_reach_EndStation'])
    """
    node = (0, initial, [initial])

    """ 
      Use frontierDict a dictionary to keep track of the elements inside the frontier (queue)
      Contains structure as {'EndStation': [Total_Time_Taken, ['List_of_Station_explored_from_Initial_to_reach_EndStation']]}
    """
    frontierDict[node[1]] = [node[0], node[2]]

    heapq.heappush(frontier, node)  # insert the node inside the frontier (queue)
    explored = set()  # set of explored nodes

    TransitTime = 2  # Station Transit Time if the TubeLine has to be changed

    linelist = ['LineStart']

    while frontier:
        if len(frontier) == 0:  # if frontier is empty, return no solution
            return None

        node = heapq.heappop(frontier)  # pop elemenet with lower path cost in the queue
        del frontierDict[node[1]]  # delete the element that has been popped from frontierIndex

        if node[1] == goal:  # check if the solution has been found
            print(f'Number of explorations = {len(explored)}')
            return node
        explored.add(node[1])  # add node to explored

        # get a list of all the child nodes of current node
        neighbours = reversed(list(tube_graph.neighbors(node[1]))) if reverse else tube_graph.neighbors(node[1])
        path = node[2]  # path contains a list of all the nodes explored to reach current node

        for child in neighbours:
            path.append(child)

            current_line = findline(node[1], child)

            if current_line not in linelist:
                if current_line == linelist[-1]:
                    cost_child = node[0] + tube_graph.get_edge_data(node[1], child)["weight"]
                else:
                    if linelist[-1] == 'LineStart':
                        cost_child = node[0] + tube_graph.get_edge_data(node[1], child)["weight"]
                    cost_child = node[0] + tube_graph.get_edge_data(node[1], child)["weight"] + TransitTime
                    linelist.append(current_line)
            else:
                cost_child = node[0] + tube_graph.get_edge_data(node[1], child)["weight"]

            childNode = (cost_child, child, path)  # childNode that will be inserted in frontier

            # check if the child node is not explored and not in frontier through the dictionary
            if child not in explored and child not in frontierDict:
                heapq.heappush(frontier, childNode)
                frontierDict[child] = [childNode[0], childNode[2]]  # update the frontierDict
            elif child in frontierDict:
                # Checks if the child node has a lower path cost than the node already in frontier, if present replace it with lower cost node
                if childNode[0] < frontierDict[child][0]:
                    highCost_node = (frontierDict[child][0], child, frontierDict[child][1])
                    frontier.remove(highCost_node)  # if node with higher cost path exist, remove it from frontier
                    heapq.heapify(frontier)  # transforms frontier into heapq
                    del frontierDict[child]  # delete the node from frontierDict
                    heapq.heappush(frontier, childNode)  # insert the childNode in frontier
                    frontierDict[child] = [childNode[0], childNode[2]]  # update the frontierDict

            path = path[:-1]  # update the path if child is explored

def construct_path_from_root(node, root):
    path_from_root = [node['label']]
    while node['parent']:
        node = node['parent']
        path_from_root = [node['label']] + path_from_root
    return path_from_root

def compute_path_cost(graph, path):
    "  # Compute cost of a path\n",
    cost = 0
    for index_city in range(len(path) - 1):
        cost += graph[path[index_city]][path[index_city + 1]]["weight"]
    return cost

def construct_path(node):
    path_from_root = [node.label]
    while node.parent:
        node = node.parent
        path_from_root = [node.label] + path_from_root
    return path_from_root

if __name__ == '__main__':
    solution_dfs = DFS(read_csv_file(), 'Euston', 'Victoria', True)
    #print(solution_dfs)
    path_dfs = construct_path_from_root(solution_dfs, 'Euston')
    print(path_dfs)
    path_cost_dfs = compute_path_cost(read_csv_file(), path_dfs)
    print(path_cost_dfs)
    solution_bfs = BFS(read_csv_file(), 'Euston', 'Victoria', True)
    #print(solution_bfs)
    path_bfs = construct_path_from_root(solution_bfs, 'Euston')
    print(path_bfs)
    path_cost_bfs = compute_path_cost(read_csv_file(), path_bfs)
    print(path_cost_bfs)
    solution_ucs = UCS(read_csv_file(), 'Euston', 'Victoria')
    print(solution_ucs)
    path_ucs = construct_path(solution_ucs)
    print(path_ucs)
    path_cost_ucs = compute_path_cost(read_csv_file(), path_ucs)
    print(path_cost_ucs)
    solution=Astar(read_csv_file(), 'Baker Street', 'Wembley Park')
    print(solution)
    UCS_Extended_solution = UCSwithextendedcost(read_csv_file(), 'Euston', 'Victoria')
    #path_ucs_1= construct_path(UCS_Extended_solution)
    print("Cost "+str(UCS_Extended_solution[0]))
    print(UCS_Extended_solution[2])
    #path_cost_ucs = compute_path_cost(read_csv_file(), path_ucs_1)
    #print(path_cost_ucs)
    #print(UCS_Extended_solution)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
