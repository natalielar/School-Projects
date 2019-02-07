# breadth_first_search.py
"""Volume 2: Breadth-First Search.
<Name> Natalie Larsen
<Class> 001
<Date> 10-25-18
"""

from collections import deque
import networkx as nx
from matplotlib import pyplot as plt
from math import ceil

# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        keys = self.d.keys()
        #check for node in graph
        if n not in keys:
            self.d.update({str(n): set()})

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        keys = self.d.keys()
        #if nodes are not in graph, add them
        if u not in keys:
            self.add_node(u)
        if v not in keys:
            self.add_node(v)
        #add each node to the value set of each other
        u_old = self.d[u]
        u_new = u_old.union(set(str(v)))
        v_old = self.d[v]
        v_new = v_old.union(set(str(u)))
        self.d.update({u:u_new, v:v_new})


    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        keys = self.d.keys()
        #check for node in graph
        if n not in keys:
            raise KeyError(str(n) + " is not in graph")
        self.d.pop(n)
        #discard each occurence of node in the values of others
        for k in keys:
            edges = self.d[k]
            new = edges.discard(n)

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        elements = self.d.keys()
        #check for nodes in graph
        if u not in elements or v not in elements:
            raise KeyError(str(u) + " and " + str(v) + " must be in graph")
        #remove other node from each value set
        self.d[u].remove(v)
        self.d[v].remove(u)

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        key = self.d.keys()
        #check for source in graph
        if source not in key:
            raise KeyError(str(source) + " is not in graph!")
        #initialize V, Q and M
        V = []
        Q = deque()
        Q.append(source)
        M = set(source)
        #while Q is not empty
        while Q:
            #take first element of queue
            current = Q.popleft()
            #add it to V
            V.append(current)
            neighbors = self.d[current]
            #for each value associated with this key
            for n in neighbors:
                #if it isn't in M, add it to M and end of Q
                if n not in M:
                    Q.append(n)
                    M.add(n)
        return V



    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endpoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        key = self.d.keys()
        #check that endpoints are in graph
        if source not in key or target not in key:
            raise KeyError(str(source) + " and " + str(target) + " must be in graph")
        #initialize V,Q and M
        V = []
        vis = dict()
        Q = deque()
        Q.append(source)
        M = set(source)
        #while target has not been visited
        while target not in M:
            #take first element of Q
            current = Q.popleft()
            #add element to visited
            V.append(current)
            neighbors = self.d[current]
            #for each neighbor of element
            for n in neighbors:
                #if element has not been checked, add it to queue
                #also save traveled edge in visited
                if n not in M:
                    Q.append(n)
                    vis.update({n:current})
                    M.add(n)
        L = [target]
        #reverse the order of the traveled edges
        while L[-1] in vis.keys():
            L.append(vis[L[-1]])
        return L[::-1]

# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data_small.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        self.titles = set()
        self.actors = set()
        self.graph = nx.Graph()
        #read data from file
        with open(filename, encoding="utf8") as myfile:
            content = myfile.readlines()
        for c in content:
            #split title and each name
            cline = c.strip().split('/')
            #add title to title set and graph
            self.titles.add(cline[0])
            self.graph.add_node(cline[0])
            #for each actor in movie
            for cl in cline[1:]:
                #if actor is not already in graph, add them
                if cl not in self.actors:
                    self.actors.add(cl)
                    self.graph.add_node(cl)
                #create edge between actor and movie in graph
                self.graph.add_edge(cline[0],cl)




    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints.
            (int): the number of steps from source to target, excluding movies.
        """
        #shortest path
        path = nx.shortest_path(self.graph, source, target)
        #shortest path length....
        len_path = nx.shortest_path_length(self.graph, source, target)
        return path, len_path/2


    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        #find all shortest paths to target node
        lens = nx.shortest_path_length(self.graph, target=target)
        path_len = []
        #only include paths from actors in list
        for i in self.actors:
            path_len.append(lens[i]/2)

        #plot histogram
        plt.hist(path_len, bins=[i-.5 for i in range(8)])
        plt.title("Path Length Distribution")
        plt.xlabel("Path Length")
        plt.ylabel("Occurences")
        plt.show()

        return sum(path_len)/(len(path_len))


