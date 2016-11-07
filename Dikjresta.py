import sys
import networkx as nx
import matplotlib.pyplot as plt
import time

class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        # Set distance to infinity for all nodes
        self.distance = sys.maxint
        # Mark all nodes unvisited        
        self.visited = False  
        # Predecessor
        self.previous = None

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous

def shortest(v, path):
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return

import heapq

def dijkstra(aGraph, start):
    print '''Dijkstra's shortest path'''
    # Set the distance for the start node to zero 
    start.set_distance(0)

    # Put tuple pair into the priority queue
    unvisited_queue = [(v.get_distance(),v) for v in aGraph]
    heapq.heapify(unvisited_queue)

    while len(unvisited_queue):
        # Pops a vertex with the smallest distance 
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        #for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next)
            
            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
                print 'updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())
            else:
                print 'not updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())


        # Rebuild heap
        # 1. Pop every item
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        # 2. Put all vertices not visited into the queue
        unvisited_queue = [(v.get_distance(),v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)


if __name__ == '__main__':

    g = Graph()

    g.add_vertex('a')
    g.add_vertex('b')
    g.add_vertex('c')
    g.add_vertex('d')
    g.add_vertex('e')
    g.add_vertex('f')
    g.add_vertex('g')
    g.add_vertex('h')
    g.add_vertex('i')

    g.add_edge('a', 'b', 10)  
    g.add_edge('a', 'c', 9)
    g.add_edge('a', 'f', 14)
    g.add_edge('b', 'e', 1)
    g.add_edge('b', 'd', 15)
    g.add_edge('c', 'd', 11)
    g.add_edge('c', 'e', 15)
    g.add_edge('d', 'e', 6)
    g.add_edge('e', 'f', 9)
    g.add_edge('f', 'h', 23)
    g.add_edge('g', 'i', 30)
    g.add_edge('i', 'f', 2)
    g.add_edge('g', 'a', 8)

    gx = nx.Graph()
    
    print 'Graph data:'
    for v in g:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print '( %s , %s, %3d)'  % ( vid, wid, v.get_weight(w))
            nm = v.get_weight(w)
            gx.add_edge (vid, wid, weight = v.get_weight(w))

    # pos=nx.spring_layout(gx)
    # nx.draw_networkx_nodes(gx,pos,node_size=700)


    dijkstra(g, g.get_vertex('a')) 

    target = g.get_vertex('h')
    print target
    path = [target.get_id()]
    shortest(target, path)


    print 'The shortest path : %s' %(path[::-1])

    
    #plot step
    pos=nx.spring_layout(gx, scale=2)
    nx.draw_networkx_nodes(gx,pos,node_size=700, k=0.15,iterations=20)
    ebase=[(u,v) for (u,v,d) in gx.edges(data=True) if d['weight'] >0.5]
    nx.draw_networkx_edges(gx, pos, edgelist=ebase, width=6)
    nx.draw_networkx_labels(gx,pos,font_size=20,font_family='sans-serif')
    labels = nx.get_edge_attributes(gx,'weight')
    nx.draw_networkx_edge_labels(gx,pos,edge_labels=labels)

    plt.axis('off')
    plt.show() # display

    rpath = path[::-1]

    for x in rpath:
    	nx.draw_networkx_nodes(gx,pos,node_size=700)
    	nx.draw_networkx_labels(gx,pos,font_size=20,font_family='sans-serif')
    	nx.draw_networkx_edges(gx, pos, edgelist=ebase, width=3,alpha=0.1)

    	epath=[(u,v) for (u,v,d) in gx.edges(data=True) if u == x or v == x]

    	nx.draw_networkx_edges(gx,pos,edgelist=epath,width=6,edge_color='g',style='dashed')
    	labels = nx.get_edge_attributes(gx,'weight')
    	nx.draw_networkx_edge_labels(gx,pos,edge_labels=labels, edgelist=epath)

    	plt.axis('off')
        plt.show() # display




    #plot
    #pos=nx.spring_layout(gx)
    nx.draw_networkx_nodes(gx,pos,node_size=700, scale = 10)
    elarge=[(u,v) for (u,v,d) in gx.edges(data=True) if not u in path or not v in path]
    esmall=[(u,v) for (u,v,d) in gx.edges(data=True) if u in path and v in path]

    nx.draw_networkx_nodes(gx,pos,node_size=700)
    nx.draw_networkx_edges(gx, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(gx,pos,edgelist=esmall,
                    width=6,alpha=0.5,edge_color='g',style='dashed')
    nx.draw_networkx_labels(gx,pos,font_size=20,font_family='sans-serif')
 
    labels = nx.get_edge_attributes(gx,'weight')
    nx.draw_networkx_edge_labels(gx,pos,edge_labels=labels)

    plt.axis('off')
    #plt.savefig("weighted_graph.png") # save as png
    plt.show() # display
    
     


