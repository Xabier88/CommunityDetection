# SQL
import sqlite3

# Pandas
import pandas as pd

# Graph
import community
import networkx as nx

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

# Combinations
import itertools

# Work with lists.
import numpy as np

# Generate randomness.
import random as rm

# Measure execution time.
import time as tm

# Work with queues.
import collections as co

#Function that builds the networkx graph.
def sortu_grafoa():
    
    # Read data
    connect = sqlite3.connect('./dataset/database.sqlite')
    query = """
    SELECT pa.paper_id, pa.author_id, a.name
    FROM paper_authors AS pa JOIN papers AS p ON pa.paper_id = p.id
    JOIN authors as a ON pa.author_id = a.id
    WHERE p.Year BETWEEN '2014' AND '2015'
    """
    df = pd.read_sql(query, connect)
    
    # Initialize graph
    G = nx.Graph()

    # Transform
    # We use the name of the author instead of the id.
    for p, a in df.groupby('paper_id')['name']: 
        for u, v in itertools.combinations(a, 2):
            # We ignore the self-connections.
            if u != v:
                if G.has_edge(u, v):
                    G[u][v]['weight'] +=1
                else:
                    G.add_edge(u, v, weight=1)
    return G

#Computes the modularity of a partition in an efficient way.
def modularity(partition, graph, links, degree, weight='weight'):
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")
    inc = dict([])
    deg = dict([])
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")
    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + degree[node]
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.
    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res

######## RANDOM SEARCH ALGORITHM ########
class Random_Search_Algorithm:
    
    ################ Random search procedure ################
    
    def random_search(self):
        start = tm.time()
        dict_degrees = dict(zip(self.G.nodes,self.node_degrees))
        #Initial random solution
        best_sol = [rm.randint(0,self.communities-1) for i in range(len(self.G))]
        best_act = modularity(dict(zip(self.G.nodes, best_sol)),self.G,self.weight,dict_degrees)
        evals = 1
        #Until we have consumed all the allowed evaluations
        while evals < self.max_evals:
            #Create a random solution
            sol = [rm.randint(0,self.communities-1) for i in range(len(self.G))]
            act = modularity(dict(zip(self.G.nodes, sol)),self.G,self.weight,dict_degrees)
            evals += 1
            #If there is an improvement, we update the best solution
            if act > best_act:
                best_sol = sol
                best_act = act
        end = tm.time()
        return best_sol,best_act,evals,end-start
    
    ################ Initialize the global atributtes ################
    
    def __init__(self,communities,max_evals):
        #Graph
        self.G = sortu_grafoa()
        #Total weight of the edges in the graph
        self.weight = self.G.size(weight="weight")
        #Node degrees
        self.node_degrees = [self.G.degree(node,weight="weight") for node in self.G]
        #Maximum number of allowed communities. Precondition: Communities > 0
        self.communities = communities
        #Allowed number of evaluations. Precondition: Max_evals > 0
        self.max_evals = max_evals

######## ONE SOLUTION ALGORITHM ########
class One_Solution_Algorithm:

    ################ Initialize local structures ################

    ### Initialize the degree of each community (sum of the degrees of the nodes in the community)
    def create_deg(self,sol):
        deg = [0 for i in range(self.communities)]
        for node in range(len(sol)):
              deg[sol[node]] += self.node_degrees[node]
        return deg

    ### Initialize the community to community (f1) and the node to community (f2) structures
    def create_f(self,sol):
        #Community-community: Stores the sum of the weights of the links between two communities
        f1 = [[0 for i in range(self.communities)] for j in range(self.communities)]
        #Node-Community: Stores the sum of the weights of the links between a node and a community
        f2 = [[0 for i in range(self.communities)] for j in range(len(self.G))]
        for node1 in range(len(sol)):
            for node2 in range(len(sol)):
                edge = self.G.get_edge_data(self.names[node1],self.names[node2],default={"weight":0})["weight"]
                f1[sol[node1]][sol[node2]] += edge
                f2[node1][sol[node2]] += edge
        return f1,f2  

    ################ Compute modularity improvement ################

    ### Modularity improvement for local moving step
    def improve_lm(self,deg,f2,node,C,D):
        f_V = self.weight*2
        deg_n = self.node_degrees[node]
        return ((2*f2[node][D]-2*f2[node][C])/f_V)-((2*deg_n*deg[D]-2*deg_n*(deg[C]-deg_n))/f_V**2)

    ### Modularity improvement for cluster joining step
    def improve_cj(self,deg,f1,C,D):
        f_V = self.weight*2
        return ((2*f1[C][D])/f_V-(2*deg[C]*deg[D])/f_V**2)
      
    ################ Update structures ################

    ### Update structures after local moving modification
    def update_lm(self,deg,f1,f2,node,my_com,new_com):
        my_name = self.names[node]
        #We update the degree of the previous and the new community
        degree = self.node_degrees[node]
        deg[my_com] -= degree
        deg[new_com] += degree
        #For every neighbor community of the node that we have modified, we update the community to community "f1" structure
        for com in [i for i in range(len(f2[node])) if f2[node][i]>0]:
            f1[com][my_com] -= f2[node][com]
            f1[my_com][com] -= f2[node][com]
            f1[com][new_com] += f2[node][com]
            f1[new_com][com] += f2[node][com]
        #For every neighbor node of the modified node, we update the node to community "f2" structure
        for neighbor in self.neighbors[node]:
            edge = self.G.get_edge_data(my_name,self.names[neighbor],default={"weight":0})["weight"]
            f2[neighbor][my_com] -= edge
            f2[neighbor][new_com] += edge

    ### Update structures after cluster joining modification
    def update_cj(self,deg,f1,f2,com1,com2):
        #We update the degree of the joined communities (One of them disappears)
        deg[com2] += deg[com1]
        deg[com1] = 0
        #For every neighbor community of the community that has disappeared, we update the community to community "f1" structure
        for com in [i for i in range(len(f1)) if i!=com1 and f1[i][com1]>0]:
            f1[com][com2] += f1[com][com1]
            f1[com2][com] += f1[com][com1]
            f1[com][com1] = 0
            f1[com1][com] = 0
        #We update the self "f1" of the joined communities
        f1[com2][com2] += f1[com1][com1]
        f1[com1][com1] = 0
        #For every neighbor node of the community that has disappeared, we update the node to community "f2" structure
        for neighbor in [i for i in range(len(f2)) if f2[i][com1]>0]:
            f2[neighbor][com2] += f2[neighbor][com1]
            f2[neighbor][com1] = 0

    ################ Perform one step of local search ################

    ### Search for local moving steps that improve the modularity
    def local_moving(self,deg,f1,f2,best_sol,best_act,evals):
        better = False
        order1 = [i for i in range(len(best_sol))]
        rm.shuffle(order1)
        #We consider all the nodes in a random order
        for node in order1:
            my_com = best_sol[node]
            order2 = [i for i in set(best_sol) if i!=my_com and f2[node][i]>0]
            rm.shuffle(order2)
            #We consider all the communities in a random order
            for new_com in order2:
                #We check if we have exceed the maximum number of allowed evaluations
                if evals >= self.max_evals:
                    return best_sol,best_act,False,evals
                #We compute the modularity improvement obtained from moving the node "n" to the community "c"
                imp = self.improve_lm(deg,f2,node,my_com,new_com)
                evals += 1
                #If the modularity improves, we update the solution and we continue with the next node
                if imp > 0:
                    self.update_lm(deg,f1,f2,node,my_com,new_com)
                    best_sol[node] = new_com
                    best_act = best_act + imp
                    better = True
                    break
        return best_sol,best_act,better,evals

    ### Search for a cluster joining step that improves the modularity
    def cluster_joining(self,deg,f1,f2,best_sol,best_act,evals):
        order1 = list(set(best_sol))
        rm.shuffle(order1)
        #We consider all the communities in a random order
        for com1 in order1:
            order2 = [i for i in set(best_sol) if i!=com1 and f1[i][com1]>0]
            rm.shuffle(order2)
            #We consider all the communities in a random order
            for com2 in order2:
                #We check if we have exceed the maximum number of allowed evaluations
                if evals >= self.max_evals:
                    return best_sol,best_act,False,evals
                #We compute the modularity improvement obtained from merging the community "c1" and the community "c2"
                imp = self.improve_cj(deg,f1,com1,com2)
                evals += 1
                #If the modularity improves, we update the solution and we stop
                if imp > 0:
                    self.update_cj(deg,f1,f2,com1,com2)
                    for i in range(len(best_sol)):
                        if best_sol[i]==com1:
                            best_sol[i]=com2
                    best_act = best_act + imp
                    return best_sol,best_act,True,evals
        return best_sol,best_act,False,evals

    ################ Perform variable neighborhood search ################

    def vns(self,best_sol,best_act,evals):
        #Initialize the deg, f1 and f2 structures
        deg = self.create_deg(best_sol)
        f1, f2 = self.create_f(best_sol)
        first_neigh = True
        not_improve = 0
        #Until we can't improve more in any of the neighborhoods or we have consumed all the allowed evaluations
        while not_improve<2 and evals < self.max_evals:
            start_act = best_act
            if first_neigh:
                #Perform local search in first neighborhood (Local moving)
                ls_continue = True
                while ls_continue:
                    best_sol,best_act,ls_continue,evals = self.local_moving(deg,f1,f2,best_sol,best_act,evals)
            else:
                #Perform local search in second neighborhood (Cluster joining)
                ls_continue = True
                while ls_continue:
                    best_sol,best_act,ls_continue,evals = self.cluster_joining(deg,f1,f2,best_sol,best_act,evals)
            #If there is no improvement, we add one to the "not improvement" indicator
            if start_act == best_act:
                not_improve += 1
            #If there is an improvement, we set the "not improvement" indicator to one (This is done to avoid exploring
            #the same neighborhood twice)
            else:
                not_improve = 1
            #We change to the other neighborhood
            first_neigh = not first_neigh
        return best_sol,best_act,evals

    ################ Perturb solutions ################

    def prob(self,size_com):
        probability = [val**self.probability_exponent for val in size_com]
        total = sum(probability)
        return [val/total for val in probability]
        
    def perturbation(self,sol):
        #We consider all the non-empty communities
        used_coms = list(set(sol))
        #We compute the probability of selecting each non-empty community
        probability = self.prob([sol.count(com) for com in used_coms])
        #We compute the number of communities that have to be perturbed
        size = min(self.num_perturbation,len(used_coms))
        #We select "size" non-empty communities
        perturbed_coms = list(np.random.choice(used_coms, p=probability, size=size, replace=False))
        #We assign each node in the selected communities to a random community
        for i in range(len(sol)):
            if sol[i] in perturbed_coms:
                sol[i] = rm.randint(0,self.communities-1)
        return sol

    ################ Iterative VNS procedure ################

    def iterative_vns(self):
        start = tm.time()
        dict_degrees = dict(zip(self.G.nodes,self.node_degrees))
        #We create a random initial solution
        initial_sol = [rm.randint(0,self.communities-1) for i in range(len(self.G))]
        initial_act = modularity(dict(zip(self.G.nodes, initial_sol)),self.G,self.weight,dict_degrees)
        #We perform an initial VNS
        best_sol,best_act,evals = self.vns(initial_sol,initial_act,1)
        #Until we have consumed all the allowed evaluations
        while evals < self.max_evals:
            #We perturb the best solution obtained until now
            initial_sol = self.perturbation(best_sol.copy())
            initial_act = modularity(dict(zip(self.G.nodes, initial_sol)),self.G,self.weight,dict_degrees)
            evals += 1
            #We perform a VNS taking into account the perturbed solution
            sol,act,evals = self.vns(initial_sol,initial_act,evals)
            #If there is an improvement, we update the best solution
            if best_act < act:
                best_sol = sol
                best_act = act
        end = tm.time()
        return best_sol,best_act,evals,end-start

    ################ Initialize the global atributtes ################
    
    def __init__(self,communities,max_evals,num_perturbation,probability_exponent):
        #Graph
        self.G = sortu_grafoa()
        #Names of the nodes
        self.names = list(self.G.nodes)
        #Total weight of the edges in the graph
        self.weight = self.G.size(weight="weight")
        #Node degrees
        self.node_degrees = [self.G.degree(node,weight="weight") for node in self.G]
        #Neigbors of each node
        self.neighbors = [[k for k in range(len(self.G)) if self.G.get_edge_data(self.names[j],self.names[k],default={"weight":0})["weight"]>0] for j in range(len(self.G))]
        #Maximum number of allowed communities. Precondition: Communities > 0
        self.communities = communities
        #Allowed number of evaluations. Precondition: Max_evals > 0
        self.max_evals = max_evals
        #Number of communities that are perturbed in each iteration of the IVNS. 
        #Precondition: 0 < Num_pertubation <= Communities
        self.num_perturbation = num_perturbation
        #Exponent applied to the size of each community when computing the probabilities of being pertubed.
        #Precondition: Probability_exponent >= 0
        self.probability_exponent = probability_exponent

######## POPULATION BASED ALGORITHM ########
class Population_Based_Algorithm:

    ################ Random initialization of initial population ################
    
    def initialize(self):
        #Check if we will exceed the maximum number of evaluations and adjust the population size if true.
        if self.pop_size > self.max_evals:
            pop_size = self.max_evals
        else:
            pop_size = self.pop_size
        #Each gene can only be associated to its label or the label of one of its neighbors
        indexes = [np.random.choice(a=[i]+self.neighbors[i], size=pop_size, replace=True) for i in range(len(self.G))]
        pop = [[indexes[node][individual] for node in range(len(self.G))] for individual in range(pop_size)]
        return pop                 

    ################ Create individuals based in the given probability distribution ################
    
    def create_individuals(self,distribution,remaining_evals):
        #Check if we will exceed the maximum number of evaluations and adjust the population size if true.
        if self.pop_size > remaining_evals:
            pop_size = remaining_evals
        else:
            pop_size = self.pop_size
        #Each gene can only be associated to its label or the label of one of its neighbors
        indexes = [np.random.choice(a=[j for j in [i]+self.neighbors[i]],p=distribution[i],size=pop_size,replace=True) for i in range(len(self.G))]
        individuals = [[indexes[node][individual] for node in range(len(self.G))] for individual in range(pop_size)]
        return individuals  

    ################ Translate individuals from locus-based adjacency representation ################
    
    ### Construct the community graphs that represent the individuals and translate them
    def translate(self,individuals):
        translation = []
        for individual in range(len(individuals)):
            #First, we construct the graph that represents the individual
            individual_graph = [[] for j in range(len(self.G))]
            for node in range(len(self.G)):
                #We ignore self loops
                if individuals[individual][node] != node:
                    individual_graph[node].append(individuals[individual][node])
                    individual_graph[individuals[individual][node]].append(node)
            #When translating the solution, we take into account the maximum number of allowed communities
            translation.append(self.detect_communities(individual_graph))
        return translation

    ### Detect the connected components of a community graph and translate the solution to our solution space
    ### (Taking into account the maximum number of allowed communities)
    def detect_communities(self,individual_graph):
        individual = [None for i in range(len(self.G))]
        current_com = 0
        #For each individual that doesn't have an associated community, we perform a BFS
        for node in range(len(self.G)):
            if individual[node] == None:
                q = co.deque()
                q.appendleft(node)
                visited = set([node])
                while len(q)>0:
                    current_node = q.pop()
                    individual[current_node] = current_com
                    for adj_node in individual_graph[current_node]:
                        if adj_node not in visited:
                            visited.add(adj_node)
                            q.appendleft(adj_node)
                #If the current community isn't the last one, we increase the community label
                if current_com < self.communities-1:
                    current_com += 1
                #If the current community is the last one, we reset the community label to zero
                else:
                    current_com = 0
        return individual

    ################ Evaluate the translated solutions using modularity ################
    
    def evaluate(self,individuals,dict_degrees,evals):
        return [modularity(dict(zip(self.G.nodes, individuals[i])),self.G,self.weight,dict_degrees) for i in range(len(individuals))],evals+len(individuals)
    
    ################ Select the "sel_size" best individuals in the population (Truncation) ################
    
    def select(self,values):
        ordered = list(reversed([index for _,index in sorted(zip(values, [i for i in range(len(values))]))]))
        return ordered[:self.sel_size]

    ################ Create new population (Elitism) ################
    
    def new_population(self,values,individuals):
        #We remove the worst solution
        worst = values.index(min(values))
        return individuals[:worst]+individuals[worst+1:],values[:worst]+values[worst+1:]

    ################ Mutate the individuals in the population with a certain mutation rate ################
    
    def mutate(self,individuals,rate):
        #Each node in an individual has a certain probability of being modified
        for individual in individuals:
            for node in range(len(individual)):
                #If the node is selected, the value of its gene is randomly replaced with the label of one of its neighbors
                if rm.random() < rate:
                    individual[node] = np.random.choice(self.neighbors[node])
        #We compute the new mutation rate
        new_rate = rate-self.mutation_decrease
        #If new mutation rate above/equal minimum, we accept the new rate
        if new_rate >= self.min_mutation_rate:
            rate = new_rate
        #Else, we set the new rate to the minimum allowed rate
        else:
            rate = self.min_mutation_rate 
        return individuals, rate

    ################ Estimate the univariate marginal probability distribution of each gene ################
    
    def distribution_estimation(self,population,selected):
        #We only consider the label of the corresponding node and the label of its neighbors
        distribution = [[0 for i in range(len(self.neighbors[j])+1)] for j in range(len(self.G))]
        #First, we count the number of appearances of each allowed value in each gene
        for node in range(len(self.G)):
            indexes = {neighbor: index for (index, neighbor) in enumerate([node]+self.neighbors[node])}
            for sel in selected:
                distribution[node][indexes[population[sel][node]]] += 1
        #Then, we compute the frequency of each value
        for node in range(len(self.G)):
            total = sum(distribution[node])
            distribution[node] = [j/total for j in distribution[node]]
        return distribution

    ################ Estimation of distribution algorithm ################
    
    def EDA(self):
        start = tm.time()
        dict_degrees = dict(zip(self.G.nodes,self.node_degrees))
        #Initialize first population.
        population = self.initialize()
        #Translate and evaluate initial population.
        values,evals = self.evaluate(self.translate(population),dict_degrees,0)
        best_act = max(values)
        best_sol = population[values.index(best_act)]
        #Initialize mutation rate.
        mutation_rate = self.max_mutation_rate
        #Until there is no improvement after 5 attempts
        while evals < self.max_evals:
            #Elitism selection.
            selected = self.select(values)
            #Estimate distribution.
            distribution = self.distribution_estimation(population,selected)
            #Create new individuals.
            new_individuals = self.create_individuals(distribution,self.max_evals-evals)
            #Mutate new individuals.
            new_individuals,mutation_rate = self.mutate(new_individuals,mutation_rate)
            #Translate and evaluate new individuals.
            new_individuals_values,evals = self.evaluate(self.translate(new_individuals),dict_degrees,evals)
            #Create new population (Elitism). Consider new individuals + best solution.
            population,values = self.new_population(new_individuals_values+[best_act], new_individuals+[best_sol])
            #Update best solution if needed.
            act = max(values)
            if act > best_act:
                best_sol = population[values.index(act)]
                best_act = act
        #Translate the solution to the usual representation before returning
        end = tm.time()
        return self.translate([best_sol])[0], best_act, evals, end-start
      
    ################ Initialize the global atributtes ################
    
    def __init__(self,communities,max_evals,pop_size,sel_size,max_mutation_rate,min_mutation_rate,mutation_decrease):
        #Graph
        self.G = sortu_grafoa()
        #Names of the nodes
        self.names = list(self.G.nodes)
        #Total weight of the edges in the graph
        self.weight = self.G.size(weight="weight")
        #Node degrees
        self.node_degrees = [self.G.degree(node,weight="weight") for node in self.G]
        #Neighbors of each node
        self.neighbors = [[k for k in range(len(self.G)) if self.G.get_edge_data(self.names[j],self.names[k],default={"weight":0})["weight"]>0] for j in range(len(self.G))] 
        #Maximum number of communities. Precondition: Communities > 0
        self.communities = communities
        #Allowed number of evaluations. Precondition: Max_evals > 0
        self.max_evals = max_evals
        #Population size. Precondition: Pop_size > 0
        self.pop_size = pop_size
        #Size of the selected individuals in each iteration. Precondition: 0 < Sel_size <= Pop_size
        self.sel_size = sel_size
        #Maximum mutation rate. Precondition: Maximum_mutation_rate > 0
        self.max_mutation_rate = max_mutation_rate
        #Minimum mutation rate. Precondition: 0 < Minimum_mutation_rate <= Maximum_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        #Decrease value of the mutation rate
        self.mutation_decrease = mutation_decrease