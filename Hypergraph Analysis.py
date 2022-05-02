#!/usr/bin/env python
# coding: utf-8

# ## Import data from IMDB

# In[1]:


import pandas as pd

pricipals_df = pd.read_csv("title.pricipals.tsv", sep="\t")
pricipals_df


# In[2]:


basics_df = pd.read_csv("title.basics.tsv", sep="\t", low_memory=False)
basics_df


# In[3]:


name_df = pd.read_csv("name.basics.tsv", sep="\t")
name_df


# ## Data cleaning

# In[4]:


# Only keep category column is "actor"
pricipals_df = pricipals_df[pricipals_df['category'] == 'actor']
# drop ordering, category, job and characters columns
pricipals_df = pricipals_df.drop(['ordering', 'category', 'job', 'characters'], axis=1) 
pricipals_df


# In[5]:


# drop titleType, originalTitle, isAdult, endYear, runtimeMinutes and genres columns
basics_df = basics_df.drop(['titleType', 'originalTitle', 'isAdult', 'endYear', 'runtimeMinutes', 'genres'], axis=1) 
# Filter data by startYear is "1990"
basics_df = basics_df[basics_df['startYear'] == '1990']
basics_df


# In[6]:


# drop birthYear, deathYear, primaryProfession, and knownForTitles columns
name_df = name_df.drop(['birthYear', 'deathYear', 'primaryProfession', 'knownForTitles'], axis=1) 
name_df


# In[7]:


# Merge the three tables
df = pd.merge(left=basics_df,right=pricipals_df,on='tconst')
df = pd.merge(left=df,right=name_df,on='nconst')
# Export data to "hypergraph_data.csv"
df.to_csv('hypergraph_data.csv',index = False)


# In[1]:


import pandas as pd

df = pd.read_csv('hypergraph_data.csv')
df


# ## The COO representation

# In[2]:


import numpy as np
import pandas as pd
import nwhy as nwhy
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Select all values of the tconst column from the dataframe
tconst = copy.copy(df.iloc[:,0].values)
# Select all values of the primaryTitle column from the dataframe
primaryTitle = copy.copy(df.iloc[:,1].values)
# Select all values of the nconst column from the dataframe
nconst = copy.copy(df.iloc[:,3].values)
# Select all values of the primaryName column from the dataframe
primaryName = copy.copy(df.iloc[:,4].values)

hyperedge_to_title_dic = dict()
title_to_hyperedge_dic = dict()
title_to_tconst_dic = dict()
tconst_dic = dict()
i = 0
j = 0
for item in tconst:
    if(not tconst_dic.__contains__(item)):
        tconst_dic[item] = i
        hyperedge_to_title_dic[i] = primaryTitle[j]
        if(not title_to_hyperedge_dic.__contains__(primaryTitle[j])):
            title_to_hyperedge_dic[primaryTitle[j]] = []
            title_to_tconst_dic[primaryTitle[j]] = []
        title_to_hyperedge_dic[primaryTitle[j]].append(i)
        title_to_tconst_dic[primaryTitle[j]].append(tconst[j])
        i += 1
    tconst[j] = tconst_dic[item]
    j += 1

vertex_to_name_dic = dict()
name_to_vertex_dic = dict()
name_to_nconst_dic = dict()
nconst_dic = dict()
i = 0
j = 0
for item in nconst:
    if(not nconst_dic.__contains__(item)):
        nconst_dic[item] = i
        vertex_to_name_dic[i] = primaryName[j]
        if(not name_to_vertex_dic.__contains__(primaryName[j])):
            name_to_vertex_dic[primaryName[j]] = []
            name_to_nconst_dic[primaryName[j]] = []
        name_to_vertex_dic[primaryName[j]].append(i)
        name_to_nconst_dic[primaryName[j]].append(nconst[j])
        i += 1
    nconst[j] = nconst_dic[item]
    j += 1

weight = [1] * tconst.size

# Row of sparse matrix of the hypergraph (hyperedges)
row = tconst
# Columns of sparse matrix of the hypergraph (vertices)
col = nconst
# Weights of sparse matrix of the hypergraph
data = np.array(weight)


# ## Explore data relationships by manipulating the data table

# ### Query 1: How many TV shows/movies in IMDB whose startYear is 1990?

# In[3]:


def query_1_by_table():
    num_tv_movie_1990 = len(tconst_dic.keys())
    print('The number of TV shows/movies whose startYear is 1990: ', num_tv_movie_1990)
query_1_by_table()


# ### Query 2: How many actors who have acted in TV shows/movies with startYear of 1990 in IMDB?

# In[4]:


def query_2_by_table():
    num_actor_1990 = len(nconst_dic.keys())
    print('The number of actors who have acted in TV shows/movies with startYear of 1990: ', num_actor_1990)
query_2_by_table()


# ### Query 3: Enter a TV show/movie with startYear of 1990 in IMDB, and query the number of actors in that TV show/movie.

# In[5]:


title_tv_movie = input("Please enter a TV show/movie title: ")
def query_3_by_table(title_tv_movie):
    if(title_to_tconst_dic.__contains__(title_tv_movie)):
        # More than one line of output indicates that multiple movies have the same name.
        for tconst in title_to_tconst_dic[title_tv_movie]:
            temp_df = df[df['tconst'] == tconst]
            print('The number of actors in "', title_tv_movie, '": ', len(temp_df))
    else:
        print('Sorry, the TV show/movie "', title_tv_movie, '" was not found!')
query_3_by_table(title_tv_movie)


# ### Query 4: Enter a TV show/movie with startYear of 1990 in IMDB, and query the actors in that TV show/movie.

# In[6]:


title_tv_movie = input("Please enter a TV show/movie title: ")
def query_4_by_table(title_tv_movie):
    if(title_to_tconst_dic.__contains__(title_tv_movie)):
        # More than one block of output indicates that multiple movies have the same name.
        for tconst in title_to_tconst_dic[title_tv_movie]:
            temp_df = df[df['tconst'] == tconst]
            name_col = copy.copy(temp_df.iloc[:,4].values).tolist()
            for name in name_col:
                print('Actor', name_col.index(name) + 1, 'in "', title_tv_movie, '":', name)
    else:
        print('Sorry, the TV show/movie "', title_tv_movie, '" was not found!')
query_4_by_table(title_tv_movie)


# ### Query 5: Enter an actor, and query the number of TV shows/movies with starYear of 1990 in which the actor is in.

# In[7]:


name_actor = input("Please enter an actor: ")
def query_5_by_table(name_actor):
    if(name_to_nconst_dic.__contains__(name_actor)):
        # More than one line of output indicates that multiple actors have the same name.
        for nconst in name_to_nconst_dic[name_actor]:
            temp_df = df[df['nconst'] == nconst]
            print('The number of TV shows/movies with startYear of 1990 "', name_actor, '" is in: ', len(temp_df))
    else:
        print('Sorry, the actor "', name_actor, '" was not found!')
query_5_by_table(name_actor)


# ### Query 6: Enter an actor, and query TV shows/movies with starYear of 1990 in which the actor is in.

# In[8]:


name_actor = input("Please enter an actor: ")
def query_6_by_table(name_actor):
    if(name_to_nconst_dic.__contains__(name_actor)):
        # More than one block of output indicates that multiple actors have the same name.
        for nconst in name_to_nconst_dic[name_actor]:
            temp_df = df[df['nconst'] == nconst]
            title_col = copy.copy(temp_df.iloc[:,1].values).tolist()
            for title in title_col:
                print('TV show/movie', title_col.index(title) + 1, '"', name_actor, '" is in: ', title)
    else:
        print('Sorry, the actor "', name_actor, '" was not found!')
query_6_by_table(name_actor)


# ## Create Hypergraph

# In[9]:


# Create the hypergraph 
h = nwhy.NWHypergraph(row, col, data)
print('Hypergraph created successfully!', h)


# ## Visualize a part of the hypergraph through Hypernetx-Widget

# In[10]:


import imp
import hypernetx as hnx

N_hyperedges = int(input("Please enter the number of hyperedges that you want to form a part of the hypergraph: "))
hyperedges = np.arange(N_hyperedges)
scenes = dict()
for hyperedge in hyperedges:
    title = hyperedge_to_title_dic[hyperedge]
    scenes[title] = []
    vertices = h.edge_incidence(hyperedge)
    for vertex in vertices:
        name = vertex_to_name_dic[vertex]
        scenes[title].append(name)
    scenes[title] = tuple(scenes[title])
    
H = hnx.Hypergraph(scenes)
hnx.draw(H)


# ## NWHypergraph class methods:

# In[12]:


# NWHypergraph class methods:

# print('-- collapsing edges without returning equal class')
# equal_class = h.collapse_edges()
# print(equal_class)

# print('-- collapsing nodes without returning equal class')
# equal_class = h.collapse_nodes()
# print(equal_class)

# print('-- collapsing nodes and edges without returning equal class')
# equal_class = h.collapse_nodes_and_edges()
# print(equal_class)

# print('-- collapsing edges with returning equal class')
# equal_class = h.collapse_edges(return_equivalence_class=True)
# print(equal_class)

# print('-- collapsing nodes with returning equal class')
# equal_class = h.collapse_nodes(return_equivalence_class=True)
# print(equal_class)

# print('-- collapsing nodes and edges with returning equal class')
# equal_class = h.collapse_nodes_and_edges(return_equivalence_class=True)
# print(equal_class)

# print('-- edge_size_dist()')
# equal_class = h.edge_size_dist()
# print(equal_class)

# print('-- node_size_dist()')
# equal_class = h.node_size_dist()
# print(equal_class)

# print('-- edge_incidence(edge)')
# equal_class = h.edge_incidence(666)
# print(equal_class)

# print('-- node_incidence(node)')
# equal_class = h.node_incidence(666)
# print(equal_class)

# print('-- degree(node, min_size=1, max_size=None)')
# equal_class = h.degree(666, min_size=1, max_size=None)
# print(equal_class)

# print('-- size(edge, min_degree=1, max_degree=None)')
# equal_class = h.size(666, min_degree=1, max_degree=None)
# print(equal_class)

# print('-- dim(edge)')
# equal_class = h.dim(666)
# print(equal_class)

# print('-- number_of_nodes()')
# equal_class = h.number_of_nodes()
# print(equal_class)

# print('-- number_of_edges()')
# equal_class = h.number_of_edges()
# print(equal_class)

# print('-- singletons()')
# equal_class = h.singletons()
# print(equal_class)

# print('-- toplexes()')
# equal_class = h.toplexes()
# print(equal_class)

# print('-- s_linegraph(s=1, edges=True)')
# equal_class = h.s_linegraph(s=1, edges=True)
# print(equal_class)

# print('-- s_linegraphs(l, edges=True)')
# equal_class = h.s_linegraphs([1,2,3,4,5,6], edges=True)
# print(equal_class)


# ## Hypergraph application(analysis)

# ### Query 1: How many TV shows/movies in IMDB whose startYear is 1990?

# In[11]:


def query_1_by_hypergraph():
    num_tv_movie_1990 = h.number_of_edges()
    print('The number of TV shows/movies whose startYear is 1990: ', num_tv_movie_1990)
query_1_by_hypergraph()


# ### Query 2: How many actors who have acted in TV shows/movies with startYear of 1990 in IMDB? 

# In[12]:


def query_2_by_hypergraph():
    num_actor_1990 = h.number_of_nodes()
    print('The number of actors who have acted in TV shows/movies with startYear of 1990: ', num_actor_1990)
query_2_by_hypergraph()


# ### Query 3: Enter a TV show/movie with startYear of 1990 in IMDB, and query the number of actors in that TV show/movie.

# In[13]:


title_tv_movie = input("Please enter a TV show/movie title: ")
def query_3_by_hypergraph(title_tv_movie):
    if(title_to_hyperedge_dic.__contains__(title_tv_movie)):
        # More than one line of output indicates that multiple movies have the same name.
        for hyperedge in title_to_hyperedge_dic[title_tv_movie]:
            num_actor_tv_movie = h.size(hyperedge, min_degree=1, max_degree=None)
            print('The number of actors in "', title_tv_movie, '": ', num_actor_tv_movie)
    else:
        print('Sorry, the TV show/movie "', title_tv_movie, '" was not found!')
query_3_by_hypergraph(title_tv_movie)


# ### Query 4: Enter a TV show/movie with startYear of 1990 in IMDB, and query the actors in that TV show/movie.

# In[14]:


title_tv_movie = input("Please enter a TV show/movie title: ")
def query_4_by_hypergraph(title_tv_movie):
    if(title_to_hyperedge_dic.__contains__(title_tv_movie)):
        # More than one block of output indicates that multiple movies have the same name.
        for hyperedge in title_to_hyperedge_dic[title_tv_movie]:
            vertices = h.edge_incidence(hyperedge)
            for vertex in vertices:
                name = vertex_to_name_dic[vertex]
                print('Actor', vertices.index(vertex) + 1, 'in "', title_tv_movie, '":', name)
    else:
        print('Sorry, the TV show/movie "', title_tv_movie, '" was not found!')
query_4_by_hypergraph(title_tv_movie)


# ### Query 5: Enter an actor, and query the number of TV shows/movies with starYear of 1990 in which the actor is in.

# In[15]:


name_actor = input("Please enter an actor: ")
def query_5_by_hypergraph(name_actor):
    if(name_to_vertex_dic.__contains__(name_actor)):
        # More than one line of output indicates that multiple actors have the same name.
        for vertex in name_to_vertex_dic[name_actor]:
            num_tv_movie_actor = h.degree(vertex, min_size=1, max_size=None)
            print('The number of TV shows/movies with startYear of 1990 "', name_actor, '" is in: ', num_tv_movie_actor)
    else:
        print('Sorry, the actor "', name_actor, '" was not found!')
query_5_by_hypergraph(name_actor)


# ### Query 6: Enter an actor, and query TV shows/movies with starYear of 1990 in which the actor is in.

# In[16]:


name_actor = input("Please enter an actor: ")
def query_6_by_hypergraph(name_actor):
    if(name_to_vertex_dic.__contains__(name_actor)):
        # More than one block of output indicates that multiple actors have the same name.
        for vertex in name_to_vertex_dic[name_actor]:
            hyperedges = h.node_incidence(vertex)
            for hyperedge in hyperedges:
                title = hyperedge_to_title_dic[hyperedge]
                print('TV show/movie', hyperedges.index(hyperedge) + 1, '"', name_actor, '" is in: ', title)
    else:
        print('Sorry, the actor "', name_actor, '" was not found!')
query_6_by_hypergraph(name_actor)


# ### Query 7: Find the top N TV shows/movies with the most actors

# In[17]:


N = int(input("Please enter the value of N: "))
num_vertices_arr = h.edge_size_dist()
desc_num_vertices_arr = sorted(num_vertices_arr, reverse = True)
N_desc_num_vertices_arr = desc_num_vertices_arr[:N]
print('Top #\tTV show/movie title\tthe number of actors\tactors\n')
i = 1
title_arr = []
num_actor_arr = []
for item in N_desc_num_vertices_arr:
    idx = num_vertices_arr.index(item)
    num_vertices_arr[idx] = -1
    title = hyperedge_to_title_dic[idx]
    title_arr.append(title)
    num_actor_arr.append(item)
    print('Top', i, '\t', title, '\t\t', item, end='\t')
    vertices = h.edge_incidence(idx)
    for vertex in vertices:
        name = vertex_to_name_dic[vertex]
        print(name,end = ', ')
    print('\n')
    i += 1


# In[18]:


import matplotlib.pyplot as plt
x = title_arr
y = num_actor_arr
plt.figure(figsize=(16, 6))
plt.bar(x, y, width=0.5)
plt.xticks(fontsize=10, rotation=20)
plt.yticks(fontsize=18)
plt.ylabel(u'The number of actors', fontsize=20)
diagram_title = str(N).join(['Diagram of Top ',' TV shows/movies with the most actors'])
plt.title(diagram_title, fontsize=20)
plt.show()


# ### Query 8: Find the top N actors with the most appearance times

# In[19]:


N = int(input("Please enter the value of N: "))
num_hyperedges_arr = h.node_size_dist()
desc_num_hyperedges_arr = sorted(num_hyperedges_arr, reverse = True)
N_desc_num_hyperedges_arr = desc_num_hyperedges_arr[:N]
print('Top #\tname\tthe number of TV shows/movies\tTV shows/movies\n')
i = 1
name_arr = []
num_tv_movie_arr = []
for item in N_desc_num_hyperedges_arr:
    idx = num_hyperedges_arr.index(item)
    num_hyperedges_arr[idx] = -1
    name = vertex_to_name_dic[idx]
    name_arr.append(name)
    num_tv_movie_arr.append(item)
    print('Top', i, '\t', name, '\t\t', item, end='\t')
    hyperedges = h.node_incidence(idx)
    for hyperedge in hyperedges:
        if(hyperedges.index(hyperedge) == 10):
            print('......')
            break
        title = hyperedge_to_title_dic[hyperedge]
        print(title,end = ', ')
    print('\n')
    i += 1


# In[20]:


x = name_arr
y = num_tv_movie_arr
plt.figure(figsize=(16, 6))
plt.bar(x, y, width=0.5)
plt.xticks(fontsize=10, rotation=20)
plt.yticks(fontsize=18)
plt.ylabel(u'The apperance times', fontsize=20)
diagram_title = str(N).join(['Diagram of Top ',' actors with the most appearance times'])
plt.title(diagram_title, fontsize=20)
plt.show()


# ### Query 9: Find movies with only one actor in it, and that actor only acted in that movie.

# In[21]:


singleton_hyperedges = h.singletons()
if(len(singleton_hyperedges) > 0):
    print('title\t\t\t\t\t\t\tactor\n')
else:
    print('There is no such movie.')
for hyperedge in singleton_hyperedges:
    if(singleton_hyperedges.index(hyperedge) == 30):
        print('......')
        break
    title = hyperedge_to_title_dic[hyperedge]
    print(title, end = '\t\t\t\t\t\t\t')
    vertices = h.edge_incidence(hyperedge)
    name = vertex_to_name_dic[vertices[0]]
    print(name)


# ## Comparisons of results and runtime between NWhy lib and the approach to manipulate the data table  

# ### Comparison 1 (Query 1)

# In[22]:


from time import time

print('Query 1:')
print('\n--- Query 1 through manipulating the data table ---')
t_start=time()
query_1_by_table()
t_end=time()
t_cost=t_end-t_start
print('Manipulating the data table takes %0.8f seconds'%t_cost)

print('\n--- Query 1 through NWhy lib ---')
t_start=time()
query_1_by_hypergraph()
t_end=time()
t_cost=t_end-t_start
print('NWhy lib takes %0.8f seconds'%t_cost)

del t_end,t_start,t_cost


# ### Comparison 2 (Query 2)

# In[23]:


print('Query 2:')
print('\n--- Query 2 through manipulating the data table ---')
t_start=time()
query_2_by_table()
t_end=time()
t_cost=t_end-t_start
print('Manipulating the data table takes %0.8f seconds'%t_cost)

print('\n--- Query 2 through NWhy lib ---')
t_start=time()
query_2_by_hypergraph()
t_end=time()
t_cost=t_end-t_start
print('NWhy lib takes %0.8f seconds'%t_cost)

del t_end,t_start,t_cost


# ### Comparison 3 (Query 3)

# In[24]:


print('Query 3:\n')
title_tv_movie = input("Please enter a TV show/movie title: ")
print('\n--- Query 3 through manipulating the data table ---')
t_start=time()
query_3_by_table(title_tv_movie)
t_end=time()
t_cost=t_end-t_start
print('Manipulating the data table takes %0.8f seconds'%t_cost)

print('\n--- Query 3 through NWhy lib ---')
t_start=time()
query_3_by_hypergraph(title_tv_movie)
t_end=time()
t_cost=t_end-t_start
print('NWhy lib takes %0.8f seconds'%t_cost)

del t_end,t_start,t_cost


# ### Comparison 4 (Query 4)

# In[25]:


print('Query 4:\n')
title_tv_movie = input("Please enter a TV show/movie title: ")
print('\n--- Query 4 through manipulating the data table ---')
t_start=time()
query_4_by_table(title_tv_movie)
t_end=time()
t_cost=t_end-t_start
print('Manipulating the data table takes %0.8f seconds'%t_cost)

print('\n--- Query 4 through NWhy lib ---')
t_start=time()
query_4_by_hypergraph(title_tv_movie)
t_end=time()
t_cost=t_end-t_start
print('NWhy lib takes %0.8f seconds'%t_cost)

del t_end,t_start,t_cost


# ### Comparison 5 (Query 5)

# In[26]:


print('Query 5:\n')
name_actor = input("Please enter an actor: ")
print('\n--- Query 5 through manipulating the data table ---')
t_start=time()
query_5_by_table(name_actor)
t_end=time()
t_cost=t_end-t_start
print('Manipulating the data table takes %0.8f seconds'%t_cost)

print('\n--- Query 5 through NWhy lib ---')
t_start=time()
query_5_by_hypergraph(name_actor)
t_end=time()
t_cost=t_end-t_start
print('NWhy lib takes %0.8f seconds'%t_cost)

del t_end,t_start,t_cost


# ### Comparison 6 (Query 6)

# In[27]:


print('Query 6:\n')
name_actor = input("Please enter an actor: ")
print('\n--- Query 6 through manipulating the data table ---')
t_start=time()
query_6_by_table(name_actor)
t_end=time()
t_cost=t_end-t_start
print('Manipulating the data table takes %0.8f seconds'%t_cost)

print('\n--- Query 6 through NWhy lib ---')
t_start=time()
query_6_by_hypergraph(name_actor)
t_end=time()
t_cost=t_end-t_start
print('NWhy lib takes %0.8f seconds'%t_cost)

del t_end,t_start,t_cost


# ## Create Slinegraph

# In[28]:


# Create the slinegraph
s = h.s_linegraph(s=1, edges=True)
print('Slinegraph created successfully! (s=1)', s)


# ## Slinegraph class methods:

# In[25]:


# Slinegraph class methods:

# print('-- get_singletons()')
# equal_class = s.get_singletons()
# print(equal_class)

# print('-- s_connected_components()')
# equal_class = s.s_connected_components()
# print(equal_class)

# print('-- is_s_connected()')
# equal_class = s.is_s_connected()
# print(equal_class)

# print('-- s_distance(src, dest)')
# equal_class = s.s_distance(src, dest)
# print(equal_class)

# print('-- s_diameter(src, dest)')
# equal_class = s.s_diameter(src, dest)
# print(equal_class)

# print('-- s_path(src, dest)')
# equal_class = s.s_path(src, dest)
# print(equal_class)

# print('-- s_betweenness_centrality(normalized=True)')
# equal_class = s.s_betweenness_centrality(normalized=True)
# print(equal_class)

# print('-- s_closeness_centrality(v=None)')
# equal_class = s.s_closeness_centrality(v=None)
# print(equal_class)

# print('-- s_harmonic_closeness_centrality(v=None)')
# equal_class = s.s_harmonic_closeness_centrality(v=None)
# print(equal_class)

# print('-- s_eccentricity(v=None)')
# equal_class = s.s_eccentricity(v=None)
# print(equal_class)

# print('-- s_neighbors(v)')
# equal_class = s.s_neighbors(v)
# print(equal_class)

# print('-- s_degree(v)')
# equal_class = s.s_degree(v)
# print(equal_class)


# ## S-line graphs construction

# In[29]:


s_line_graph_arr = h.s_linegraphs([1,2,3,4,5,6,7,8,9,10], edges=True)
print(s_line_graph_arr)


# ## Diagram of the relationship between S and S-linegraph size

# In[30]:


x_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x = np.arange(len(x_ticks))
y = [891400, 539207, 119446, 34237, 7370, 1334, 334, 115, 42, 0]

plt.figure(figsize=(13, 6))
plt.plot(x, y, color='#FF0000', label='S-linegraph', linewidth=3.0)
for a, b in zip(x, y):
    plt.text(a, b, '%d'%b, ha='center', va= 'bottom', fontsize=18)
    
plt.xticks([r for r in x], x_ticks, fontsize=18, rotation=20)
plt.yticks(fontsize=18)
plt.xlabel(u'S', fontsize=18)
plt.ylabel(u'S-linegraph size(# of edges)', fontsize=18)
plt.title(u'Diagram of the relationship between S and S-line graph size', fontsize=18)
plt.legend(fontsize=18)

plt.show()


# #### S-line graph size decreases sharply with increasing S.

# ## S-line graph application(analysis)

# In[31]:


s1 = s_line_graph_arr[0] # S(=1)-line graph
s3 = s_line_graph_arr[2] # S(=3)-line graph
s5 = s_line_graph_arr[4] # S(=5)-line graph
s7 = s_line_graph_arr[6] # S(=7)-line graph

def find_top_10_betweenness_centrality_score(s):
    betweenness_centrality_score_arr = s.s_betweenness_centrality(normalized=True)
    print('Top #\t\ttitle\t\tbetweenness centrality score')
    title_arr = []
    score_arr = []
    for i in range(10):
        score = max(betweenness_centrality_score_arr)
        hyperedge = np.argmax(betweenness_centrality_score_arr)
        title = hyperedge_to_title_dic[hyperedge]
        score_arr.append(score)
        title_arr.append(title)
        betweenness_centrality_score_arr[hyperedge] = 0
        print('Top ',i + 1, '\t\t', title, '\t\t', score)
    return [title_arr, score_arr]
    
def find_top_10_closeness_centrality_score(s):
    closeness_centrality_score_arr = s.s_closeness_centrality(v=None)
    print('Top #\t\ttitle\t\tcloseness centrality score')
    title_arr = []
    score_arr = []
    for i in range(10):
        score = max(closeness_centrality_score_arr)
        hyperedge = np.argmax(closeness_centrality_score_arr)
        title = hyperedge_to_title_dic[hyperedge]
        score_arr.append(score)
        title_arr.append(title)
        closeness_centrality_score_arr[hyperedge] = 0
        print('Top ',i + 1, '\t\t', title, '\t\t', score)
    return [title_arr, score_arr]
    
def find_top_10_harmonic_closeness_centrality_score(s):
    harmonic_closeness_centrality_score_arr = s.s_harmonic_closeness_centrality(v=None)
    print('Top #\t\ttitle\t\tharmonic closeness centrality score')
    title_arr = []
    score_arr = []
    for i in range(10):
        score = max(harmonic_closeness_centrality_score_arr)
        hyperedge = np.argmax(harmonic_closeness_centrality_score_arr)
        title = hyperedge_to_title_dic[hyperedge]
        score_arr.append(score)
        title_arr.append(title)
        harmonic_closeness_centrality_score_arr[hyperedge] = 0
        print('Top ',i + 1, '\t\t', title, '\t\t', score)
    return [title_arr, score_arr]

print('-- Top 10 movies with the highest betweenness centrality score:')
print('\nFor S(=1)-line graph:')
s1_betweenness_result = find_top_10_betweenness_centrality_score(s1)
print('\nFor S(=3)-line graph:')
s3_betweenness_result = find_top_10_betweenness_centrality_score(s3)
print('\nFor S(=5)-line graph:')
s5_betweenness_result = find_top_10_betweenness_centrality_score(s5)
print('\nFor S(=7)-line graph:')
s7_betweenness_result = find_top_10_betweenness_centrality_score(s7)

print('\n\n-- Top 10 movies with the highest closeness centrality score:')
print('\nFor S(=1)-line graph:')
s1_closeness_result = find_top_10_closeness_centrality_score(s1)
print('\nFor S(=3)-line graph:')
s3_closeness_result = find_top_10_closeness_centrality_score(s3)
print('\nFor S(=5)-line graph:')
s5_closeness_result = find_top_10_closeness_centrality_score(s5)
print('\nFor S(=7)-line graph:')
s7_closeness_result = find_top_10_closeness_centrality_score(s7)

print('\n\n-- Top 10 movies with the highest harmonic closeness centrality score:')
print('\nFor S(=1)-line graph:')
s1_harmonic_closeness_result = find_top_10_harmonic_closeness_centrality_score(s1)
print('\nFor S(=3)-line graph:')
s3_harmonic_closeness_result = find_top_10_harmonic_closeness_centrality_score(s3)
print('\nFor S(=5)-line graph:')
s5_harmonic_closeness_result = find_top_10_harmonic_closeness_centrality_score(s5)
print('\nFor S(=7)-line graph:')
s7_harmonic_closeness_result = find_top_10_harmonic_closeness_centrality_score(s7)


# ## Diagrams of Top 10 movies based on different scores(S=1/3/5/7)

# In[32]:


def plot_diagram_of_top_10_score(result, output):
    x = result[0]
    y = result[1]
    plt.figure(figsize=(16, 6))
    plt.bar(x, y, width=0.5)
    plt.xticks(fontsize=10, rotation=20)
    plt.yticks(fontsize=18)
    plt.ylabel(output, fontsize=20)
    diagram_title = 'Diagram of Top 10 movies with the highest ' + output
    plt.title(diagram_title, fontsize=20)
    plt.show()


# ### Diagram of Top 10 movies with the highest betweenness centrality score(S=1)

# In[33]:


plot_diagram_of_top_10_score(s1_betweenness_result, 'betweenness centrality score(S=1)')


# ### Diagram of Top 10 movies with the highest betweenness centrality score(S=3)

# In[34]:


plot_diagram_of_top_10_score(s3_betweenness_result, 'betweenness centrality score(S=3)')


# ### Diagram of Top 10 movies with the highest betweenness centrality score(S=5)

# In[35]:


plot_diagram_of_top_10_score(s5_betweenness_result, 'betweenness centrality score(S=5)')


# ### Diagram of Top 10 movies with the highest betweenness centrality score(S=7)

# In[36]:


plot_diagram_of_top_10_score(s7_betweenness_result, 'betweenness centrality score(S=7)')


# ### Diagram of Top 10 movies with the highest closeness centrality score(S=1)

# In[37]:


plot_diagram_of_top_10_score(s1_closeness_result, 'closeness centrality score(S=1)')


# ### Diagram of Top 10 movies with the highest closeness centrality score(S=3)

# In[38]:


plot_diagram_of_top_10_score(s3_closeness_result, 'closeness centrality score(S=3)')


# ### Diagram of Top 10 movies with the highest closeness centrality score(S=5)

# In[39]:


plot_diagram_of_top_10_score(s5_closeness_result, 'closeness centrality score(S=5)')


# ### Diagram of Top 10 movies with the highest closeness centrality score(S=7)

# In[40]:


plot_diagram_of_top_10_score(s7_closeness_result, 'closeness centrality score(S=7)')


# ### Diagram of Top 10 movies with the highest harmonic closeness centrality score(S=1)

# In[41]:


plot_diagram_of_top_10_score(s1_harmonic_closeness_result, 'harmonic closeness centrality score(S=1)')


# ### Diagram of Top 10 movies with the highest harmonic closeness centrality score(S=3)

# In[42]:


plot_diagram_of_top_10_score(s3_harmonic_closeness_result, 'harmonic closeness centrality score(S=3)')


# ### Diagram of Top 10 movies with the highest harmonic closeness centrality score(S=5)

# In[43]:


plot_diagram_of_top_10_score(s5_harmonic_closeness_result, 'harmonic closeness centrality score(S=5)')


# ### Diagram of Top 10 movies with the highest harmonic closeness centrality score(S=7)

# In[44]:


plot_diagram_of_top_10_score(s7_harmonic_closeness_result, 'harmonic closeness centrality score(S=7)')

