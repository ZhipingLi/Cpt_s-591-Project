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


# In[8]:


import pandas as pd

df = pd.read_csv('hypergraph_data.csv')
df


# ## The COO representation

# In[9]:


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
tconst_dic = dict()
i = 0
j = 0
for item in tconst:
    if(not tconst_dic.__contains__(item)):
        tconst_dic[item] = i
        hyperedge_to_title_dic[i] = primaryTitle[j]
        if(not title_to_hyperedge_dic.__contains__(primaryTitle[j])):
            title_to_hyperedge_dic[primaryTitle[j]] = []
        title_to_hyperedge_dic[primaryTitle[j]].append(i)
        i += 1
    tconst[j] = tconst_dic[item]
    j += 1

vertex_to_name_dic = dict()
name_to_vertex_dic = dict()
nconst_dic = dict()
i = 0
j = 0
for item in nconst:
    if(not nconst_dic.__contains__(item)):
        nconst_dic[item] = i
        vertex_to_name_dic[i] = primaryName[j]
        if(not name_to_vertex_dic.__contains__(primaryName[j])):
            name_to_vertex_dic[primaryName[j]] = []
        name_to_vertex_dic[primaryName[j]].append(i)
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


# ## Create Hypergraph

# In[10]:


# Create the hypergraph 
h = nwhy.NWHypergraph(row, col, data)
print('Hypergraph created successfully!', h)


# ## Visualize a part of the hypergraph through Hypernetx-Widget

# In[11]:


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

# In[13]:


num_tv_movie_1990 = h.number_of_edges()
print('The number of TV shows/movies whose startYear is 1990: ', num_tv_movie_1990)


# ### Query 2: How many actors who have acted in TV shows/movies with startYear of 1990 in IMDB? 

# In[14]:


num_actor_1990 = h.number_of_nodes()
print('The number of actors who have acted in TV shows/movies with startYear of 1990: ', num_actor_1990)


# ### Query 3: Enter a TV show/movie with startYear of 1990 in IMDB, and query the number of actors in that TV show/movie.

# In[15]:


title_tv_movie = input("Please enter a TV show/movie title: ")
if(title_to_hyperedge_dic.__contains__(title_tv_movie)):
    # More than one line of output indicates that multiple movies have the same name.
    for hyperedge in title_to_hyperedge_dic[title_tv_movie]:
        num_actor_tv_movie = h.size(hyperedge, min_degree=1, max_degree=None)
        print('The number of actors in "', title_tv_movie, '": ', num_actor_tv_movie)
else:
    print('Sorry, the TV show/movie "', title_tv_movie, '" was not found!')


# ### Query 4: Enter a TV show/movie with startYear of 1990 in IMDB, and query the actors in that TV show/movie.

# In[16]:


title_tv_movie = input("Please enter a TV show/movie title: ")
if(title_to_hyperedge_dic.__contains__(title_tv_movie)):
    # More than one block of output indicates that multiple movies have the same name.
    for hyperedge in title_to_hyperedge_dic[title_tv_movie]:
        vertices = h.edge_incidence(hyperedge)
        for vertex in vertices:
            name = vertex_to_name_dic[vertex]
            print('Actor', vertices.index(vertex) + 1, 'in "', title_tv_movie, '":', name)
else:
    print('Sorry, the TV show/movie "', title_tv_movie, '" was not found!')


# ### Query 5: Enter an actor, and query the number of TV shows/movies with starYear of 1990 in which the actor is in.

# In[17]:


name_actor = input("Please enter an actor: ")
if(name_to_vertex_dic.__contains__(name_actor)):
    # More than one line of output indicates that multiple actors have the same name.
    for vertex in name_to_vertex_dic[name_actor]:
        num_tv_movie_actor = h.degree(vertex, min_size=1, max_size=None)
        print('The number of TV shows/movies with startYear of 1990 "', name_actor, '" is in: ', num_tv_movie_actor)
else:
    print('Sorry, the actor "', name_actor, '" was not found!')


# ### Query 6: Enter an actor, and query TV shows/movies with starYear of 1990 in which the actor is in.

# In[18]:


name_actor = input("Please enter an actor: ")
if(name_to_vertex_dic.__contains__(name_actor)):
    # More than one block of output indicates that multiple movies have the same name.
    for vertex in name_to_vertex_dic[name_actor]:
        hyperedges = h.node_incidence(vertex)
        for hyperedge in hyperedges:
            title = hyperedge_to_title_dic[hyperedge]
            print('TV show/movie', hyperedges.index(hyperedge) + 1, '"', name_actor, '" is in: ', title)
else:
    print('Sorry, the actor "', name_actor, '" was not found!')


# ### Query 7: Find the top N TV shows/movies with the most actors

# In[19]:


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


# In[20]:


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

# In[21]:


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


# In[22]:


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

# In[23]:


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


# ## Create Slinegraph

# In[24]:


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

# In[26]:


s_line_graph_arr = h.s_linegraphs([1,2,3,4,5,6,7,8,9,10], edges=True)
print(s_line_graph_arr)


# ## Diagram of the relationship between S and S-linegraph size

# In[27]:


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
plt.ylabel(u'S-linegraph size', fontsize=18)
plt.title(u'Diagram of the relationship between S and S-linegraph size', fontsize=18)
plt.legend(fontsize=18)

plt.show()


# #### S-linegraph size decreases sharply with increasing S
