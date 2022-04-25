#!/usr/bin/env python
# coding: utf-8

# ## Import data from IMDB

# In[1]:


import pandas as pd


# In[2]:


pricipals_df = pd.read_csv("title.pricipals.tsv", sep="\t")
pricipals_df


# In[3]:


basics_df = pd.read_csv("title.basics.tsv", sep="\t", low_memory=False)
basics_df


# ## Data cleaning

# In[4]:


# Only keep category column is "actor"
pricipals_df = pricipals_df[pricipals_df['category'] == 'actor']
# drop category, job and characters columns
pricipals_df = pricipals_df.drop(['category', 'job', 'characters'], axis=1) 
pricipals_df


# In[5]:


# drop titleType, primaryTitle, originalTitle, isAdult, endYear, runtimeMinutes and genres columns
basics_df = basics_df.drop(['titleType', 'primaryTitle', 'originalTitle', 'isAdult', 'endYear', 'runtimeMinutes', 'genres'], axis=1) 
# Filter data by startYear is "1990"
basics_df = basics_df[basics_df['startYear'] == '1990']
basics_df


# In[6]:


# Merge two tables
df = pd.merge(left=pricipals_df,right=basics_df,on='tconst')
# Export data to "hypergraph_data.csv"
df.to_csv('hypergraph_data.csv',index = False)


# In[7]:


import pandas as pd
df = pd.read_csv('hypergraph_data.csv')
df


# ## The COO representation

# In[8]:


import numpy as np
import pandas as pd
import nwhy as nwhy
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Select all values of the tconst column from the dataframe
tconst = copy.copy(df.iloc[:,0].values)
# Select all values of the nconst column from the dataframe
nconst = copy.copy(df.iloc[:,2].values)

tconst_dic = dict()
i = 0
j = 0
for item in tconst:
    if(not tconst_dic.__contains__(item)):
        tconst_dic[item] = i
        i += 1
    tconst[j] = tconst_dic[item]
    j += 1

nconst_dic = dict()
i = 0
j = 0
for item in nconst:
    if(not nconst_dic.__contains__(item)):
        nconst_dic[item] = i
        i += 1
    nconst[j] = nconst_dic[item]
    j += 1

weight = [1] * tconst.size

# Row of sparse matrix of the hypergraph (hyperedges)
row = np.array(tconst)
# Columns of sparse matrix of the hypergraph (vertices)
col = np.array(nconst)
# Weights of sparse matrix of the hypergraph
data = np.array(weight)


# ## Create the hypergraph

# In[9]:


# Create the hypergraph 
h = nwhy.NWHypergraph(row, col, data)
print('Hypergraph created successfully!', h)


# ## NWHypergraph class methods:

# In[10]:


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

