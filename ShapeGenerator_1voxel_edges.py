#!/usr/bin/env python
# coding: utf-8

# In[1]:


#INPUT: Resolution of images
import time
import gc
ts = time.time()
res=50


# In[2]:


#Generate tensor that represents the wedges 
##1) Generate the wedges as tensors 
import numpy as np

wedge1s=np.ones((res, res,res), dtype = int)
wedge1s=np.tril(wedge1s)
wedge2s=np.flip(wedge1s, axis=1)
wedge3s=np.flip(wedge1s, axis=(1,2))
wedge4s=np.flip(wedge1s, axis=(2))
wedge5s=np.rot90(wedge1s, k=1, axes=(0, 1))
wedge6s=np.flip(wedge5s, axis=(2))
wedge7s=np.flip(wedge5s, axis=(2,0))
wedge8s=np.flip(wedge5s, axis=(1,0))
wedge9s=np.rot90(wedge1s, k=1, axes=(0, 2))
wedge10s=np.flip(wedge9s, axis=(1))
wedge11s=np.flip(wedge9s, axis=(1,0))
wedge12s=np.flip(wedge9s, axis=(2,0,))

import ipyvolume as ipv
#ipv.quickvolshow(wedge1s,level=[1, 1], opacity=1, data_min=0, data_max=1)


# In[3]:


#This is the vector of length 12 that control which wedges to have.

#Now we need to merge multiple wedhges and output a new set of 6 images
def merge_wedges_single_voxel(action):

  # We muptipli the action vector for each of the wedges and them add them all together
  wedge_1t=np.multiply(wedge1s,action[0])
  wedge_2t=np.multiply(wedge2s,action[1])
  wedge_3t=np.multiply(wedge3s,action[2])
  wedge_4t=np.multiply(wedge4s,action[3])
  wedge_5t=np.multiply(wedge5s,action[4])
  wedge_6t=np.multiply(wedge6s,action[5])
  wedge_7t=np.multiply(wedge7s,action[6])
  wedge_8t=np.multiply(wedge8s,action[7])
  wedge_9t=np.multiply(wedge9s,action[8])
  wedge_10t=np.multiply(wedge10s,action[9])
  wedge_11t=np.multiply(wedge11s,action[10])
  wedge_12t=np.multiply(wedge12s,action[11])

  #add all wedges
  voxel=wedge_1t++wedge_2t++wedge_3t++wedge_4t++wedge_5t++wedge_6t++wedge_7t++wedge_8t++wedge_9t++wedge_10t++wedge_11t++wedge_12t

  voxel = np.where(voxel > 1, 1, voxel)
  return voxel


# In[4]:


#This functions merges all the final voxels into one final shape
def merge_voxels(action_v, over_p):

    final_shape=np.zeros(((res*2), (res*2),(res*2)), dtype = int)

    #Just if we need  to overlap to plot
    over=over_p
    final_shape[0:res,0:res,0:res]=final_shape[0:res,0:res,0:res]++merge_wedges_single_voxel(action_v[0])
    final_shape[(res-over):((res*2)-over),0:res,0:res]=final_shape[(res-over):((res*2)-over),0:res,0:res]++merge_wedges_single_voxel(action_v[1])
    final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),0:res]=final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),0:res]++merge_wedges_single_voxel(action_v[2])
    final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),(res-over):((res*2)-over)]=final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[3])

    final_shape[0:res,(res-over):((res*2)-over),0:res]=final_shape[0:res,(res-over):((res*2)-over),0:res]++merge_wedges_single_voxel(action_v[4])
    final_shape[0:res,(res-over):((res*2)-over),(res-over):((res*2)-over)]=final_shape[0:res,(res-over):((res*2)-over),(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[5])

    final_shape[(res-over):((res*2)-over),0:res,(res-over):((res*2)-over)]=final_shape[(res-over):((res*2)-over),0:res,(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[6])
    final_shape[0:res,0:res,(res-over):((res*2)-over)]=final_shape[0:res,0:res,(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[7])

    return final_shape


# In[5]:


action_v=[[0,0,1,0,1,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0]]

final_shape=merge_voxels(action_v,0)
#ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1) 


# 
# 

# # Edges using lines and Network of nodes
# 
# https://stackoverflow.com/questions/42301481/adding-specific-lines-to-a-plotly-scatter3d-plot

# ## Create list of nodes for 1 voxels

# In[6]:


#NODE LIST  1 voxel
import numpy as np
import pandas as pd
res_s=res
nodes=np.array([
    [res_s-1,-1,res_s],
[res_s,-1,-1],
[res_s,res_s,-1],
[res_s,res_s,res_s],
[res_s, res_s/2, res_s/2],
[res_s/2,-1,res_s/2],
[res_s/2,res_s/2,0],
[res_s/2,res_s,res_s/2],
[res_s/2, res_s/2, res_s],
[-1,-1,res_s],
[-1,-1,-1],
[-1,res_s,-1],
[-1,res_s,res_s],
[-1,res_s/2,res_s/2],
[res_s/2,res_s/2,res_s/2],
])

nodes=pd.DataFrame(nodes, columns = ['x','y','z'], index=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"])
nodes


# Ploting the nodes as per the "reference" voxel, this to help identify where the nodes are

# In[7]:


import plotly.graph_objs as go
import plotly
plotly.offline.init_notebook_mode()


trace1 = go.Scatter3d(
    x=nodes["x"],
    y=nodes["y"],
    z=nodes["z"],
    mode='markers',
    text=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"],
    marker=dict(
        size=12,
        color=["green", "green","green","green","green",
              "blue","blue","blue","blue",
              "red", "red", "red", "red", "red",
              "blue"]
    )
)


fig = go.Figure(data=[trace1])
plotly.offline.iplot(fig, filename='simple-3d-scatter')


# ## Creating the different wedges|

# In[8]:


#FIXED MATCH ACTION_V

wedge1= np.array([
        [3,12],
        [12,13],
        [13,4],
        [4,3],
        [2,3],
        [2,11],
        [11,12],
        [3,12],
        [2,5],
        [4,5],
        [11,14],
        [13,14],
]) 

wedge2= np.array([
    [1,2],
    [2,3],
    [3,5],
    [1,5],
    [2,11],
    [11,12],
    [1,10],
    [10,11],
    [10,14],
    [12,14],
    [3,12]
]) 



wedge3= np.array([
        
    [1,10],
    [10,11],
    [1,2],
    [2,11],
    [10,13],
    [4,13],
    [1,4],
    [1,10],
    [11,14],
    [13,14],
    [2,5],
    [4,5]

])  


wedge4= np.array([
    [1,4],
    [4,13],
    [10,13],
    [1,10],
    [3,4],
    [3,12],
    [12,13],
    [4,13],
    [1,5],
    [10,14],
    [12,14],
    [3,5],

])

wedge5= np.array([
    [2,3],
    [3,12],
    [11,12],
    [2,11],
    [3,8],
    [8,13],
    [2,6],
    [6,10],
    [10,13],
    [12,13],
    [3,8],
    [8,13],
    [10,11],
])



wedge6= np.array([
    [1,4],
    [1,10],
    [4,13],
    [10,13],
    [12,13],
    [4,8],
    [8,12],
    [1,6],
    [6,11],
    [10,11],
    [11,12],
]) 

wedge7= np.array([ 
    
    [3,4],
    [1,4],
    [1,2],
    [2,3],
    [4,13],
    [1,10],
    [1,4],
    [10,13],
    [3,8],
    [8,13],
    [2,6],
    [6,10]
    
    
])

wedge8= np.array([
    [2,3],
    [3,12],
    [11,12],
    [2,11],
    [3,4],
    [1,4],
    [1,2],
    [2,3],
    [4,8],
    [8,12],
    [1,6],
    [6,11],
    
])


wedge9= np.array([
        [1,2],
        [2,3],
        [3,4],
        [1,4],
        [3,12],
        [12,13],
        [4,13],
        [3,4],
        [1,9],
        [9,13],
        [2,7],
        [7,12],
]) 


wedge10= np.array([
        [1,2],
        [2,3],
        [3,4],
        [1,4],
        [1,10],
        [1,2],
        [2,11],
        [4,9],
        [9,10],
        [3,7],
        [7,11],
        [10,11],

]) 


wedge11 = np.array([
        [10,13],
        [12,13],
        [11,12],
        [1,9],
        [9,13],
        [2,7],
        [7,12],
        [1,10],
        [1,2],
        [2,11],
        [10,11],
])

wedge12= np.array([
        [10,13],
        [11,12],
        [4,9],
        [9,10],
        [3,7],
        [7,11],
        [10,11],
        [4,13],
        [3,4],
        [3,12],
        [12,13],
])


# ## Generate  wedge node and edge data

# In[9]:


def plot_wedge(wedge,new_shape_pd):
    
    
    import plotly.graph_objs as go
    import plotly
    plotly.offline.init_notebook_mode()
    
    
    Colors_list=["white"]*new_shape_pd.shape[0]



    trace1 = go.Scatter3d(
        x=new_shape_pd["x"],
        y=new_shape_pd["y"],
        z=new_shape_pd["z"],
        mode='markers',

        marker=dict(
            size=2,
            color=Colors_list
        )
    )
   



    x=nodes["x"]
    y=nodes["y"]
    z=nodes["z"]

    x_lines = list()
    y_lines = list()
    z_lines = list()


    for p in wedge:                             ###################CHANGE WEDGE NAME HERE
        for i in range(2):
            x_lines.append(x.loc[str(p[i])])
            y_lines.append(y.loc[str(p[i])])
            z_lines.append(z.loc[str(p[i])])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    trace2 = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(
        color='black',
        width=10)
    )

    scene2 = dict(
            xaxis = dict(
                 backgroundcolor="rgba(0, 0, 0,0)",
                 gridcolor="white",
                 showbackground=True,
                 zerolinecolor="white",),
            yaxis = dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"),
            zaxis = dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="rgba(0, 0, 0,0)",
                showbackground=True,
                zerolinecolor="rgba(0, 0, 0,0)",),)


    fig = go.Figure(data=[trace2,trace1,trace2])
    #fig.update_layout(scene=scene2)         #REMOVE" backgroun
    #fig.update_layout(font_color="white")    #"REMOVE" text


    plotly.offline.iplot(fig, filename='simple-3d-scatter')


# In[10]:



def plot_blue_wedges_confirn(input_w):
    #Empty action_v
    action_v=[[0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0]]

    #update action_v based on the wedges to merge
    for w in range(len(input_w)):
        action_v[0][input_w[w]-1]=1


    #use th action_v to generate the 3D matrix of "pixels" (i.e., 0s and 1s)
    final_shape=merge_voxels(action_v,0) 

    final_shape_index_w1=np.where(final_shape == 1)
    new_shape_matrix=np.stack((final_shape_index_w1[0],final_shape_index_w1[1],final_shape_index_w1[2])) 
    new_shape_pd=pd.DataFrame(new_shape_matrix,index=["x","y","z"]).T

    
    
    
    List_of_weges=list([wedge1,wedge2,wedge3, wedge4, wedge5, wedge6, wedge7, wedge8, wedge9, wedge10, wedge11, wedge12]) #list of wedges to easy filter

    
    x=nodes["x"]
    y=nodes["y"]
    z=nodes["z"]

    x_lines = list()
    y_lines = list()
    z_lines = list()

    
    
    import plotly.graph_objs as go
    import plotly
    plotly.offline.init_notebook_mode()

    Colors_list=["white"]*new_shape_pd.shape[0]

    #Colors_list[reference_points_index[0]]="red"

    trace1 = go.Scatter3d(
        x=new_shape_pd["x"],
        y=new_shape_pd["y"],
        z=new_shape_pd["z"],
        mode='markers',

        marker=dict(
            size=2,
            color=Colors_list
        )
    )
   


    for p in List_of_weges[input_w[0]-1]:                            
        for i in range(2):
            x_lines.append(x.loc[str(p[i])])
            y_lines.append(y.loc[str(p[i])])
            z_lines.append(z.loc[str(p[i])])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    trace2 = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(
        color='black',
        width=15)
    )

    scene2 = dict(
            xaxis = dict(
                 backgroundcolor="rgba(0, 0, 0,0)",
                 gridcolor="white",
                 showbackground=True,
                 zerolinecolor="white",),
            yaxis = dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"),
            zaxis = dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="rgba(0, 0, 0,0)",
                showbackground=True,
                zerolinecolor="rgba(0, 0, 0,0)",),)


    fig = go.Figure(data=[trace2,trace1,trace2])
    #fig.update_layout(scene=scene2)         #REMOVE" backgroun
    #fig.update_layout(font_color="white")    #"REMOVE" text


    plotly.offline.iplot(fig, filename='simple-3d-scatter')


# In[11]:


#plot_blue_wedges_confirn([1])


# ## Adding wedges

# In[12]:


#dictionary of reference points
key = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o', 'p','q']
value = [[1,10,25,0,res-1], [10,11,0,0,25], [1,2,res-1,0,25],[11,2,25,0,0],[10,13,0,25,res-1],
         [13,4,25,res-1,res-1],[1,4,res-1,25,res-1],[11,13,0,25,25],[2,4,res-1,25,25],[2,10,25,25,25],
         [2,3,res-1,25,0],[3,4,res-1,res-1,25],[3,13,25,res-1,25],[11,12,0,25,0],
         [3,12,25,res-1,0],[10,12,0,25,25],
         [1,3,res-1,25,25],[1,11,25,0,25],[4,12,25,res-1,25],
        [12,13,0,res-1,25],[4,10,25,25,res-1],[3,11,25,25,0],[1,13,25,25,res-1],
         [2,12,25,25,0],[2,11,25,0,0],
         [4,13,25,res-1,res-1],
         [13,14,0,37,37],[11,14,0,12,12],[11,12,0,25,0],[1,9,37,13,res-1],[9,13,13,37,res-1],
         [2,7,37,13,0],[7,12,13,37,0],
        [4,9,37,37,res-1],[9,10,13,13,res-1],[3,7,37,37,0],[2,5,res-1,13,13],[4,5,res-1,37,37],
         [4,3,res-1,res-1,25],
         [3,5,res-1,37,13],
         [12,14,0,37,13],[10,14,0,13,37],[1,5,res-1,13,37],[8,13,13,res-1,37],
         [3,8,37,res-1,13],[7,11,13,13,0], [2,6,37,0,13], [6,10,13,0,37], [1,6,37,0,37],[4,8,37,res-1,37], [6,11,13,0,13],
        [8,12,13,res-1,13]]

b_dictionary={}

for i in range(len(key)):
    b_dictionary[key[i]]=value[i]

print(b_dictionary)


# In[13]:


df = pd.DataFrame(data=value)
df.columns =['Edge1', 'Edge2', 'x', 'y','z']
df.filter(regex='Edge')
df.head()


# In[14]:


data_v1=[[3,8,12,8,8,15]
,[4,8,8,13,8,15]
,[8,13,8,12,8,15]
,[4,8,3,8,8,15]
,[6,10,1,6,6,15]
,[6,11,2,6,6,15]
,[6,10,6,11,6,15]
,[1,6,2,6,6,15]
,[1,5,4,5,5,15]
,[2,5,3,5,5,15]
,[3,5,4,5,5,15]
,[2,5,1,5,5,15]
,[13,14,10,14,14,15]
,[12,14,11,14,14,15]
,[13,14,12,14,14,15]
,[10,14,11,14,14,15]
,[9,13,9,10,9,15]
,[4,9,1,9,9,15]
,[9,10,1,9,9,15]
,[9,13,4,9,9,15]
,[7,11,7,12,7,15]
,[2,7,3,7,7,15]
,[3,7,7,12,7,15]
,[7,11,2,7,7,15]]


np_v1 =np.array(data_v1)
np_v1


# ## Algoriths to Merge and  remove unwanted  edges

# In[15]:


# INPUT:
input_w=[7,3] #Wedges to merge



#Empty action_v
action_v=[[0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0]]

#update action_v based on the wedges to merge
for w in range(len(input_w)):
    action_v[0][input_w[w]-1]=1
    

#use th action_v to generate the 3D matrix of "pixels" (i.e., 0s and 1s)
final_shape=merge_voxels(action_v,0)
final_shape_index_w1=np.where(final_shape == 1)
new_shape_matrix=np.stack((final_shape_index_w1[0],final_shape_index_w1[1],final_shape_index_w1[2])) 

#Pandas dataframe with the x,y,z cordinates of all the "pixels" equal to 1
new_shape_pd=pd.DataFrame(new_shape_matrix,index=["x","y","z"]).T


#Merge wedges
List_of_weges=list([wedge1,wedge2,wedge3, wedge4, wedge5, wedge6, wedge7, wedge8, wedge9, wedge10, wedge11, wedge12]) #list of wedges to easy filter

Wedges_to_merge=list(List_of_weges[input_w[0]-1])  #need at least one input
for w in range(1,len(input_w)):
    Wedges_to_merge=(Wedges_to_merge,List_of_weges[input_w[w]-1])
Wedges_to_merge       #list of wedges to merge
k=np.concatenate(Wedges_to_merge,axis=0) 
data = np.unique(k, axis=0)


# In[16]:


new_shape_pd


# In[17]:


## For each "edge" the list of merge_unique edges
edges_to_remove=list()
for i in range(data.shape[0]):
    #print(data[i])
    #i=2  # for testing without loop
    #Find the reference point in the dataframe df
    reference_point_coordinates=df[  (df["Edge1"]==data[i][0]) & (df["Edge2"]==data[i][1])]  #Filter rows
    reference_point_coordinates=reference_point_coordinates[["x","y","z"]]             #Select columns
    reference_point_coordinates=reference_point_coordinates.to_numpy()
    ## For the x, y and z value of the reference points do:
    total_sum=0
    sum_of_plus_minus_one_x=0
    sum_of_plus_minus_one_y=0
    sum_of_plus_minus_one_z=0
    
           
            #Check if +1 & -1 on the cordinate has both "pixel"equal to 1

    #For coordinate X DO
    if (reference_point_coordinates[0,0]!=0) & (reference_point_coordinates[0,0]!=res-1):
        sum_of_plus_minus_one_x=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0,0]+3) &
             (new_shape_pd["y"]==reference_point_coordinates[0,1]) &
             (new_shape_pd["z"]==reference_point_coordinates[0,2])]).shape[0]) +

             ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0,0]-3) &
             (new_shape_pd["y"]==reference_point_coordinates[0,1]) &
             (new_shape_pd["z"]==reference_point_coordinates[0,2])]).shape[0]) )

    #if the coordinate we are looking is y DO
    if (reference_point_coordinates[0,1]!=0) & (reference_point_coordinates[0,1]!=res-1):
        sum_of_plus_minus_one_y=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0,0]) &
             (new_shape_pd["y"]==reference_point_coordinates[0,1]+3) &
             (new_shape_pd["z"]==reference_point_coordinates[0,2])]).shape[0]) +

             ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0,0]) &
             (new_shape_pd["y"]==reference_point_coordinates[0,1]-3) &
             (new_shape_pd["z"]==reference_point_coordinates[0,2])]).shape[0]) )

    #if the coordinate we are looking is z DO
    if (reference_point_coordinates[0,2]!=0) & (reference_point_coordinates[0,2]!=res-1):
        sum_of_plus_minus_one_z=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0,0]) &
             (new_shape_pd["y"]==reference_point_coordinates[0,1]) &
             (new_shape_pd["z"]==reference_point_coordinates[0,2]+3)]).shape[0]) +

             ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0,0]) &
             (new_shape_pd["y"]==reference_point_coordinates[0,1]) &
             (new_shape_pd["z"]==reference_point_coordinates[0,2]-3)]).shape[0]) )


    total_sum=(sum_of_plus_minus_one_x+sum_of_plus_minus_one_y+sum_of_plus_minus_one_z)
    
    #If total sum is >4  (i.e., there are pixels +1 and -1 in at least 2 axis): REMOVE
    if total_sum>=4:
        edges_to_remove.append(i)


# In[18]:


# Update the edges list by removing unwanted edges
for e in range(len(edges_to_remove)):
#edges_to_remove
    data=np.delete(data, edges_to_remove[e]-e, 0)


# In[19]:


##Add Edge from Big V
for e in range(np_v1.shape[0]):
    found_edge=0
    node1_v1=np_v1[e,0]
    node2_v1=np_v1[e,1]
    node1_v2=np_v1[e,2]
    node2_v2=np_v1[e,3]
    add_edge=np.reshape(np.array([np_v1[e,4],np_v1[e,5]]), (1,2))

    for n in range(data.shape[0]):
        if(data[n,0]== node1_v1 and data[n,1]==node2_v1):
            found_edge=found_edge+1
    for n in range(data.shape[0]):
        if(data[n,0]== node1_v2 and data[n,1]==node2_v2):
            found_edge=found_edge+1
    if(found_edge==2):
        #print(add_edge)
        data=np.concatenate((data, add_edge))

        
##Remove middle lines from fullShape to visualize inner lines
new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["y"]==25) & (new_shape_pd["z"]==25)].index)
new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["x"]==25) & (new_shape_pd["z"]==25)].index)
new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["y"]==25) & (new_shape_pd["x"]==25)].index)

new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["y"]==24) & (new_shape_pd["z"]==24)].index)
new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["x"]==24) & (new_shape_pd["z"]==24)].index)
new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["y"]==24) & (new_shape_pd["x"]==24)].index)

new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["y"]==26) & (new_shape_pd["z"]==26)].index)
new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["x"]==26) & (new_shape_pd["z"]==26)].index)
new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["y"]==26) & (new_shape_pd["x"]==26)].index)

#new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["y"]==24) & (new_shape_pd["z"]==25)].index)
#new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["x"]==24) & (new_shape_pd["z"]==25)].index)
#new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["y"]==24) & (new_shape_pd["x"]==25)].index)


#new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["y"]==25) & (new_shape_pd["z"]==24)].index)
#new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["x"]==25) & (new_shape_pd["z"]==24)].index)
#new_shape_pd=new_shape_pd.drop(new_shape_pd[ (new_shape_pd["y"]==25) & (new_shape_pd["x"]==24)].index)


# In[20]:


#plot_wedge(data,new_shape_pd)


# ### Wedge from point cloud

# In[21]:



from tqdm import tqdm
import plotly.io as pio

def draw_shape_from_cloud(input_w):
    #Empty action_v
    action_v=[[0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0]]

    #update action_v based on the wedges to merge
    for w in range(len(input_w)):
        action_v[0][input_w[w]-1]=1


    #use th action_v to generate the 3D matrix of "pixels" (i.e., 0s and 1s)
    final_shape=merge_voxels(action_v,0)
    final_shape_index_w1=np.where(final_shape == 1)
    new_shape_matrix=np.stack((final_shape_index_w1[0],final_shape_index_w1[1],final_shape_index_w1[2])) 

    #Pandas dataframe with the x,y,z cordinates of all the "pixels" equal to 1
    new_shape_pd=pd.DataFrame(new_shape_matrix,index=["x","y","z"]).T
    #new_shape_pd

    # Go over each point in the cloud and check for +-1 in the 3 axis

    #add the new colums to keep track of this
    new_shape_pd["sum"]=0

    for i in tqdm(range(new_shape_pd.shape[0])):

        reference_point_coordinates=new_shape_pd.iloc[i,:].to_numpy()
        ## For the x, y and z value do:
        total_sum=0
        sum_of_plus_minus_one_x=0
        sum_of_plus_minus_one_y=0
        sum_of_plus_minus_one_z=0

     #Check if +1 & -1 on the cordinate has both "pixel"equal to 1
        #For coordinate X DO
        if (reference_point_coordinates[0]!=0) & (reference_point_coordinates[0]!=res-1):
            sum_of_plus_minus_one_x=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]+1) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) +

                 ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]-1) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) )

        #if the coordinate we are looking is y DO
        if (reference_point_coordinates[1]!=0) & (reference_point_coordinates[1]!=res-1):
            sum_of_plus_minus_one_y=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]+1) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) +

                 ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]-1) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) )

        #if the coordinate we are looking is z DO
        if (reference_point_coordinates[2]!=0) & (reference_point_coordinates[2]!=res-1):
            sum_of_plus_minus_one_z=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2]+1)]).shape[0]) +

                 ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2]-1)]).shape[0]) )


        total_sum=(sum_of_plus_minus_one_x+sum_of_plus_minus_one_y+sum_of_plus_minus_one_z)
        new_shape_pd.iloc[i,3]=total_sum

    #select size and color of pixels based on edge vs not
    new_shape_pd["color"]="rgba(255, 255, 255,1)"
    new_shape_pd.loc[ new_shape_pd["sum"] <=2 , "color"] = "rgba(0, 0,0,1)"
    new_shape_pd.loc[ new_shape_pd["sum"] ==5 , "color"] = "rgba(0, 0,0,1)"

    new_shape_pd["size"]=5
    new_shape_pd.loc[ new_shape_pd["color"] =="rgba(0, 0,0,1)" , "size"] = 10
    new_shape_pd.loc[ new_shape_pd["sum"] ==5 , "size"] = 15


    x=nodes["x"]
    y=nodes["y"]
    z=nodes["z"]

    x_lines = list()
    y_lines = list()
    z_lines = list()



    import plotly.graph_objs as go
    import plotly
    plotly.offline.init_notebook_mode()

    Colors_list=new_shape_pd["color"]
    size_list=new_shape_pd["size"]
    
    trace1 = go.Scatter3d(
        x=new_shape_pd["x"],
        y=new_shape_pd["y"],
        z=new_shape_pd["z"],
        mode='markers',


        marker=dict(
            size=size_list,
            symbol="circle",
            color=Colors_list,
            opacity= 1,
            line=dict(width=0,
                     color=Colors_list)
        )
    )

    fig = go.Figure(data=[trace1])

  #REMOVE" background
    scene2 = dict(
            xaxis = dict(
                 backgroundcolor="rgba(255, 255, 255,1)",
                 gridcolor="white",
                 showbackground=False,
                 zerolinecolor="white",),
            yaxis = dict(
                backgroundcolor="rgba(255, 255, 255,1)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white"),
            zaxis = dict(
                backgroundcolor="rgba(255, 255, 255,1)",
                gridcolor="rgba(255, 255, 255,1)",
                showbackground=False,
                zerolinecolor="rgba(255, 255, 255,1)",),)
    fig.update_layout(scene=scene2)   
    
    #"REMOVE" text
    fig.update_layout(font_color="rgba(255, 255, 255,0.2)")    

    # Change Camera setting
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2.25, y=2.25, z=2.25)
    )

    fig.update_layout(scene_camera=camera)

    #save image
    image_name=str(res)+"_"+str(input_w[0])+"_"+str(input_w[1])+".png"
    pio.write_image(fig, image_name,scale=10, width=1080, height=1080)
    
    #show images
    plotly.offline.iplot(fig, filename='simple-3d-scatter')


# In[ ]:


input_w=[11,2]
draw_shape_from_cloud(input_w)
 
#FOR TESTING
#import random
#for i in range(500):  
 #   input_w=[random.randint(0,11),random.randint(0,11)]  
 #   print(input_w)
 #   draw_shape_from_cloud(input_w)


# ##### NEED TO WORK ON ROTATING THE CAMERA ANGLE USING CODE (i.e., rotating the image automatically). 
# 
# https://community.plotly.com/t/rotating-3d-plots-with-plotly/34776

# ### Wedge with surfaces
# 
# https://stackoverflow.com/questions/62403763/how-to-add-planes-in-a-3d-scatter-plot
# 

# In[ ]:


#Generate Surfaces based on 4 points corrdinates

surface1_nodes=[2,11,13,4]   #####ORDER IS VERY IMPORTANT!!!!!!


# In[ ]:


import gdspy as gy
import numpy as np

# INPUT:
input_w=[1]   #Wedges to merge

#Empty action_v
action_v=[[0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0]]

#update action_v based on the wedges to merge
for w in range(len(input_w)):
    action_v[0][input_w[w]-1]=1


#use th action_v to generate the 3D matrix of "pixels" (i.e., 0s and 1s)
final_shape=merge_voxels(action_v,0) 

final_shape_index_w1=np.where(final_shape == 1)
new_shape_matrix=np.stack((final_shape_index_w1[0],final_shape_index_w1[1],final_shape_index_w1[2])) 
new_shape_pd=pd.DataFrame(new_shape_matrix,index=["x","y","z"]).T

    
    

List_of_weges=list([wedge1,wedge2,wedge3, wedge4, wedge5, wedge6, wedge7, wedge8, wedge9, wedge10, wedge11, wedge12]) #list of wedges to easy filter


x=nodes["x"]
y=nodes["y"]
z=nodes["z"]

x_lines = list()
y_lines = list()
z_lines = list()




for p in List_of_weges[input_w[0]-1]:                            
    for i in range(2):
        x_lines.append(x.loc[str(p[i])])
        y_lines.append(y.loc[str(p[i])])
        z_lines.append(z.loc[str(p[i])])
    x_lines.append(None)
    y_lines.append(None)
    z_lines.append(None)

trace2 = go.Scatter3d(
    x=x_lines,
    y=y_lines,
    z=z_lines,
    mode='lines',
    line=dict(
    color='black',
    width=8)
)







#########################################################################################
x=list(nodes[nodes.index.isin(     map(str, surface1_nodes)    )]["x"])
y=list(nodes[nodes.index.isin(     map(str, surface1_nodes)    )]["y"])
z=list(nodes[nodes.index.isin(     map(str, surface1_nodes)    )]["z"])

lightdict=dict(ambient=1,diffuse=1,fresnel=1,specular=1,roughness=1)

mesh = go.Mesh3d(x=x, y=y, z=z, color='white', showscale=False, lighting=lightdict)
mesh2 = go.Mesh3d(x=x, y=y, z=z, color='white', showscale=False, lighting=lightdict)


fig = go.Figure(data=[trace2,mesh])
fig.add_trace(go.Surface(x=x, y=y, z=z_plane_pos, colorscale=light_yellow,  showscale=False))
fig


# In[ ]:


surface1_nodes=[1,2,3,4]   #####ORDER IS VERY IMPORTANT!!!!!!


# In[ ]:



x=list(nodes[nodes.index.isin(     map(str, surface1_nodes)    )]["x"])
y=list(nodes[nodes.index.isin(     map(str, surface1_nodes)    )]["y"])
z=list(nodes[nodes.index.isin(     map(str, surface1_nodes)    )]["z"])

lightdict=dict(ambient=1,diffuse=1,fresnel=1,specular=1,roughness=1)

mesh = go.Mesh3d(x=x, y=y, z=z, color='white', showscale=False, lighting=lightdict)


fig = go.Figure(data=[trace2,mesh])
fig


# In[ ]:


import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/tiago-peres/immersion/master/Platforms_dataset.csv')

fig = px.scatter_3d(df, x='Functionality ', y='Accessibility', z='Immersion', color='Platforms')

bright_blue = [[0, '#7DF9FF'], [1, '#7DF9FF']]
bright_pink = [[0, '#FF007F'], [1, '#FF007F']]
light_yellow = [[0, '#FFDB58'], [1, '#FFDB58']]

# need to add starting point of 0 to each dimension so the plane extends all the way out
zero_pt = pd.Series([0])
z = zero_pt.append(df['Immersion'], ignore_index = True).reset_index(drop = True)
y = zero_pt.append(df['Accessibility'], ignore_index = True).reset_index(drop = True)
x = zero_pt.append(df['Functionality '], ignore_index = True).reset_index(drop = True)

length_data = len(z)
z_plane_pos = 40*np.ones((length_data,length_data))

fig.add_trace(go.Surface(x=x, y=y, z=z_plane_pos, colorscale=light_yellow,  showscale=False))
fig.add_trace(go.Surface(x=x.apply(lambda x: 10), y=y, z = np.array([z]*length_data), colorscale= bright_blue, showscale=False))
fig.add_trace(go.Surface(x=x, y= y.apply(lambda x: 1), z =  np.array([z]*length_data).transpose(), colorscale=bright_pink, showscale=False))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




