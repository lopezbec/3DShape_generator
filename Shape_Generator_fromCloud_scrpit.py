#!/usr/bin/env python
# coding: utf-8

# In[1]:


#INPUT: Resolution of images
import time
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.io as pio
from datetime import datetime
import sys


# In[2]:


def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

#WIP
def rotate_x(x, y, z, theta):
    w = y+1j*z
    return x, y, np.imag(np.exp(1j*theta)*w)


def rotate_y(x, y, z, theta):
    w = z+1j*x
    return np.real(np.exp(1j*theta)*w), y, z


# In[3]:


def draw_shape_from_cloud(name, action_v,t,axis,res):
    
    #Gnerate tensor that represents the wedges 
##1) Generate the wedges as tensors 


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

    #This functions merges all the final voxels into one final shape
    def merge_voxels(action_v, over_pm, res):

        final_shape=np.zeros(((res*2), (res*2),(res*2)), dtype = int)

        #Just if we need  to overlap to plot
        over=0
        final_shape[0:res,0:res,0:res]=final_shape[0:res,0:res,0:res]++merge_wedges_single_voxel(action_v[0])
        final_shape[(res-over):((res*2)-over),0:res,0:res]=final_shape[(res-over):((res*2)-over),0:res,0:res]++merge_wedges_single_voxel(action_v[1])
        final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),0:res]=final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),0:res]++merge_wedges_single_voxel(action_v[2])
        final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),(res-over):((res*2)-over)]=final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[3])

        final_shape[0:res,(res-over):((res*2)-over),0:res]=final_shape[0:res,(res-over):((res*2)-over),0:res]++merge_wedges_single_voxel(action_v[4])
        final_shape[0:res,(res-over):((res*2)-over),(res-over):((res*2)-over)]=final_shape[0:res,(res-over):((res*2)-over),(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[5])

        final_shape[(res-over):((res*2)-over),0:res,(res-over):((res*2)-over)]=final_shape[(res-over):((res*2)-over),0:res,(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[6])
        final_shape[0:res,0:res,(res-over):((res*2)-over)]=final_shape[0:res,0:res,(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[7])

        return final_shape

    
    #Empty action_v

    #use th action_v to generate the 3D matrix of "pixels" (i.e., 0s and 1s)
    final_shape=merge_voxels(action_v,0, res)
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
        
        #For diagonal
        sum_xy=0
        sum_xz=0
        sum_yz=0
        sum_xyz=0

     #Check if +1 & -1 on the cordinate has both "pixel"equal to 1
        #For coordinate X DO
        if (reference_point_coordinates[0]!=0) & (reference_point_coordinates[0]!=new_shape_pd.max().max()):
            sum_of_plus_minus_one_x=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]+1) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) +

                 ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]-1) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) )

        #if the coordinate we are looking is y DO
        if (reference_point_coordinates[1]!=0) & (reference_point_coordinates[1]!=new_shape_pd.max().max()):
            sum_of_plus_minus_one_y=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]+1) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) +

                 ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]-1) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) )

        #if the coordinate we are looking is z DO
        if (reference_point_coordinates[2]!=0) & (reference_point_coordinates[2]!=new_shape_pd.max().max()):
            sum_of_plus_minus_one_z=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2]+1)]).shape[0]) +

                 ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2]-1)]).shape[0]) )
            
        #Checing for diagonals
        
        for ax in (-1,+1):
            for ay in (-1,+1):
                  sum_xy= sum_xy+ (new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]+ax) &
                    (new_shape_pd["y"]==reference_point_coordinates[1]+ay) &
                    (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]
                       
        for ax in (-1,+1):
            for az in (-1,+1):
                  sum_xz= sum_xz+ (new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]+ax) &
                    (new_shape_pd["y"]==reference_point_coordinates[1]) &
                    (new_shape_pd["z"]==reference_point_coordinates[2]+az)]).shape[0]
                    
        for ay in (-1,+1):
            for az in (-1,+1):
                sum_yz= sum_yz+ (new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                (new_shape_pd["y"]==reference_point_coordinates[1]+ay) &
                (new_shape_pd["z"]==reference_point_coordinates[2]+az)]).shape[0]
                
        for ay in (-1,+1):
            for az in (-1,+1):
                for ax in (-1,+1):
                    sum_xyz= sum_xyz+ (new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]+ax) &
                    (new_shape_pd["y"]==reference_point_coordinates[1]+ay) &
                    (new_shape_pd["z"]==reference_point_coordinates[2]+az)]).shape[0]


        total_sum=(sum_of_plus_minus_one_x+sum_of_plus_minus_one_y+sum_of_plus_minus_one_z+sum_xy+sum_xz+sum_yz+sum_xyz)
        new_shape_pd.iloc[i,3]=total_sum
        
    #select size and color of pixels based on edge vs not
    new_shape_pd["color"]="rgba(0, 0,0,1)"
    
    sumf =[26,16,17,23]

    new_shape_pd.loc[ new_shape_pd["sum"].isin(sumf) , "color"] = "rgba(255, 255, 255,1)"     

    new_shape_pd["size"]=5
    new_shape_pd.loc[ new_shape_pd["color"] =="rgba(0, 0,0,1)" , "size"] = 10
    


   # x=nodes["x"]
   # y=nodes["y"]
   #z=nodes["z"]

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
                showticklabels=False,
                showgrid=False,
                 gridcolor="white",
                 showbackground=False,
                 zerolinecolor="white",),
            yaxis = dict(
                backgroundcolor="rgba(255, 255, 255,1)",
                gridcolor="white",
                showticklabels=False,
                showgrid=False,
                showbackground=False,
                zerolinecolor="white"),
            zaxis = dict(
                backgroundcolor="rgba(255, 255, 255,1)",
                gridcolor="rgba(255, 255, 255,1)",
                showticklabels=False,
                showgrid=False,
                showbackground=False,
                zerolinecolor="rgba(255, 255, 255,1)",),)
    fig.update_layout(scene=scene2)   

    #"REMOVE" text
    fig.update_layout(font_color="rgba(255, 255, 255,0.0)")
    
    #Rotate
    x_eye = -1.25
    y_eye = 2
    z_eye = 1.5
    if axis=="z":
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
    if axis=="x":
        xe, ye, ze = rotate_x(x_eye, y_eye, z_eye, -t)
    if axis=="y":
        xe, ye, ze = rotate_y(x_eye, y_eye, z_eye, -t)
        
    fig.update_layout(scene_camera_eye=dict(x=xe, y=ye, z=ze))
    
    #save image
    now = datetime.now()
    image_name=str(name)+str(axis)+str(res)+"_"+str(t)+"_t_"+str(now.strftime("%H_%M_%S"))+".png"
    pio.write_image(fig, image_name,scale=6, width=1080, height=1080)
    
    #show images
    plotly.offline.iplot(fig, filename='simple-3d-scatter')


# In[4]:


action_v=[[1,0,0,0,0,0,0,0,0,0,0,0],
          [1,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0]]
t=4
for t in np.arange(0,6.25,0.5):
    draw_shape_from_cloud(str(sys.argv[1]), action_v, t, str(sys.argv[2]),int(sys.argv[3]))
#draw_shape_from_cloud(action_v,t, "z",10)


# In[ ]:




