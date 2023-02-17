#!/usr/bin/env python
# coding: utf-8

# In[19]:


#INPUT: Resolution of images
import math
import time
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.io as pio
from datetime import datetime
import plotly.graph_objs as go
import plotly


# In[20]:


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


# In[21]:


def draw_shape_from_cloud(action_v,t,axis,res):
    
    # Gnerate tensor that represents the wedges 
    # 1) Generate the wedges as tensors 

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

    # Now we need to merge multiple wedhges and output a new set of 6 images
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

    # This functions merges all the final voxels into one final shape
    def merge_voxels(action_v, over_pm, res):

        final_shape=np.zeros(((res*2), (res*2),(res*2)), dtype = int)

        # Just if we need  to overlap to plot
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

    
    # Empty action_v

    # use th action_v to generate the 3D matrix of "pixels" (i.e., 0s and 1s)
    final_shape=merge_voxels(action_v,0, res)
    final_shape_index_w1=np.where(final_shape == 1)
    new_shape_matrix=np.stack((final_shape_index_w1[0],final_shape_index_w1[1],final_shape_index_w1[2])) 

    # Pandas dataframe with the x,y,z cordinates of all the "pixels" equal to 1
    new_shape_pd=pd.DataFrame(new_shape_matrix,index=["x","y","z"]).T
    


    # Go over each point in the cloud and check for +-1 in the 3 axis

    # add the new colums to keep track of this
    new_shape_pd["sum"]=0

    for i in tqdm(range(new_shape_pd.shape[0])):

        reference_point_coordinates=new_shape_pd.iloc[i,:].to_numpy()
        # For the x, y and z value do:
        total_sum=0
        sum_of_plus_minus_one_x=0
        sum_of_plus_minus_one_y=0
        sum_of_plus_minus_one_z=0
        
        #For diagonal
        sum_xy=0
        sum_xz=0
        sum_yz=0
        sum_xyz=0

        # Check if +1 & -1 on the cordinate has both "pixel"equal to 1
        # For coordinate X DO
        if (reference_point_coordinates[0]!=0) & (reference_point_coordinates[0]!=new_shape_pd.max().max()):
            sum_of_plus_minus_one_x=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]+1) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) +

                 ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]-1) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) )

        # if the coordinate we are looking is y DO
        if (reference_point_coordinates[1]!=0) & (reference_point_coordinates[1]!=new_shape_pd.max().max()):
            sum_of_plus_minus_one_y=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]+1) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) +

                 ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]-1) &
                 (new_shape_pd["z"]==reference_point_coordinates[2])]).shape[0]) )

        # if the coordinate we are looking is z DO
        if (reference_point_coordinates[2]!=0) & (reference_point_coordinates[2]!=new_shape_pd.max().max()):
            sum_of_plus_minus_one_z=(((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2]+1)]).shape[0]) +

                 ((new_shape_pd[ (new_shape_pd["x"]==reference_point_coordinates[0]) &
                 (new_shape_pd["y"]==reference_point_coordinates[1]) &
                 (new_shape_pd["z"]==reference_point_coordinates[2]-1)]).shape[0]) )
            
        # Checing for diagonals
        
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
        
    # remove innner part of shape
    new_shape_pd=new_shape_pd.drop(new_shape_pd[new_shape_pd['sum']== 26].index)
    
  
    # select size and color of pixels based on edge vs not
    new_shape_pd["color"]="rgba(0, 0,0,1)"
    
    sumf =[26,16,17,23]

    new_shape_pd.loc[ new_shape_pd["sum"].isin(sumf) , "color"] = "rgba(255, 255, 255,1)"     

    # EDGES
#     edge_sum =[edge_n]

    # new_shape_pd.loc[ new_shape_pd["sum"].isin(edge_sum) , "color"] = "rgba(255, 0, 0,1)"
    new_shape_pd["size"]=5
    new_shape_pd.loc[ new_shape_pd["color"] =="rgba(0, 0,0,1)" , "size"] = 10
    # new_shape_pd.loc[ new_shape_pd["color"] =="rgba(255, 0, 0,1)" , "size"] = 10
    
   # x=nodes["x"]
   # y=nodes["y"]
   # z=nodes["z"]

    x_lines = list()
    y_lines = list()
    z_lines = list()

        
    if axis=="x":
        new_shape_pd.rename(columns = {'x':'y', 'y':'z','z':'x'}, inplace = True)
    if axis=="y":
        new_shape_pd.rename(columns = {'x':'z', 'y':'x','z':'y'}, inplace = True)
    # new_shape_pd


    
    #plotly.offline.init_notebook_mode()

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

#     "REMOVE" background
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
#     fig.update_layout(scene=scene2)   

    # "REMOVE" text
    fig.update_layout(font_color="rgba(255, 255, 255,0.0)")
    
    # Rotate
    x_eye = -1.25
    y_eye = 2
    z_eye = 1.5
    
    xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
    
    fig.update_layout(scene_camera_eye=dict(x=xe, y=ye, z=ze))
    
    # save image
    now = datetime.now()
    image_name=str(axis)+str(res)+"_"+str(t)+"_t_"+str(now.strftime("%H_%M_%S"))+".png"
#     pio.write_image(fig, image_name,scale=6, width=1080, height=1080)
    
    # show images
  #  plotly.offline.iplot(fig, filename='simple-3d-scatter')
#     print(edge_n)


    return {'fig' : fig, 'img_name' : image_name, 'points' : np.array(new_shape_pd.loc[:])[:,[0,1,2]], 'np_shape':new_shape_pd}



# In[22]:


def get_img_id(path):
    with open(path, 'r') as f:
        id = int(f.readline())
    with open(path, 'w') as f:
        f.write(str(id + 1))
    return id
    
def path_by_imageID(image_id):
    return str(image_id) + '.txt'

def writeData(path, action_v, metrixarr):
    with open(path, 'a') as file:
        file.write('\n')
        file.write('[' + "".join( str(a) + '\n'  for a in action_v[:-1]) + str(action_v[-1]) + ']')
        file.write('\n')
        file.write(str(metrixarr))


# In[23]:


def save_image(new_shape_pd2, axis, t, image_id, metrix_dic, res):
    new_shape_pd=new_shape_pd2.copy()
    x_lines = list()
    y_lines = list()
    z_lines = list()

    if axis=="x":
        new_shape_pd.rename(columns = {'x':'y', 'y':'z','z':'x'}, inplace = True)
    if axis=="y":
        new_shape_pd.rename(columns = {'x':'z', 'y':'x','z':'y'}, inplace = True)
    # new_shape_pd
    
    
    #plotly.offline.init_notebook_mode()

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

#     "REMOVE" background
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

    # "REMOVE" text
    fig.update_layout(font_color="rgba(255, 255, 255,0.0)")
    
    # Rotate
    x_eye = -1.25
    y_eye = 2
    z_eye = 1.5
    
    xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
    
    fig.update_layout(scene_camera_eye=dict(x=xe, y=ye, z=ze))
    
    # save image
    now = datetime.now()
    
    now = datetime.now()
    img_name = str(image_id) + str([x for x in metrix_dic.values()]) + "_" + str(axis)+str(res)+"_"+str(t)+"_t_"+str(now.strftime("%H_%M_%S"))+".png"
    pio.write_image(fig, img_name,scale=6, width=1080, height=1080)


# In[24]:



# Metrix methods start here
def metrix(vectors, setPoints, reso):
    nV = numberOfVoxel(vectors)
    nCV = numberOfCompleteVoxel(vectors)
    nIV = nV - nCV
    nIP_by_voxel = 0
    for vec in vectors:
        if not isComplete(vec) == 1: nIP_by_voxel += sum(vec)
    nIP = nIP_by_voxel - negativeInclinePlaneCounter(setPoints, reso)
    return {'num_voxel' : nV, 'num_complete_voxel' : nCV, 'num_incomplete_voxel' : nIV, 'num_incline_plane' : nIP}


def numberOfVoxel(whole_vector):
    # proving that the vector of a voxel contains one 1 is enough to say that it has at least a wedge
    count = 0
    for vox in range(0, len(whole_vector)):
        for wed in range(0, len(whole_vector[vox])):
            if whole_vector[vox][wed] == 1:
                count += 1
                break
    return count


def numberOfCompleteVoxel(whole_vector):
    count = 0
    for vox in range(0, len(whole_vector)): count += isComplete(whole_vector[vox])
    return count

def numberOfIncompleteVoxel(whole_vector):
    return numberOfVoxel(whole_vector) - numberOfCompleteVoxel(whole_vector)

def inclinePlaneCounter(setPoints, reso):
    counter = 0
    for plane in inclinePlanesByResolution(reso):
        counter += isInclinePlane(reso, setPoints, plane)

    return counter

# Sub methods for complete voxel counter
def isComplete(voxel): # packman + complementary packman or one wedges
    # in the wedge documentation, we can observe that
    # each set of consecutive 4 wedges of a voxel represents rotation of the same wedge
    # and there are 3 sets of these in a voxel
    # we can observe that in one such set, if we let [0, 1, 2, 3] be the indexes,
    # wedge 0 and wedge 2 are complementary; so are wedge 1 and wedge 3
    wed = 2
    while wed < len(voxel):
        if voxel[wed] + voxel[wed - 2] == 2 : return 1
        wed += 1
        if wed % 4 == 0 : wed += 2
    # it is guaranted that the voxel does not have 2 complementary wedges
    # however, it could still be complete
    sumOfVoxel = sum(voxel)
    if sumOfVoxel >= 3:
        return isVoxInCompleteVoxBase(voxel)
    return 0

def isVoxInCompleteVoxBase(voxel):
    completeVoxBase = [[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                       [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0], [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]]

    for vox in completeVoxBase:
        if arr1InArr2(vox, voxel):
            return True
    return False

def arr1InArr2(arr1, arr2):
    num = sum(arr1)
    count = 0
    countMatch = 0
    for ind in range(12):
        if arr1[ind] == 1:
            count += 1
            if arr2[ind] == 1:
                countMatch += 1
        if count == num:
            break
    return countMatch == num


# Sub methods for inline plane counter
def getPoints():
    return np.array(new_shape_pd.loc[:])[:,[0,1,2]]

def inclinePlanesByResolution(reso):
    # reso is the resolution
    posNeg = np.array([1,0,-1])
    result = np.empty((0,4), int)
    for a in posNeg:
        for b in posNeg:
            for c in posNeg:
                if c == -1:
#                     for z0 in range(-reso,reso):
                    for z0 in [-reso, 0, reso]:
#                         for z0 in range(-2*reso,2*reso):
                        result = np.append(result, [[a, b, c, z0]], axis=0)
                else:
#                     for z0 in range(reso,3*reso):
                    for z0 in [reso, 2*reso, 3*reso]:
#                         for z0 in range(0,3*reso):
                        result = np.append(result, [[a, b, c, z0]], axis=0)

    return result


def negativeInclinePlaneCounter(setPoints, reso):
    counter = 0
    for plane in inclinePlanesByResolution(reso):
        counter += negativeInclinePlane(reso, setPoints, plane)

    return counter

def negativeInclinePlane(reso, setPoints, plane):
    zeroCounterPlane = 0
    for i in plane[:-1]:
        if i == 0: zeroCounterPlane += 1
    if zeroCounterPlane >= 2: return 0

    # number of points on a full incline plane in one voxel
    refNP = int(2*(reso**2)) - 1
    # print('reference number = ', refNP)

    nPointsOnPlane = numberPointsOnPlane(setPoints, plane)
    # print('numPoints ', nPointsOnPlane)
    np = nPointsOnPlane
    negativeCounter = 0
    while nPointsOnPlane > refNP/2 :
        nPointsOnPlane -= refNP
        if nPointsOnPlane >= refNP/2 :
            negativeCounter += 1
#     if negativeCounter > 0 :
#         print ('negaIn = ', negativeCounter, ' numP = ', np, ' plane = ', plane)
    return negativeCounter

def numberPointsOnPlane(setPoints, plane):
    # setPoints is 2D numpy array of point arrays of element coordinates
    # plane is numpy array of 4 values x, y, and z coefficients and constant c0
    # create a counter of points on the plane
    counter = 0
    for point in setPoints:
        if pointOnPlane(point, plane):
            counter += 1
    # print(counter)
    return counter

def pointOnPlane(point, plane):
    # point and plane both numpy arrays
    # point is numpy array of 3 values x, y, and z coordinates
    # plane is numpy array of 4 values x, y, and z coefficients and constant c0
    return sum(point * plane[:3]) == plane[3]


# In[ ]:
import sys
RESOLUTION=int(sys.argv[1])
for ID in range(20):
    
    #ID=2
    action_v =[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    for i in range(12):
        action_v[0][i] = np.random.choice([0, 1], p = [0.8, 0.2])
        action_v[1][i] = np.random.choice([0, 1], p = [0.8, 0.2])
       # action_v[2][i] = np.random.choice([0, 1], p = [0.8, 0.2])
       # action_v[3][i] = np.random.choice([0, 1], p = [0.8, 0.2])
       # action_v[4][i] = np.random.choice([0, 1], p = [0.8, 0.2])
     #   action_v[5][i] = np.random.choice([0, 1], p = [0.8, 0.2])
     #   action_v[6][i] = np.random.choice([0, 1], p = [0.8, 0.2])
     #   action_v[7][i] = np.random.choice([0, 1], p = [0.8, 0.2])



    # action_v
    all = draw_shape_from_cloud(action_v,0,"x",RESOLUTION)
    m = metrix(action_v, all['points'], RESOLUTION)
    writeData(str(ID)+'.txt', action_v, m)
    for axis in ("x", "y", "z"):
        print(axis)
        for t in tqdm((0.00,0.25,0.50,0.75,1.00, 1.25,1.50,1.75,2.00,2.25, 2.50,2.75, 3.00, 3.25,3.50,3.75,4.00,4.25,4.50,4.75,5.00,5.25,5.50,5.75,6.00 )):
            
           # all = draw_shape_from_cloud(action_v,t,axis,5)
            save_image(all['np_shape'], axis, t, ID, m, RESOLUTION)

print("DONE")


