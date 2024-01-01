import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import math
import seaborn as sns
import os

######################################################
######## SECTION 1: FUNCTION FOR DRAWING LINE ########
######################################################

def draw_line(p1, p2, ax, colour):
    
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    z1, z2 = p1[2], p2[2]
    
    n = 100
    
    x = np.linspace(x1, x2, n)
    y = np.linspace(y1, y2, n)
    z = np.linspace(z1, z2, n)
    
    ax.scatter(x,y,z, s=7, c=colour)

#################################################################################
######## SECTION 2: LOAD 3D KEYPOINTS, SAVE 3D FIGURES FOR VISUALISATION ########
#################################################################################

sns.color_palette("husl", 8)

results_path = '/Users/sidoodler/projects/asl/datasets/video_asl/results'
session = 'video_Cory_sc110'
f = 1450
final_input_folder = f'3d_keypoints_{f}'
final_output_folder = f'final_keypoint_figures_{f}'

if os.path.isdir(f'{results_path}/{session}/{final_output_folder}') == False:
    os.mkdir(f'{results_path}/{session}/{final_output_folder}')

count = 0
d = f"{results_path}/{session}/{final_input_folder}/"

for path in os.listdir(d):
    if os.path.isfile(os.path.join(d, path)):
        count += 1
print(count)

for i in range(300):
    
    file = open(f"{results_path}/{session}/{final_input_folder}/{i+1}.txt", "r")
    kp_list = file.readlines()
    file.close()
    
    X = np.array([float(kp.split(' ')[0]) for kp in kp_list])
    Y = np.array([float(kp.split(' ')[1]) for kp in kp_list])
    Z = np.array([float(kp.split(' ')[2].split('\n')[0]) for kp in kp_list])

    sns.set(style = "darkgrid")
    
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111, projection = '3d', xlabel = 'X - AXIS', ylabel = 'Y - AXIS', zlabel = 'Z - AXIS', xlim = (-50, 50), ylim = (-50, 50), zlim = (-50, 50), autoscale_on = False, autoscalex_on = False, autoscaley_on = False, autoscalez_on = False)
    
#     ax.set_xbound(-1,1)
#     ax.set_ybound(-1,1)
#     ax.set_zbound(3.75,4.25)
    
    x, y, z = X, Y, Z
    if i == 0:
        print(x)
    ax.scatter(x[:6], y[:6], z[:6], s=25, c='b')
    ax.scatter(x[6], y[6], z[6], s=60, c='r')
    ax.scatter(x[7:27], y[7:27], z[7:27], s=25, c='b')
    ax.scatter(x[27], y[27], z[27], s=60, c='r')
    ax.scatter(x[28:], y[28:], z[28:], s=25, c='b')

    for j,k in [(0,1), (1,2), (2,3), (3,4)]:

        draw_line((X[j],Y[j],Z[j]), (X[k],Y[k],Z[k]), ax, 'k')
        draw_line((X[j+21],Y[j+21],Z[j+21]), (X[k+21],Y[k+21],Z[k+21]), ax, 'k')
    
    for j,k in [(0,5), (5,6), (6,7), (7,8)]:
        
        draw_line((X[j],Y[j],Z[j]), (X[k],Y[k],Z[k]), ax, 'g')
        draw_line((X[j+21],Y[j+21],Z[j+21]), (X[k+21],Y[k+21],Z[k+21]), ax, 'g')
        
    for j,k in [(0,9), (9,10), (10,11), (11,12)]:
        
        draw_line((X[j],Y[j],Z[j]), (X[k],Y[k],Z[k]), ax, 'm')
        draw_line((X[j+21],Y[j+21],Z[j+21]), (X[k+21],Y[k+21],Z[k+21]), ax, 'm')
        
    for j,k in [(0,13), (13,14), (14,15), (15,16)]:
        
        draw_line((X[j],Y[j],Z[j]), (X[k],Y[k],Z[k]), ax, 'c')
        draw_line((X[j+21],Y[j+21],Z[j+21]), (X[k+21],Y[k+21],Z[k+21]), ax, 'c')
        
    for j,k in [(0,17), (17,18), (18,19), (19,20)]:
    
        draw_line((X[j],Y[j],Z[j]), (X[k],Y[k],Z[k]), ax, 'r')
        draw_line((X[j+21],Y[j+21],Z[j+21]), (X[k+21],Y[k+21],Z[k+21]), ax, 'r')
    
    # SET ELEV=270 FOR FRONT VIEW OF THE 3D HAND OR ELEV=0 FOR TOP VIEW OF THE 3D HAND. IN BOTH CASES AZIM=270.
    ax.view_init(elev=270, azim=270)
    ax.autoscale_view(scalex = False, scaley = False)
    plt.savefig(f"{results_path}/{session}/{final_output_folder}/{i+1}_front.png")
    plt.close('all')
    
