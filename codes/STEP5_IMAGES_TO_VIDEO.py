import pickle
import numpy as np
import cv2
import math

##################################################################################################
######## OPTIONAL CODE: CREATE A VIDEO OF THE IMAGES OF 3D KEYPOINTS GENERATED PREVIOUSLY ########
##################################################################################################

session = 'Caroline_Session_6'
results_path = '/Users/sidoodler/projects/asl/datasets/video_asl/results'
keypoint_folder = '3d_keypoints_450'
figure_folder = 'final_keypoint_figures_450'

img_list = []

count = 0
d = f"{results_path}/{session}/{keypoint_folder}/"
for path in os.listdir(d):
    if os.path.isfile(os.path.join(d, path)):
        count += 1
print count

for i in range(count):

    filename = f'{results_path}/{session}/{figure_folder}/{i+1}.png'
    img = cv2.imread(filename)
    h, w, c = img.shape
    size = (w, h)
    img_list.append(img)
    
save_path = f'{results_path}/{session}/final_videos'

if os.path.isdir(save_path) == False:
    os.mkdir(save_path)

save_path = f'{results_path}/{session}/final_videos/5fps.avi'
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_list)):
    out.write(img_list[i])
out.release()









# img_list = []

# for i in range(100): #################################################

#     filename1 = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_keypoint_figures_not_averaged_not_scaled/{i+1}.png'
#     filename2 = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_keypoint_figures_not_averaged_not_scaled/{i+1}_new.png'
#     img1 = cv2.imread(filename1)
#     img2 = cv2.imread(filename2)
#     img = np.concatenate((img1,img2), axis=1)
#     h, w, c = img.shape
#     size = (w, h)
#     img_list.append(img)
    
# save_path = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_videos_not_averaged_not_scaled/2_views_100frames_2_fps.avi'
# out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
 
# for i in range(len(img_list)):
#     out.write(img_list[i])
# out.release()

# ###########################################################################################################################
# img_list = []

# for i in range(100): #################################################

#     filename1 = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_keypoint_figures_averaged_not_scaled/{i+1}.png'
#     filename2 = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_keypoint_figures_averaged_not_scaled/{i+1}_new.png'
#     img1 = cv2.imread(filename1)
#     img2 = cv2.imread(filename2)
#     img = np.concatenate((img1,img2), axis=1)
#     h, w, c = img.shape
#     size = (w, h)
#     img_list.append(img)
    
# save_path = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_videos_averaged_not_scaled_2/2_views_100frames_2_fps.avi'
# out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
 
# for i in range(len(img_list)):
#     out.write(img_list[i])
# out.release()

# ###########################################################################################################################
# img_list = []

# for i in range(100): #################################################

#     filename1 = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_keypoint_figures_not_averaged_scaled/{i+1}.png'
#     filename2 = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_keypoint_figures_not_averaged_scaled/{i+1}_new.png'
#     img1 = cv2.imread(filename1)
#     img2 = cv2.imread(filename2)
#     img = np.concatenate((img1,img2), axis=1)
#     h, w, c = img.shape
#     size = (w, h)
#     img_list.append(img)
    
# save_path = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_videos_not_averaged_scaled_2/2_views_100frames_2_fps.avi'
# out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
 
# for i in range(len(img_list)):
#     out.write(img_list[i])
# out.release()

# ###########################################################################################################################
# img_list = []

# for i in range(100): #################################################

#     filename1 = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_keypoint_figures_averaged_scaled/{i+1}.png'
#     filename2 = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_keypoint_figures_averaged_scaled/{i+1}_new.png'
#     img1 = cv2.imread(filename1)
#     img2 = cv2.imread(filename2)
#     img = np.concatenate((img1,img2), axis=1)
#     h, w, c = img.shape
#     size = (w, h)
#     img_list.append(img)
    
# save_path = f'/Users/sidoodler/projects/asl/datasets/video_asl/results/{session}/{video}/final_videos_averaged_scaled_2/2_views_100frames_2_fps.avi'
# out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
 
# for i in range(len(img_list)):
#     out.write(img_list[i])
# out.release()