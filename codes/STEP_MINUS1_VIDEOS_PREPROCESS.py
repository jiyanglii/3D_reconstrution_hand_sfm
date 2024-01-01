# THIS PREPROCESSING SCRIPT IS NOT REQUIRED FOR THE PUBLIC ASL DATASET AND HENCE, SHOULD BE RUN ONLY FOR SELF-COLLECTED ASL SESSIONS

# BOTH THE CAMERAS OF THE SELF-COLLECTED ASL SESSIONS HAVE DIFFERENT FPS (30 AND 60 RESPECTIVELY) AND VIDEO RESOLUTIONS. THIS SCRIPT CHANGES ALL THE VIDEOS TO 30 FPS AND THEIR RESOLUTION TO WHICHEVER VIDEO HAS LOWER RESOLUTION.


import numpy as np
import cv2
import math
from moviepy.editor import *

# edit this list named 'sessions' to contain only the asl sessions that you want to preprocess
sessions = ['Hanna_Session_5']
video1 = 'left'
video2 = 'right'
inputs_path = '/Users/sidoodler/projects/asl/datasets/video_asl/inputs'

for session in sessions:
    
    #take care of the video file extensions (.MOV and mp4)
    filename1 = f'{inputs_path}/asl_ours/{session}/{video1}.MOV'
    filename2 = f'{inputs_path}/asl_ours/{session}/{video2}.mp4'
    
    clip1 = VideoFileClip(filename1)
    clip2 = VideoFileClip(filename2)
    
    # comment the code below this print statement to know whether the 2 views have the same fps. If yes, then no need to create same_fps_video.mp4.
    print(clip1.fps, clip2.fps)
    
    same_fps = min(clip1.fps, clip2.fps)
    
    clip1 = VideoFileClip(filename1).set_fps(same_fps)
    clip2 = VideoFileClip(filename2).set_fps(same_fps)
    
    same_fps_filename1 = f'{inputs_path}/asl_ours/{session}/same_fps_{video1}.mp4'
    same_fps_filename2 = f'{inputs_path}/asl_ours/{session}/same_fps_{video2}.mp4'
    
    clip1.write_videofile(same_fps_filename1, fps=clip1.fps)
    clip2.write_videofile(same_fps_filename2, fps=clip2.fps)
    
# USE THE BELOW CODE IF THE RESOLUTIONS OF THE TWO VIEWS ARE DIFFERENT (ALL HANNA SESSIONS HAVE THE SAME RESOLUTION 1920x1080)

#     same_fps_frames = np.array([cv2.VideoCapture(same_fps_filename1), cv2.VideoCapture(same_fps_filename2)])
#     vid_heights = np.zeros(2)
#     vid_widths = np.zeros(2)
    
#     for i in range(2):

#         while(same_fps_frames[i].isOpened()):

#             success, image = same_fps_frames[i].read()
#             if success == False:
#                 break

#             vid_heights[i], vid_widths[i], _ = image.shape
#             break

#     save_path1 = f'{inputs_path}/asl_ours/{session}/final_{video1}.mp4'
#     out1 = cv2.VideoWriter(save_path1, cv2.VideoWriter_fourcc(*'mp4v'), clip1.fps, (int(vid_widths[1]), int(vid_heights[1])))
    
#     save_path2 = f'{inputs_path}/asl_ours/{session}/final_{video2}.mp4'
#     out2 = cv2.VideoWriter(save_path2, cv2.VideoWriter_fourcc(*'mp4v'), clip2.fps, (int(vid_widths[1]), int(vid_heights[1])))
    
#     for i in range(2):

#         while(same_fps_frames[i].isOpened()):

#             success, image = same_fps_frames[i].read()
#             if success == False:
#                 break

#             if vid_heights[i] > vid_heights[(i+1)%2]:
                
#                 channel0 = cv2.resize(image[:,:,0], (int(vid_widths[(i+1)%2]), int(vid_heights[(i+1)%2])))
#                 channel1 = cv2.resize(image[:,:,1], (int(vid_widths[(i+1)%2]), int(vid_heights[(i+1)%2])))
#                 channel2 = cv2.resize(image[:,:,2], (int(vid_widths[(i+1)%2]), int(vid_heights[(i+1)%2])))
                
#                 channel0 = np.reshape(channel0, (int(vid_heights[(i+1)%2]), int(vid_widths[(i+1)%2]), 1))
#                 channel1 = np.reshape(channel1, (int(vid_heights[(i+1)%2]), int(vid_widths[(i+1)%2]), 1))
#                 channel2 = np.reshape(channel2, (int(vid_heights[(i+1)%2]), int(vid_widths[(i+1)%2]), 1))
                
#                 image = np.concatenate((channel0, channel1, channel2), axis=2)

#             if i == 0:
#                 out1.write(image)

#             else:
#                 out2.write(image)
        
#         if i == 0:
#             out1.release()
            
#         if i == 1:
#             out2.release()