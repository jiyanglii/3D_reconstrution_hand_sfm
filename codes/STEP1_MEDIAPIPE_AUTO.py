# --------------------------------------REVISIONS---------------------------------------
# Date        Name        Ver#    Description
# --------------------------------------------------------------------------------------
# 08/05/21    S. Shah     [1]     Initial Creation
# 08/06/21    J. Li       [2]     First Review
# 10/08/21    D. Wang     [3]     Adopting multiprocessing
# **************************************************************************************

"""
Run MediaPipe on pre-processed videos. Refer to 'https://google.github.io/mediapipe/solutions/hands'
Input: pre-processed 2-view videos
Output: 'kp_2.5D.pickle', 'handedness.pickle', 'dim_and_crop.txt'
"""
from multiprocessing import Process
import pickle
import numpy as np
import os
import cv2
import math
import mediapipe as mp
import PATH
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pdb
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

inputs_path = PATH.inputs_path
results_path = PATH.results_path

def run_video(video):
    #################################################################################################################
    ######## SECTION 1: SETTING VARIOUS FILE NAMES AND PATHS, AND CREATING NECESSARY DIRECTORIES USING MKDIR ########
    #################################################################################################################
    video_name = str(video.split('.')[:-1][0])
    video_name = video_name.split('/')[-1]

    session_results_path = os.path.join(results_path, session)
    if not os.path.isdir(session_results_path):
        os.mkdir(session_results_path)

    video_results_path = os.path.join(session_results_path, video_name)
    if not os.path.isdir(video_results_path):
        os.mkdir(video_results_path)

    save_folder = os.path.join(video_results_path, 'images')
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    kp_folder = video_results_path + '/kp_2.5D.pickle'
    handedness_folder = video_results_path + '/handedness.pickle'
    dim_and_crop_folder = video_results_path + '/dim_and_crop.txt'

    #################################################################################################################
    ######## SECTION 2: FINDING THE CORRECT CROPPING FOR THE VIDEO TO FOCUS ONLY ON THE HUMAN PERFORMING ASL ########
    #################################################################################################################

    frames = cv2.VideoCapture(video)
    image_height, image_width, _ = [0, 0, 0]

    idx = 0
    up = math.inf
    down = -math.inf
    left = math.inf
    right = -math.inf

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while frames.isOpened():

            success, full_image = frames.read()
            if not success:
                break

            image = full_image
            img_in = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
            results = hands.process(img_in)

            image_height, image_width, _ = img_in.shape

            if results.multi_hand_landmarks:
                for single_hand_landmarks in results.multi_hand_landmarks:

                    up_temp = math.floor(min([single_hand_landmarks.landmark[i].y for i in range(21)]) * image_height)
                    if up_temp < up:
                        up = up_temp

                    down_temp = math.ceil(max([single_hand_landmarks.landmark[i].y for i in range(21)]) * image_height)
                    if down_temp > down:
                        down = down_temp

                    right_temp = math.floor(min([single_hand_landmarks.landmark[i].x for i in range(21)]) * image_width)
                    if right_temp < image_width - right:
                        right = image_width - right_temp

                    left_temp = math.ceil(max([single_hand_landmarks.landmark[i].x for i in range(21)]) * image_width)
                    if left_temp > image_width - left:
                        left = image_width - left_temp

            idx = idx + 1

            print(idx)

    crop_row_up = up - 5
    crop_row_down = down + 5
    crop_col_left = left - 5
    crop_col_right = right + 5

    if crop_row_up < 0:
        crop_row_up = 0

    if crop_row_down > image_height - 1:
        crop_row_down = image_height - 1

    if crop_col_left < 0:
        crop_col_left = 0

    if crop_col_right > image_width - 1:
        crop_col_right = image_width - 1

    ###############################################################################################################################
    ######## SECTION 3: GET THE KEYPOINTS, HANDEDNESS OF ALL THE FRAMES OF THE VIDEO AND SAVE 200 INITIAL ANNOTATED IMAGES ########
    ###############################################################################################################################

    frames = cv2.VideoCapture(video)
    idx = 1
    all_kp_dict, all_side_dict = {}, {}
    full_image_height, full_image_width, _ = [0, 0, 0]

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while (frames.isOpened()):

            success, full_image = frames.read()
            if success == False:
                break

            image = full_image[crop_row_up: crop_row_down, crop_col_left: crop_col_right]
            img_in = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
            results = hands.process(img_in)

            image_height, image_width, _ = img_in.shape
            full_image_height, full_image_width, _ = full_image.shape

            key = f'frame_{idx}'
            all_kp_dict[key] = results.multi_hand_landmarks
            all_side_dict[key] = results.multi_handedness

            if idx < 201:
                annotated_image = img_in.copy()

                if results.multi_hand_landmarks:
                    for single_hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(annotated_image, single_hand_landmarks, mp_hands.HAND_CONNECTIONS)

                annotated_image = cv2.flip(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 1)
                full_annotated_image = full_image
                full_annotated_image[crop_row_up: crop_row_down, crop_col_left: crop_col_right] = annotated_image
                full_annotated_image = cv2.flip(full_annotated_image, 1)
                cv2.imwrite(save_folder + f'{idx}.png', full_annotated_image)

            if type(all_kp_dict[key]) is list:
                n = len(all_kp_dict[key])
                for i in range(n):
                    for k in range(21):
                        all_kp_dict[key][i].landmark[k].x = (math.floor(all_kp_dict[key][i].landmark[
                                                                            k].x * image_width) + full_image_width - crop_col_right + 1) / full_image_width
                        all_kp_dict[key][i].landmark[k].y = (math.floor(
                            all_kp_dict[key][i].landmark[k].y * image_height) + crop_row_up) / full_image_height

            idx = idx + 1

    #############################################################################################################
    ######## SECTION 4: SAVE THE KEYPOINTS AND HANDEDNESS OF ALL THE FRAMES OF THE VIDEO IN PICKLE FILES ########
    #############################################################################################################

    with open(kp_folder, 'wb') as f:
        pickle.dump(all_kp_dict, f)

    with open(handedness_folder, 'wb') as f:
        pickle.dump(all_side_dict, f)

    f = open(dim_and_crop_folder, "w")
    f.write(str(full_image_height) + ' ' + str(full_image_width) + ' ' + str(crop_row_up) + ' ' + str(
        crop_row_down) + ' ' + str(crop_col_left) + ' ' + str(crop_col_right))
    f.close()


def run(session):  # session is the folder name of two videos

    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    videos_dir = os.path.join(inputs_path, session)
    videos = [os.path.join(videos_dir, 's01_front_s3_20210410_fps_sync.mp4'),
              os.path.join(videos_dir, 's01_side_s3_20210410_fps_sync.mp4')]

    process_list = []
    if not os.path.isdir(videos[0]) and os.path.isdir(videos[1]):
        print("Two-view videos are required to run this function!")
        return

    for video in videos:

        process_list.append(Process(target=run_video, args=(video,)))
        process_list[-1].start()

    for p in process_list:
        p.join()

if __name__ == "__main__":
    session = 's01_Session_3'
    run(session)
