# --------------------------------------REVISIONS---------------------------------------
# Date        Name        Ver#    Description
# --------------------------------------------------------------------------------------
# 08/05/21    S. Shah     [1]     Initial Creation
# 08/15/21    J. Li       [2]     First Review
# **************************************************************************************
"""
Input: pickle files from STEP1
Output: "frame_index_pairs.txt" and "corresponding_points.txt" for STEP3
"""

import pickle
import PATH
import os

######################################################################################
######## SECTION 1: LOAD THE PICKLE FILES CONTAINING KEYPOINTS AND HANDEDNESS ########
######################################################################################

results_path = PATH.results_path

class PrerequisitesMatlab:

    def __init__(self, session, video_1, video_2):
        self.session_folder = os.path.join(results_path, session)
        self.video1_folder = os.path.join(self.session_folder, video_1)
        self.video2_folder = os.path.join(self.session_folder, video_2)
        self.corresponding_landmarks = []


    def return_pickles(self, path):

        handedness_file = os.path.join(path, 'handedness.pickle')
        keypoints_file = os.path.join(path, 'kp_2.5D.pickle')

        with open(keypoints_file,'rb') as file:
            keypoints = pickle.load(file)

        with open(handedness_file,'rb') as file:
            handedness = pickle.load(file)

        return keypoints, handedness

    #################################################################################################################################
    ######## SECTION 2: ORGANISE ALL THE KEYPOINTS OF ALL THE FRAMES FROM BOTH THE CAMERAS IN corresponding_landmarks OBJECT ########
    #################################################################################################################################

    def return_landmarks(self):

        both_hands = 0
        both_types_of_hands = 0
        some_hands = 0

        keypoints1, handedness1 = self.return_pickles(self.video1_folder)
        keypoints2, handedness2 = self.return_pickles(self.video2_folder)

        for i in range(len(keypoints1)):

            if type(handedness1[f'frame_{i+1}']) is list and type(handedness2[f'frame_{i+1}']) is list:

                some_hands += 1

                num_hands_1 = len(handedness1[f'frame_{i+1}'])
                num_hands_2 = len(handedness2[f'frame_{i+1}'])

                if num_hands_1 == 2 and num_hands_2 == 2:

                    #both frames have the presence of 2 hands
                    both_hands += 1

                    if (handedness1[f'frame_{i+1}'][0].classification[0].label != handedness1[f'frame_{i+1}'][1].classification[0].label) and (handedness2[f'frame_{i+1}'][0].classification[0].label != handedness2[f'frame_{i+1}'][1].classification[0].label):

                        #both frames have both types of hands, right and left
                        both_types_of_hands += 1

                        self.corresponding_landmarks.append((f'frame_{i+1}', {}))

                        if handedness1[f'frame_{i+1}'][0].classification[0].label == handedness2[f'frame_{i+1}'][0].classification[0].label:

                            #matching indices
                            hand1_corresponding_keypoints_list = []
                            hand2_corresponding_keypoints_list = []

                            for k in range(21):

                                hand1_corresponding_keypoints_list.append((keypoints1[f'frame_{i+1}'][0].landmark[k], keypoints2[f'frame_{i+1}'][0].landmark[k]))
                                hand2_corresponding_keypoints_list.append((keypoints1[f'frame_{i+1}'][1].landmark[k], keypoints2[f'frame_{i+1}'][1].landmark[k]))

                            self.corresponding_landmarks[-1][1][handedness1[f'frame_{i+1}'][0].classification[0].label] = hand1_corresponding_keypoints_list
                            self.corresponding_landmarks[-1][1][handedness1[f'frame_{i+1}'][1].classification[0].label] = hand2_corresponding_keypoints_list

                        else:

                            #reversed indices
                            hand1_corresponding_keypoints_list = []
                            hand2_corresponding_keypoints_list = []

                            for k in range(21):

                                hand1_corresponding_keypoints_list.append((keypoints1[f'frame_{i+1}'][0].landmark[k], keypoints2[f'frame_{i+1}'][1].landmark[k]))
                                hand2_corresponding_keypoints_list.append((keypoints1[f'frame_{i+1}'][1].landmark[k], keypoints2[f'frame_{i+1}'][0].landmark[k]))

                            self.corresponding_landmarks[-1][1][handedness1[f'frame_{i+1}'][0].classification[0].label] = hand1_corresponding_keypoints_list
                            self.corresponding_landmarks[-1][1][handedness1[f'frame_{i+1}'][1].classification[0].label] = hand2_corresponding_keypoints_list

                    else:

                        #when both hands are same
                        left_hand_cam1 = 0
                        left_hand_cam2 = 0

                        x0_hand_1_cam1 = keypoints1[f'frame_{i+1}'][0].landmark[0].x
                        x0_hand_2_cam1 = keypoints1[f'frame_{i+1}'][1].landmark[0].x
                        x0_hand_1_cam2 = keypoints2[f'frame_{i+1}'][0].landmark[0].x
                        x0_hand_2_cam2 = keypoints2[f'frame_{i+1}'][1].landmark[0].x

                        if x0_hand_1_cam1 > x0_hand_2_cam1 and x0_hand_1_cam2 > x0_hand_2_cam2:

                            left_hand_cam1 = 1
                            left_hand_cam2 = 1

                        elif x0_hand_1_cam1 > x0_hand_2_cam1 and x0_hand_1_cam2 < x0_hand_2_cam2:

                            left_hand_cam1 = 1

                        elif x0_hand_1_cam1 < x0_hand_2_cam1 and x0_hand_1_cam2 > x0_hand_2_cam2:

                            left_hand_cam2 = 1

                        self.corresponding_landmarks.append((f'frame_{i+1}', {}))

                        handleft_corresponding_keypoints_list = []
                        handright_corresponding_keypoints_list = []

                        for k in range(21):

                            handleft_corresponding_keypoints_list.append((keypoints1[f'frame_{i+1}'][left_hand_cam1].landmark[k], keypoints2[f'frame_{i+1}'][left_hand_cam2].landmark[k]))
                            handright_corresponding_keypoints_list.append((keypoints1[f'frame_{i+1}'][(left_hand_cam1 + 1)%2].landmark[k], keypoints2[f'frame_{i+1}'][(left_hand_cam2 + 1)%2].landmark[k]))

                        self.corresponding_landmarks[-1][1]['Left'] = handleft_corresponding_keypoints_list
                        self.corresponding_landmarks[-1][1]['Right'] = handright_corresponding_keypoints_list

    ####################################################################################################
    ######## SECTION 3: LOAD dim_and_crop.txt WHICH CONTAINS INFORMATION ABOUT IMAGE DIMENSIONS ########
    ####################################################################################################

    def return_dim_crop(self):
        dim_and_crop = open(os.path.join(self.video1_folder, 'dim_and_crop.txt'), "r")
        dim = [int(i) for i in dim_and_crop.read().split()[:2]]

        im_height = dim[0]
        im_width = dim[1]

        for frame, hands in self.corresponding_landmarks:

            for k in range(21):

                for i in [0, 1]:

                    hands['Left'][k][i].y = round(hands['Left'][k][i].y*im_height)
                    hands['Left'][k][i].x = round(hands['Left'][k][i].x*im_width)
                    hands['Right'][k][i].y = round(hands['Right'][k][i].y*im_height)
                    hands['Right'][k][i].x = round(hands['Right'][k][i].x*im_width)

    #######################################################################################################
    ######## SECTION 4: STORE ALL CORRESPONDING LANDMARKS IN A TXT FILE FOR USE BY THE MATLAB CODE ########
    #######################################################################################################
    def write_landmarks_frame(self):
        s = ''

        for i in range(len(self.corresponding_landmarks)):

            for j in range(21):

                s += str(self.corresponding_landmarks[i][1]['Left'][j][0].x)
                s += ' '
                s += str(self.corresponding_landmarks[i][1]['Left'][j][0].y)
                s += ' '
                s += str(self.corresponding_landmarks[i][1]['Left'][j][1].x)
                s += ' '
                s += str(self.corresponding_landmarks[i][1]['Left'][j][1].y)
                s += ' '

            for j in range(21):

                s += str(self.corresponding_landmarks[i][1]['Right'][j][0].x)
                s += ' '
                s += str(self.corresponding_landmarks[i][1]['Right'][j][0].y)
                s += ' '
                s += str(self.corresponding_landmarks[i][1]['Right'][j][1].x)
                s += ' '
                s += str(self.corresponding_landmarks[i][1]['Right'][j][1].y)
                s += ' '

        # f = open(os.path.join(self.session_folder, "corresponding_points.txt"), "w")
        # f.write(s)
        # f.close()

    #####################################################################################################
    ######## SECTION 5: STORE FRAME INDICES AND NUMBERS IN A TXT FILE FOR USE BY THE MATLAB CODE ########
    #####################################################################################################

        s = ''

        for i in range(len(self.corresponding_landmarks)):

            s += str(i)
            s += ' '
            s += str(self.corresponding_landmarks[i][0].split('_')[1])
            s += ' '

        # f = open(os.path.join(self.session_folder, "frame_index_pairs.txt"), "w")
        # f.write(s)
        # f.close()


if __name__ == "__main__":
    session = 's01_Session_2'
    video1 = 's01_front_s2_20210410_fps_sync'
    video2 = 's01_side_s2_20210410_fps_sync'

    output = PrerequisitesMatlab(session, video1, video2)
    output.return_landmarks()
    output.return_dim_crop()
    output.write_landmarks_frame()

