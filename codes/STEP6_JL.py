# --------------------------------------REVISIONS---------------------------------------
# Date        Name        Ver#    Description
# --------------------------------------------------------------------------------------
# 01/25/22    J. Li       [1]     Initial creation
# **************************************************************************************

"""
Input: 3D key points of a session
Process: resample; extract index finger position; record frame_index
"""

import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import math
import os
from scipy import interpolate


def return_angle(v1, v2):
    theta = 180 * np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) / math.pi
    return theta


def resample_hand_keypoints(data, frames_list, multiplier, plot=False):
    if multiplier == 1:

        new_frames_list = np.arange(frames_list[0], frames_list[-1] + 1, 1 / multiplier)
        func = interpolate.interp1d(frames_list, data)
        data_interpolated = func(new_frames_list)

        if plot:
            p = 120
            plt.plot(new_frames_list[:p], data_interpolated[:p], 'r')
            plt.plot(frames_list[:p], data[:p], 'b')
            plt.legend(('upsampled', 'original'))
            plt.show()

        return data_interpolated, new_frames_list

    else:

        if (frames_list[-1] - frames_list[0]) % 2 == 0:
            intermediate_frames_list = np.arange(frames_list[0], frames_list[-1] + 1, 1 / multiplier)[:-1]

        else:
            intermediate_frames_list = np.arange(frames_list[0], frames_list[-1], 1 / multiplier)

        # print(frames_list[0], frames_list[-1])
        # print(len(intermediate_frames_list))
        # print(intermediate_frames_list[0], intermediate_frames_list[-1])
        func = interpolate.interp1d(frames_list, data)
        data_interpolated = func(intermediate_frames_list)

        black_screen_begin_frames = []

        for i in range(len(frames_list) - 1):

            if frames_list[i + 1] - frames_list[i] >= 60:
                black_screen_begin_frames.append(i)

        exclude_starts = []
        exclude_ends = []
        counter = 0

        for i in range(len(black_screen_begin_frames)):

            while intermediate_frames_list[counter] <= frames_list[black_screen_begin_frames[i]]:
                counter += 1

            exclude_starts.append(counter)

            while intermediate_frames_list[counter] < frames_list[black_screen_begin_frames[i] + 1]:
                counter += 1

            exclude_ends.append(counter)

        data_segments_list = []
        frame_indices_segments_list = []

        for i in range(len(exclude_starts) + 1):

            if i == 0:
                include_start = 0

            else:
                include_start = exclude_ends[i - 1]

            if i == len(exclude_starts):
                include_end = len(intermediate_frames_list)

            else:
                include_end = exclude_starts[i]

            data_segments_list.append(data_interpolated[include_start:include_end])
            frame_indices_segments_list.append(intermediate_frames_list[include_start:include_end])

        data_interpolated = np.concatenate(tuple(data_segments_list))
        new_frames_list = np.concatenate(tuple(frame_indices_segments_list))

        return data_interpolated, new_frames_list


###################################################
######## SECTION 3: LOAD FRAME INDEX PAIRS ########
###################################################

results_path = '/Users/jiyangli/Library/Mobile Documents/com~apple~CloudDocs/Documents/UB/research/SmartRing/' \
               'virtual IMU/pub_results'
session = 'video_Cory_sc108'
session_path = os.path.join(results_path, session)

f = 5376
outlier_threshold = 50
original_diff = 60  # DO NOT CHANGE THIS
fps = 30

if 'video' in session:  # THIS MEANS THAT WE ARE RUNNING THE CODE FOR A PUBLIC SESSION
    multiplier = 1.5  # THIS IS SUPPOSED TO BE 1.5
    fps = multiplier * fps  # THIS IS SUPPOSED TO BE 45 (BECAUSE 30*1.5)
    diff = 4  # THIS IS SUPPOSED TO BE 4 (BECAUSE WITH MULTIPLIER = 1.5, FRAME INDICES HAVE A STEP SIZE OF 2/3. AND
    # (2/3)*4 = 2.6666, WHICH IS CLOSE TO 3.)

else:
    multiplier = 1  # THIS IS SUPPOSED TO BE 1
    fps = 30  # THIS IS SUPPOSED TO BE 30
    diff = 3  # THIS IS SUPPOSED TO BE 3

final_input_folder = '3d_keypoints_' + str(f)
full_interpolated_folder_path = os.path.join(session_path, final_input_folder + '_interpolated')

if not os.path.isdir(full_interpolated_folder_path):
    os.mkdir(full_interpolated_folder_path)

frame_index_pairs_file = os.path.join(session_path, 'frame_index_pairs.txt')
file = open(frame_index_pairs_file, "r")
frame_index_pairs = file.read()
file.close()

frame_index_pairs = frame_index_pairs.split()
frames_list = np.array([int(frame_index_pairs[i]) for i in range(len(frame_index_pairs)) if i % 2 == 1])

######################################
######## SECTION 4: MAIN CODE ########
######################################

d = os.path.join(session_path, final_input_folder)
count = len(os.listdir(d))
print('Number of frames with 3D hand keypoint results (initially):', count)

all_frames_keypoints = []

for i in range(count):
    file = open(os.path.join(os.path.join(session_path, final_input_folder), str(i + 1) + '.txt'), "r")
    kp_list = file.readlines()
    file.close()

    X = np.reshape(np.array([float(kp.split()[0]) for kp in kp_list]), (42, 1))
    Y = np.reshape(np.array([float(kp.split()[1]) for kp in kp_list]), (42, 1))
    Z = np.reshape(np.array([float(kp.split()[2].split('\n')[0]) for kp in kp_list]), (42, 1))

    all_frames_keypoints.append(np.concatenate((X, Y, Z), axis=1))

all_frames_keypoints = np.array(all_frames_keypoints)

all_frames_keypoints_interpolated = np.zeros((1000000, 42,
                                              3))  # 1 MILLION IS AN ARBITRARY LARGE VALUE. WE JUST NEED A LARGE ENOUGH ARRAY HERE. DON'T DECREASE IT.

for i in range(42):

    for j in range(3):
        new_data, new_frames_list = resample_hand_keypoints(np.reshape(all_frames_keypoints[:, i, j], (-1,)),
                                                            frames_list, multiplier, False)

        all_frames_keypoints_interpolated[:len(new_data), i, j] = new_data

all_frames_keypoints_interpolated = all_frames_keypoints_interpolated[:len(new_data)]

for i in range(len(new_frames_list)):

    file = open(os.path.join(full_interpolated_folder_path, str(i + 1) + '.txt'), 'w')

    frame_keypoints = list(all_frames_keypoints_interpolated[i])

    string_array = np.reshape(np.array([str(k) for j in frame_keypoints for k in j]), (42, 3))

    for j in string_array:
        file.writelines(' '.join(list(j)) + '\n')

    file.close()

with open(os.path.join(session_path, 'frame_index_pairs_new.txt'), 'w') as file:
    file.writelines(' '.join([str(float(i)) for i in new_frames_list]))

pos_prev, pos, pos_next = np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(
    (2, 3))  # position/coordinates of the sensor wrt the camera coordinate system in 3 consecutive frames
positions = np.array([pos_prev, pos, pos_next])

virtual_imu_d = open(os.path.join(session_path, 'virtual_imu_d.txt'), 'w')
virtual_imu_n = open(os.path.join(session_path, 'virtual_imu_n.txt'), 'w')
final_frames = open(os.path.join(session_path, 'final_frames.txt'), 'w')

final_frames_list = []
x_accel1, y_accel1, z_accel1 = [], [], []
x_accel2, y_accel2, z_accel2 = [], [], []

d = full_interpolated_folder_path
count = len(os.listdir(d))
print('Number of frames with 3D hand keypoint results (after interpolation of results):', count)
# print(count - (2*diff))  #21068
# print(len(new_frames_list))  #18643
for i in range(1, count - (2 * diff) + 1):

    imu_x_axis, imu_y_axis, imu_z_axis = np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(
        (2, 3))  # the sensor axes wrt the camera coordinate system
    theta = np.zeros(2)  # angle at joint 6 on the 2 hands
    acc_wrt_cam, acc_wrt_sensor = np.zeros((2, 3)), np.zeros((2, 3))  # we want acc_wrt_sensor in the end
    hands_curr = np.zeros((2, 21, 3))  # all the 3d keypoints of the 2 hands in the current frame

    for j in range(3):

        with open(os.path.join(full_interpolated_folder_path, str(i + diff * j) + '.txt'), 'r') as file:
            kp_list = file.readlines()

        X = np.reshape(-1 * np.array([float(kp.split()[0]) for kp in kp_list]), (42, 1))
        Y = np.reshape(np.array([float(kp.split()[1]) for kp in kp_list]), (42, 1))
        Z = np.reshape(np.array([float(kp.split()[2].split('\n')[0]) for kp in kp_list]), (42, 1))

        X1, Y1, Z1 = X[:21], Y[:21], Z[:21]
        X2, Y2, Z2 = X[21:], Y[21:], Z[21:]

        hands = np.array([np.concatenate((X1, Y1, Z1), axis=1), np.concatenate((X2, Y2, Z2), axis=1)])

        positions[j][0] = (hands[0][5] + hands[0][6]) / 2  # position of the imu sensor on hand1
        positions[j][1] = (hands[1][5] + hands[1][6]) / 2  # position of the imu sensor on hand2

        if j == 1:  # do this for the current frame only

            imu_y_axis[0] = (hands[0][6] - hands[0][5]) / np.linalg.norm(
                hands[0][6] - hands[0][5])  # unit vector along bone 5-6 of hand1
            imu_y_axis[1] = (hands[1][6] - hands[1][5]) / np.linalg.norm(
                hands[1][6] - hands[1][5])  # unit vector along bone 5-6 of hand2

            theta[0] = return_angle(hands[0][5] - hands[0][6], hands[0][7] - hands[0][6])  # angle at joint 5 of hand1
            theta[1] = return_angle(hands[1][5] - hands[1][6], hands[1][7] - hands[1][6])  # angle at joint 5 of hand2

            hands_curr = np.copy(hands)  # store current hands coordinates for later use. copy function is essential!

    e = 0  # threshold for angle at joint 6 in degrees

    if theta[0] <= 180 - e and theta[1] <= 180 - e:  # both hands need to have sufficient angle at joint 5
        # print(i - 1 + 2*diff)
        if new_frames_list[i - 1 + (2 * diff)] - new_frames_list[i - 1 + diff] <= original_diff and \
                new_frames_list[i - 1 + diff] - new_frames_list[i - 1] <= original_diff:

            time_diff = (new_frames_list[i - 1 + (2 * diff)] - new_frames_list[i - 1]) / (fps * 2)
            # delta t = mean of the delta ts of (curr and prev) and (next and curr) frames.
            # Instead of 30, put the fps of the videos.

            acc_wrt_cam = (positions[0] + positions[2] - 2 * positions[1]) / (time_diff ** 2)

            imu_x_axis[0] = np.cross(hands_curr[0][5] - hands_curr[0][6], hands_curr[0][7] - hands_curr[0][6])
            # cross product of vectors: 6->5 x 6->7
            imu_x_axis[0] = imu_x_axis[0] / np.linalg.norm(imu_x_axis[0])

            imu_x_axis[1] = np.cross(hands_curr[1][5] - hands_curr[1][6], hands_curr[1][7] - hands_curr[1][6])
            # cross product of vectors: 6->5 x 6->7
            imu_x_axis[1] = imu_x_axis[1] / np.linalg.norm(imu_x_axis[1])

            imu_z_axis[0] = np.cross(imu_x_axis[0], imu_y_axis[0])
            # simple cross product between x-axis and y-axis gives z-axis
            imu_z_axis[0] = imu_z_axis[0] / np.linalg.norm(imu_z_axis[0])

            imu_z_axis[1] = np.cross(imu_x_axis[1], imu_y_axis[1])
            imu_z_axis[1] = imu_z_axis[1] / np.linalg.norm(imu_z_axis[1])

            R1 = np.zeros((3, 3))  # for hand1
            R2 = np.zeros((3, 3))  # for hand2

            A1 = np.array([imu_x_axis[0], imu_y_axis[0], imu_z_axis[0]])
            # for hand1: matrix according to my algorithm as shared via google doc.

            # b1 is b vector in Ar = b, for finding all the rows r of R1
            for k, b1 in enumerate(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
                R1[k] = np.linalg.solve(A1, b1)  # gaussian elimination

            A2 = np.array([imu_x_axis[1], imu_y_axis[1], imu_z_axis[1]])  # for hand2

            for k, b2 in enumerate(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
                R2[k] = np.linalg.solve(A2, b2)

            acc_wrt_sensor = np.transpose(np.concatenate((np.matmul(R1, np.reshape(acc_wrt_cam[0], (3, 1))),
                                                          np.matmul(R2, np.reshape(acc_wrt_cam[1], (3, 1)))), axis=1))

            x_accel1.append(acc_wrt_sensor[0, 0])
            y_accel1.append(acc_wrt_sensor[0, 1])
            z_accel1.append(acc_wrt_sensor[0, 2])

            x_accel2.append(acc_wrt_sensor[1, 0])
            y_accel2.append(acc_wrt_sensor[1, 1])
            z_accel2.append(acc_wrt_sensor[1, 2])

            l = [str(acc_wrt_sensor[0, j]) for j in range(3)]
            virtual_imu_d.writelines(' '.join(l) + '\n')

            l = [str(acc_wrt_sensor[1, j]) for j in range(3)]
            virtual_imu_n.writelines(' '.join(l) + '\n')

            final_frames_list.append(str(float(new_frames_list[i + diff - 1])))

    #             if i < 201:
    #                 draw_hands(np.concatenate((hands_curr[0][:,0], hands_curr[1][:,0]), axis=0), np.concatenate((hands_curr[0][:,1], hands_curr[1][:,1]), axis=0),
    #                        np.concatenate((hands_curr[0][:,2], hands_curr[1][:,2]), axis=0), positions, imu_x_axis, imu_y_axis, imu_z_axis, i)

virtual_imu_d.close()
virtual_imu_n.close()

final_frames.writelines(' '.join(final_frames_list))
final_frames.close()

print("Total final frames before removing acceleration outliers:", len(final_frames_list))

fig, axes = plt.subplots(6, 1)

axes[0].hist(x_accel1, 100)
axes[1].hist(y_accel1, 100)
axes[2].hist(z_accel1, 100)

axes[3].hist(x_accel2, 100)
axes[4].hist(y_accel2, 100)
axes[5].hist(z_accel2, 100)

fig.tight_layout()

virtual_imu_d = open(os.path.join(session_path, 'virtual_imu_d.txt'), 'r')
virtual_imu_n = open(os.path.join(session_path, 'virtual_imu_n.txt'), 'r')

imu_hand1 = np.reshape(np.array([float(j) for i in virtual_imu_d.read().split('\n') for j in i.split()]),
                       (len(final_frames_list), 3))
imu_hand2 = np.reshape(np.array([float(j) for i in virtual_imu_n.read().split('\n') for j in i.split()]),
                       (len(final_frames_list), 3))

virtual_imu_d.close()
virtual_imu_n.close()

virtual_imu_d = open(
    os.path.join(session_path, 'virtual_imu_d_postprocessed_' + str(outlier_threshold) + '_threshold.txt'), 'w')
virtual_imu_n = open(
    os.path.join(session_path, 'virtual_imu_n_postprocessed_' + str(outlier_threshold) + '_threshold.txt'), 'w')
final_frames = open(
    os.path.join(session_path, 'final_frames_postprocessed_' + str(outlier_threshold) + '_threshold.txt'), 'w')

final_frames_list_postprocessed = []
x_accel1_postprocessed, y_accel1_postprocessed, z_accel1_postprocessed = [], [], []
x_accel2_postprocessed, y_accel2_postprocessed, z_accel2_postprocessed = [], [], []

for i in range(len(final_frames_list)):

    include_frame = 1

    for u in range(3):

        if np.abs(imu_hand1[i, u]) > outlier_threshold and i != 0 and i != len(final_frames_list) - 1:

            if np.abs(imu_hand1[i + 1, u]) <= outlier_threshold and np.abs(imu_hand1[i - 1, u]) <= outlier_threshold:

                imu_hand1[i, u] = (imu_hand1[i - 1, u] + imu_hand1[i + 1, u]) / 2

            else:

                include_frame = 0
                break

        if np.abs(imu_hand2[i, u]) > outlier_threshold and i != 0 and i != len(final_frames_list) - 1:

            if np.abs(imu_hand2[i + 1, u]) <= outlier_threshold and np.abs(imu_hand2[i - 1, u]) <= outlier_threshold:

                imu_hand2[i, u] = (imu_hand2[i - 1, u] + imu_hand2[i + 1, u]) / 2

            else:

                include_frame = 0
                break

    if np.max(np.array([np.max(imu_hand1[i]), np.max(imu_hand2[i])])) > outlier_threshold:
        continue

    if include_frame:
        x_accel1_postprocessed.append(imu_hand1[i, 0])
        z_accel1_postprocessed.append(imu_hand1[i, 1])
        y_accel1_postprocessed.append(imu_hand1[i, 2])

        x_accel2_postprocessed.append(imu_hand2[i, 0])
        z_accel2_postprocessed.append(imu_hand2[i, 1])
        y_accel2_postprocessed.append(imu_hand2[i, 2])

        l = [str(imu_hand1[i, j]) for j in range(3)]
        virtual_imu_d.writelines(' '.join(l) + '\n')

        l = [str(imu_hand2[i, j]) for j in range(3)]
        virtual_imu_n.writelines(' '.join(l) + '\n')

        final_frames_list_postprocessed.append(str(final_frames_list[i]))

final_frames.writelines(' '.join(final_frames_list_postprocessed))

print("Total final frames after removing acceleration outliers: ", len(final_frames_list_postprocessed))

virtual_imu_d.close()
virtual_imu_n.close()
final_frames.close()

fig, axes = plt.subplots(6, 1)

axes[0].hist(x_accel1_postprocessed, 100)
axes[1].hist(y_accel1_postprocessed, 100)
axes[2].hist(z_accel1_postprocessed, 100)

axes[3].hist(x_accel2_postprocessed, 100)
axes[4].hist(y_accel2_postprocessed, 100)
axes[5].hist(z_accel2_postprocessed, 100)

fig.tight_layout()
