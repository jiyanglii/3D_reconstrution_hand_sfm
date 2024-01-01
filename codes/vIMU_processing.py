# --------------------------------------REVISIONS---------------------------------------
# Date        Name        Ver#    Description
# --------------------------------------------------------------------------------------
# 08/15/21    J. Li       [1]     Initial creation
# **************************************************************************************

"""
Input: 'virtual_imu_n.txt' of a session
Process: up-sampling, plot, comparison with real IMU
"""

import os
import pandas as pd
import PATH
import numpy as np
import matplotlib.pyplot as plt
import math


inputs_path = PATH.inputs_path
results_path = PATH.results_path

fps = 30
SMP_RATE = 178
points_per_frame = 8*21
index_joints = [5, 6, 7, 8]

class JointMove():
    def __init__(self, session):
        self.file_path = os.path.join(results_path, session)
        self.x_left_cam1 = []
        self.y_left_cam1 = []
        self.x_left_cam2 = []
        self.y_left_cam2 = []

        self.x_right_cam1 = []
        self.y_right_cam1 = []
        self.x_right_cam2 = []
        self.y_right_cam2 = []
        self.session = session


    def landmark_xy(self):

        points_path = os.path.join(self.file_path, 'corresponding_points.txt')
        with open(points_path, 'r') as file:
            points = file.read()
        file.close()
        points = points.split(' ')
        point_list = []
        idx = 0
        for i in points:
            try:
                point_list.append(float(i))
            except ValueError as e:
                print("error", e, "on line", idx)
            idx += 1
        point_list = np.array(point_list)
        point_list = point_list.reshape(int(len(point_list)/points_per_frame), points_per_frame)


        point_left = point_list[:, 0:int(points_per_frame/2)]

        point_right = point_list[:, int(points_per_frame/2):]


        for i in range(int(points_per_frame/2)):
            # pos = 1 + i
            if i % 4 == 0:
                self.x_left_cam1.append(point_left[:, i])
                self.x_right_cam1.append(point_right[:, i])

            elif i % 4 == 1:
                self.y_left_cam1.append(point_left[:, i])
                self.y_right_cam1.append(point_right[:, i])

            elif i % 4 == 2:
                self.x_left_cam2.append(point_left[:, i])
                self.x_right_cam2.append(point_right[:, i])

            elif i % 4 == 3:
                self.y_left_cam2.append(point_left[:, i])
                self.y_right_cam2.append(point_right[:, i])


        # return point_list

    def plot_move(self, data):
        plt.plot(data)


    def sample_t(self):  # refer to file frame_index_pairs.txt for sampling rate
        index_path = os.path.join(self.file_path, 'frame_index_pairs.txt')
        with open(index_path, 'r') as file:
            points = file.read()
        file.close()
        points = points.split(' ')
        point_list = []
        idx = 0
        for i in points:
            try:
                point_list.append(int(i))
            except ValueError as e:
                print("error", e, "on line", idx)
            idx += 1
        point_list = np.array(point_list)
        point_list = point_list.reshape(int(len(points)/2), 2)

        sec = 1
        sample_t = []
        tmp = []
        for i in range(point_list.shape[0]):
            j = point_list[i, 1]
            if (sec-1)*fps < j <= sec * fps:
                tmp.append(j)
            else:
                sample_t.append(tmp)
                tmp = []
                tmp.append(j)
                sec += 1

        return sample_t


    def neigbor_ref(self):
        t = self.sample_t()



def read_txt(self):
    try:
        self.ref_file = pd.read_csv(self.ref_file)
        # print(self.ref_file.dtypes)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    session = 's02_Session_3'
    obj_joint = JointMove(session)
    # obj_joint.landmark_xy()
    obj_joint.sample_t()