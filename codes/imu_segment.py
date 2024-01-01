# --------------------------------------REVISIONS---------------------------------------
# Date        Name        Ver#    Description
# --------------------------------------------------------------------------------------
# 06/14/21    J. Li       [1]     Initial creation
# **************************************************************************************

import os
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pywt
from datetime import datetime
from coarse_segment import sliding_window_seg, plot_segments

package_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class VideoReference:

    def __init__(self, ref_file, ses):
        self.ref_file = ref_file
        self.start_loc = 0
        self.read_file()
        self.session = ses
        self.start_loc = np.where(self.ref_file['Session'] == self.session)[0][0]
        if max(self.ref_file['Session']) == self.session:
            self.end_loc = len(self.ref_file) - 1
        else:
            self.end_loc = np.where(self.ref_file['Session'] == self.session + 1)[0][0] - 2
        self.time_list = []
        self.labels = []

    def read_file(self):
        try:
            self.ref_file = pd.read_csv(self.ref_file)
            # print(self.ref_file.dtypes)
        except Exception as e:
            print(e)

    def _to_sec(self, h, m, s):
        secs = int(h) * 3600 + int(m) * 60 + int(s)
        return secs

    def ref_time(self):
        i = self.start_loc
        time_second = []
        while i <= self.end_loc:
            time = self.ref_file.iloc[i]['Time']
            self.time_list.append(time)
            i += 1
            # time = self.ref_file.iloc[i]['Time']

        for i in self.time_list:
            time_split = i.split(':')
            if len(time_split) == 3 and time_split[-1] == '00':
                time_second.append(self._to_sec(0, time_split[0], time_split[1]))
            else:
                print('Video Time is incorrect!')

        return time_second

    def ref_label(self):
        i = self.start_loc
        while i <= self.end_loc:
            labels = []
            j = 5
            while j < self.ref_file.shape[1]:
                # print(self.ref_file.shape[1])
                label = self.ref_file.iloc[i][j]
                # print(type(label))
                if type(label) == str:
                    label = self.ref_file.iloc[i][j]
                    tmp = label.split('\'')
                    labels.append(tmp[0])
                j += 1

            self.labels.append(labels)
            i += 1
        return self.labels


class IMUSegment:

    def __init__(self, data_file, seg_points=None):
        self.data_file = data_file
        self.data = None
        self.data_len = 0
        self.read_file()
        self.seg_axis = 'acc_x'
        self.adjusted_sample_rate = 179
        self.seg_points = seg_points
        self.seg_data = self.data[self.seg_axis]
        self.sntns_cut = []

    def read_file(self):
        try:
            self.data = pd.read_csv(self.data_file)
            self.data_len = len(self.data)
            # print(self.ref_file.dtypes)
        except Exception as e:
            print(e)

    def plot_data(self, data, title):
        plt.plot(data)
        plt.title('Selected Data')
        plt.title(title)
        plt.show()

    def sntns_segment(self, write=True):

        if not self.seg_points:
            print("There is no segment points provided by video for reference.")
            return

        seg_p = (np.array(self.seg_points) - self.seg_points[0]) * self.adjusted_sample_rate
        seg_p += int(self.seg_points[0]/2) + 3000
        # seg_p -= 0
        print('There are {} sentences in this IMU file'.format(str(len(seg_p))))
        if seg_p[-1] > self.data_len:
            print("The IMU data is incomplete!")
            seg_p = seg_p[self.data_len - seg_p > 0]


        seg_p = np.append(seg_p, self.data_len)

        modify_cut = input("Modify the start and end points? (y/n): ")

        rep_cut = []
        i = 0
        while i < len(seg_p) - 1:
        # for i in range(len(seg_p) - 1):
            # i = i + 15
            point_s = seg_p[i]
            point_e = seg_p[i + 1]
            self.plot_data(self.seg_data[point_s:point_e], str(i))
            print("Plot segment {}!".format(int(i)))
            if modify_cut == "y":
                cut = input("Cut? (y/n): ")  # yes/no
                if cut == "y":
                    point_s = int(input("Start point: "))
                    if point_s == 0:
                        point_s = seg_p[i]
                    point_e = int(input("End point: "))
                    if point_e == 0:
                        point_e = seg_p[i + 1]

                    self.plot_data(self.seg_data[point_s:point_e], 'cut' + str(i))


                seg_results = segment(self.seg_data[point_s:point_e], save_to_file=os.path.dirname(self.data_file))
                if not seg_results:
                    print("There is no segment detected!")
                    i += 1
                    continue
                print("There are {} segments detected!".format(len(seg_results)))
                seg_results = (np.array(seg_results) + point_s).tolist()

                # ## If there are false segment detected, run pick_seg function
                pick = input("Pick segments? (y/n): ")
                if pick == 'y':
                    seg_results = self.pick_seg(seg_results)
                    print("There are {} segments left!".format(len(seg_results)))

                goon = input("Continue? (y/n): ")
                if goon == 'n':
                    # i -= 1
                    continue



                # ## If there are less than 3 repetitions chosen, to make ensure the dimension of dataframe, append None
                while len(seg_results) < 3:
                    seg_results.append(None)

                self.sntns_cut.append([point_s, point_e])
                rep_cut.append(seg_results)
                # print(seg_results)
            i += 1


        if write:
            write_path = os.path.join(os.path.dirname(self.data_file), 'cut_points.csv')
            sntns_cut = pd.DataFrame(self.sntns_cut, columns=['sntns_start', 'sntns_end'])
            if rep_cut:
                try:
                    rep_cut = pd.DataFrame(rep_cut, columns=['s1', 's2', 's3'])
                    sntns_cut = pd.concat([sntns_cut, rep_cut], axis=1)
                except:
                    rep_path = os.path.join(os.path.dirname(self.data_file), 'rep_points.csv')
                    rep_cut.to_csv(rep_path, index=False)
            sntns_cut.to_csv(write_path, index=False)

        return self.sntns_cut

    def pick_seg(self, seg_results):
        cut_b = int(input("Number of false segments before: "))
        cut_a = int(input("Number of false segments after: "))
        cut_m = input("False segments in the middle: ")  # enter the false segment in the middle by '0', '1' or '2'

        # while enter == '':
        #     pick_point = int(input("Number of false segments before: "))
        if cut_a == 0:
            cut_a = len(seg_results)
        else:
            cut_a = -cut_a

        seg_results = seg_results[cut_b:cut_a]
        if cut_m == '':
            return seg_results
        cut_m = int(cut_m)
        if cut_m == 2:
            seg_results.pop(1)
            seg_results.pop(2)
        elif cut_m == 0 or cut_m == 1:
            seg_results.pop(int(cut_m + 1))

        return seg_results

    # the function is to segment each repetition of a sentence
    # expect 3 repetitions of each sentences

def save_process_log(w, w_s, a, b, file_path):
    # if not os.path.exists(data_file_name + '_log.txt'):
    save_log_path = os.path.join(file_path, 'log.txt')
    f = open(save_log_path, "a")
    f.write(str(datetime.now()))
    f.write('\n')
    f.write("window size for segmentation: %d\n" % w)
    f.write("window step for segmentation: %d\n" % w_s)
    f.write("threshold for segmentation: %d\n" % a)
    f.write("segment points are: \n")
    for i in b:
        f.write("%d %d\n" % (i[0], i[1]))
    f.close()

def segment(sntns_data, seg_window=50, seg_window_step=20, plot_axis=None, save_to_file=None):
    seg_points, threshold = sliding_window_seg(sntns_data, seg_window, seg_window_step)
    if plot_axis is None:
        plot_axis = 'acc_x'
    # time_axis = np.arange(0, len(sntns_data), 1)/self.adjusted_sample_rate
    # print(seg_points)
    if seg_points:
        plot_segments(sntns_data, np.array(seg_points), None, os.path.join(save_to_file, plot_axis))
    save_process_log(seg_window, seg_window_step, threshold, seg_points, save_to_file)
    return seg_points


def check_seg_resutls(dfile, nfile, pointfile, axis='acc_x'):
    try:
        file_path = os.path.dirname(dfile)
        dfile = pd.read_csv(dfile)
        nfile = pd.read_csv(nfile)
        pointfile = pd.read_csv(pointfile)
    except Exception as e:
        print(e)

    pfc = pointfile.copy()

    ddata = dfile[axis]
    ndata = nfile[axis]
    i =1
    seg_i = pointfile.loc[i]
    if type(seg_i['s1']) != str:
        temp = str([seg_i['sntns_start'], seg_i['sntns_end']-1])
        seg_point = []
        seg_point.append(temp)
        seg_point.append(temp)
    else:
        seg_point = [seg_i['s1'], seg_i['s2'], seg_i['s3']]
    seg_array = []
    # print(seg_point[:, 0])
    for elm in seg_point:
        if type(elm) != str:
            continue
        split_elm = elm.split(',')
        split_a = split_elm[0].split('[')
        split_b = split_elm[1].split(']')
        temp = [int(split_a[1]), int(split_b[0])]
        seg_array.append(temp)
    seg_array = np.array(seg_array) - seg_i['sntns_start']

    check_d = ddata[seg_i['sntns_start']: seg_i['sntns_end']]
    check_n = ndata[seg_i['sntns_start']: seg_i['sntns_end']]
    plot_segments(check_d, seg_array)
    plot_segments(check_n, seg_array)
    re_segment = input("Re-segment? (y/n): ")
    if re_segment == "n":
        return

    seg_results = segment(check_d, seg_window=50, seg_window_step=20, save_to_file=file_path)
    replace = input("Replace? (y/n): ")
    if replace == 'n':
        return
    seg_results = (np.array(seg_results) + seg_i['sntns_start']).tolist()
    print(seg_results)
    for j in range(3):
        seg_new = str(input("Replace seg in the plot?: "))

        pfc.loc[i, pfc.columns[2+j]] = seg_new


    writetofile = input("Write to file? (y/n): ")
    if writetofile:
        pfc.to_csv(os.path.join(file_path, 'cut_points_r.csv'), index=False)


def write_seg_file(pointfile):
    try:
        file_path = os.path.dirname(pointfile)
        pointfile = pd.read_csv(pointfile)
        point_len = len(pointfile)
    except Exception as e:
        print(e)

    seg_results = []

    for i in range(point_len):
        seg_i = pointfile.loc[i]
        seg_point = [seg_i['s1'], seg_i['s2'], seg_i['s3']]
        seg_array = []
        seg_array.append(seg_i['sntns_start'])
        seg_array.append(seg_i['sntns_end'])
        for elm in seg_point:
            if type(elm) != str:
                seg_array.append(0)
                seg_array.append(0)
                continue
            split_elm = elm.split(',')
            split_a = split_elm[0].split('[')
            split_b = split_elm[1].split(']')
            seg_array.append(int(split_a[1]))
            seg_array.append(int(split_b[0]))
        seg_results.append(seg_array)

    seg_results = pd.DataFrame(seg_results, columns=['sntns_start', 'sntns_end',
                                                     's1s', 's1e', 's2s', 's2e', 's3s', 's3e'])

    seg_results.to_csv(os.path.join(file_path, 'cut_points_f.csv'), index=False)


if __name__ == "__main__":
    subject = "Caroline"
    session = 6
    path_format = subject + "_data/"
    data_path_format = path_format + subject + " Session " + str(session) + "/IMU_data"
    data_file_path = os.path.join(package_directory, data_path_format)
    files = os.listdir(data_file_path)
    for i in files:
        t = os.path.join(data_file_path, i)
        if os.path.isdir(t):
            data_file = os.path.join(t, 'preprocessed.csv')
            if os.path.exists(data_file):
                char = i.split('_')
                if char[-1] == 'd':
                    data_file_d = data_file
                    cut_points = os.path.join(t, 'cut_points_r.csv')
                    if not os.path.exists(cut_points):
                        cut_points = os.path.join(t, 'cut_points.csv')
                elif char[-1] == 'n':
                    data_file_n = data_file
                else:
                    print("Wrong folder name!")
                    # exit()

    # path_format = subject + "_data/"
    # data_path_format = path_format + subject + " Session " + str(session) + "/IMU_data"
    # data_file_path = os.path.join(package_directory, data_path_format)
    # video_ref_path = path_format + subject + " video check.csv"
    # video_ref_path = os.path.join(package_directory, video_ref_path)
    #
    # data_file = "s01_20210411_130350_d"
    # data_file_d = os.path.join(data_file_path, data_file) + "/preprocessed.csv"
    # data_file_n = os.path.join(data_file_path, data_file[:-1]) + "n/" + "preprocessed.csv"


    # *****************************************************
    # video_ref = VideoReference(video_ref_path, session)
    # t_list = video_ref.ref_time()
    # imu_obj = IMUSegment(data_file_d, t_list)
    # imu_obj.sntns_segment(write=True)
    # *****************************************************


    # *****************************************************
    # seg_file = cut_points
    # check_seg_resutls(data_file_d, data_file_n, seg_file)
    # *****************************************************

    write_seg_file(cut_points)