#--------------------------------------REVISIONS---------------------------------------
# Date        Name        Ver#    Description
#--------------------------------------------------------------------------------------
# 11/20/20    J. Li       [1]     Initial creation
#**************************************************************************************

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pywt

package_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_file_path = os.path.join(package_directory, "IMU_data")
data_file = data_file_path + "/sean_data/" + "s01_20210202_173043_d.csv"
fig_save_path = os.path.join(package_directory, "figures")


# df = pd.read_csv("/Users/jiyangli/Downloads/acc_walking_csv/acc_walking_waist.csv")


def plot_2d(data, columns, start=None, length=None, len_wid_ratio = 1.7, save_to_file = 0):

    title_dic = dict()
    if start is None:
        start = 0

    if length is None:
        length = len(data)

    times = data['ts_sensor']
    # for i in range(1, len(times)):
    #     if times[i] <= 0:
    #         times[i] = times[i] + 2 ** 16
    times = (times - times[0]) / 1000

    fig, axs = plt.subplots(3, 1, figsize=(12, 12/len_wid_ratio))
    axs[0].set(title='accelerometer')
    # plt.title('Gyroscope')
    # plt.title('Magnetometer')
    axs[0].plot(times[start:length] - times[0], data[columns[0]][start:length])
    axs[1].plot(times[start:length] - times[0], data[columns[1]][start:length])
    axs[2].plot(times[start:length] - times[0], data[columns[2]][start:length])

    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    # plt.subplot(3,1,1)
    # plt.plot(df['acc_x'])
    # plt.subplot(3,2,1)
    # plt.plot(df['acc_y'])
    # plt.subplot(3,3,1)
    # plt.plot(df['acc_z'])
    plt.show()
    if save_to_file:
        fig.savefig(fig_save_path + '/' + 'acc_m.png')
    # return

# def stft_2d():


def sliding_window_seg(data_s, windowSize, windowstep):  # 200Hz data
    data = np.array(data_s)
    w_range = int((len(data) - windowSize + windowstep)/windowstep)
    window_var = []
    a = 0
    b = a + windowSize - 1
    i = 0
    while i < w_range:
        w_data = data[a:b]
        window_var.append(np.var(w_data))
        a = a + windowstep
        b = b + windowstep
        i = i + 1

    window_var = np.array(window_var)
    threshold = 420  # set larger threshold for a more fined segmentation
    var_b = window_var < threshold
    var_b = var_b * 1

    # check whether there are fake pauses, which has pause time < 3 windows (~300ms)
    pauses = []
    pauses_count = 0
    for i in range(len(var_b)):
        if var_b[i] == 1:
            pauses.append(i)
        else:
            if 0 < len(pauses) <= 3:
                var_b[pauses] = 0
                pauses_count += 1
            pauses = []

    print('Detected and removed {} fake pauses'.format(pauses_count))
    segment = {}  # segment data where number of (var_b == 0) > 500ms
    seg_point = []
    count = 0
    for i in range(len(var_b)):
        # print(var_b[i])
        if var_b[i] == 0:
            seg_point.append(i)
        else:
            if seg_point:
                count += 1
                segment[count] = seg_point
                seg_point = []

    if var_b[-1] == 0:
        count += 1
        segment[count] = seg_point


    start_end = []
    for i in range(len(segment)):
        seg = np.array(segment[i+1])
        if len(seg) > 5:  #around 300ms
            start = seg[0] * windowstep
            end = seg[-1] * windowstep + windowSize - 1
            start_end.append([start, end])

    # start_end = np.array(start_end)

    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(window_var)
    #
    # for i in range(len(window_var)):
    #     if var_b[i] > 0:
    #         ax.scatter(i, window_var[i])
    #
    # # ax.plot(window_var * var_b)
    # plt.ylabel('Variance')
    # plt.ylim(-1000, 20000)
    # fig.savefig(fig_save_path + '/' + 'acc_x.png')
    #
    # plt.show()

    return start_end, threshold


def plot_segments(data, seg_points, axis_x=None, savefig=None):
    if axis_x is None:
        axis_x = np.arange(0, len(data))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(axis_x, data)
    ax.vlines(axis_x[seg_points[:, 0]], min(data), max(data), 'r', 'dashed', label='Start')
    ax.vlines(axis_x[seg_points[:, 1]], min(data), max(data), 'g', 'dashed', label='End')
    plt.legend()
    plt.show()
    if savefig:
        fig.savefig(savefig + '_seg' + '.png')


def dwt_seg(data_array):
    (cA, cD) = pywt.dwt(data_array, 'db1')
    plt.plot(data_array)
    plt.show()
    # plt.plot(cA)
    plt.plot(cD)
    plt.legend(['Approximation', 'Coefficients'])
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv(data_file)
    data_length = len(df)
    # plot_2d(df, ['acc_x', 'acc_y', 'acc_z'])

    # sp = sliding_window_seg(df['acc_x'][1500:data_length-700], 30, 15)  #
    # plot_segments(df['acc_y'], sp)
    # dwt_seg(df['acc_x'])
