# --------------------------------------REVISIONS---------------------------------------
# Date        Name        Ver#    Description
# --------------------------------------------------------------------------------------
# 08/06/21    J. Li       [1]     Initial creation
# **************************************************************************************

"""
Input: 'cut_points_f.csv' for each session
Output: 'cut_frame.csv' for each session
"""
import cv2
import os
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pywt
from datetime import datetime
import time
from imu_segment import VideoReference
from moviepy.editor import *


package_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fps = 30
SMP_RATE = 178
# video_sync_s01 = [3.15, 2.22, 2.52, 3.08, 4.02, 2.42]  # The start ms of front_view wrt side_view
video_sync_s01 = [[3, 8], [2, 22], [2, 52], [3, 8], [4, 2], [2, 42]]
# video_sync_s01 = [3.25, 2.367, 2.867, 3.133, 4.033, 2.867]
# video_sync_s01 = [3.26666667, 2.43333333, 2.93333333, 3.16666667, 4.1, 2.76666667]
# video_sync_s02 = [-4.02, -3.01, 10.03, 5.47, 3.09, 2.17, 3.07]
# video_sync_s02 = [-4.033, -3.017, 10.133, 5.783, 3.15, 2.283, 3.117]
# video_sync_s02 = [[-4, -2], [-3, -1], [10, 3], [5, 47], [3, 9], [2, 17], [3, 7]]
video_sync_s02 = [[-3, -28], [-3, 0], [10, 8], [5, 47], [3, 9], [2, 12], [3, 5]]
# video_sync_s01_2 = [0.1167, 0.21333333, 0.41333333, 0.08666667, 0.08, 0.34666667]

def imu_seg_ref(video_ref_time, sync_ref, reffile):
    try:
        cut_points = pd.read_csv(reffile)
        # print(cut_points.iloc[0, :])
    except Exception as e:
        print(e)
        
    video_ref_time = np.array(video_ref_time) #- sync_ref

    len_cut = cut_points.shape[0]
    cut_sec = []
    for i in range(len_cut):
        cut_row = []
        sync_start = video_ref_time[i]
        seg_i = list(cut_points.iloc[i, :])
        tmp = []
        for ele in seg_i:
            if type(ele) == int:
                tmp.append(ele)
        seg_i = np.array(tmp)/SMP_RATE  # seconds
        # cut_row.append(seg_i[-1] - seg_i[2] + 1 + seg_i[0])
        rept_frame = np.array(seg_i[2:]) - seg_i[2] + sync_start
        for ele in rept_frame.tolist():
            cut_row.append(ele)
        cut_sec.append(cut_row)

    cut_sec = pd.DataFrame(cut_sec, columns=['s1s', 's1e', 's2s', 's2e', 's3s', 's3e'])

    return cut_sec


# def clip_video(seg_ref, video_path):
#     try:
#         cut_frames = pd.read_csv(seg_ref)
#         # print(cut_points.iloc[0, :])
#     except Exception as e:
#         print(e)
#
#     cut = list(cut_frames.loc[0])
#     #     for k in range(3):
#     init_frame = cut[2]
#     end_frame = cut[3] + 30
#
#     show_video = PlayVideo(video_path, [init_frame, end_frame], write_video=True)
#     show_video.read_frame()


def fps_reslotion_adjustment(seg_ref, sync_frame, video_path, view, crop = False, res = False):
    fps = 30
    name = video_path.split('/')[-1]
    s_name = video_path.split('/')[-2]
    output_name = s_name.split('_')[0]
    output_name_ori = output_name
    name = name.split('.')[0]
    name = name.split('_')
    # output_name = 's01'
    for i in name[1:]:
        output_name = output_name + '_' + i

    if not res:
        clip = VideoFileClip(video_path)
        print(clip.fps)
        clip = VideoFileClip(video_path).set_fps(fps)
        output_path = os.path.join(os.path.dirname(video_path), output_name + '_fps.mp4')

        clip.write_videofile(output_path, fps=fps)

    if res:
        if type(seg_ref) == list:
            seg_points = seg_ref
        else:
            seg_points = cut_video_points(seg_ref)
        # seg_points = seg_points[0:19]   # for s02_Session_3 [0:19] s02_Session_4 [0:6] for  (missing front view data)
        vidFile = cv2.VideoCapture(video_path)
        fps = vidFile.get(cv2.CAP_PROP_FPS)
        print(fps)
        sync_frame = sync_frame[0] * fps + sync_frame[1]*0.5
        output_path = os.path.join(os.path.dirname(video_path), output_name + '_sync.mp4')
        if view == 'front' and crop:
            scale = 0.5625
            x = 250
            y = int(x * scale)
            x_r = int(vidFile.get(cv2.CAP_PROP_FRAME_WIDTH) - x + 1)
            y_l = int(vidFile.get(cv2.CAP_PROP_FRAME_HEIGHT) - y + 1)
            h = 720
            w = 1280

            # # x, y, h, w = 300, 240, 720, 1280
            # x, y, h, w = 0, 0, 720, 1280
        else:
            if output_name_ori == 's01':
                x, y, h, w = 300, 100, 620, 880   # for Caroline side view
            else:
                x, y, h, w = 0, 0, 720, 1280

                # a_h = 720
                # a_w = 1280
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Now we start
        i = 0
        # prev = 0
        frame_pair_idx = 0
        while frame_pair_idx < len(seg_points):
            print(frame_pair_idx)
            # init_frame = int(seg_points[frame_pair_idx][0] * fps + 0.5)
            # end_frame = int(seg_points[frame_pair_idx][1] * fps + 0.5)
            init_frame = int(seg_points[frame_pair_idx][0] * fps + 0.5 - sync_frame)
            end_frame = int(seg_points[frame_pair_idx][1] * fps + 0.5 - sync_frame)
            while i <= end_frame:
                ret, frame = vidFile.read()

                if i >= init_frame:
                    # time_elapsed = time.time() - prev
                    # # print(time_elapsed)
                    #
                    # if time_elapsed > 1. / 30:
                    #     prev = time.time()
                    if crop:
                        crop_frame = frame[y:y_l, x:x_r]
                    else:
                        crop_frame = frame
                    if output_name_ori == 's01':  # and view == 'side':
                        channel0 = cv2.resize(crop_frame[:, :, 0], (w, h))
                        channel1 = cv2.resize(crop_frame[:, :, 1], (w, h))
                        channel2 = cv2.resize(crop_frame[:, :, 2], (w, h))

                        channel0 = np.reshape(channel0, (h, w, 1))
                        channel1 = np.reshape(channel1, (h, w, 1))
                        channel2 = np.reshape(channel2, (h, w, 1))

                        crop_frame = np.concatenate((channel0, channel1, channel2), axis=2)
                    # else:
                    #     crop_frame = frame[y:y + h, x:x + w]
                    cv2.imshow('Frame', crop_frame)
                    cv2.waitKey(1)
                    out.write(crop_frame)
                i += 1
            frame_pair_idx += 1

        # Store this frame to an image

        # cv2.imwrite(video_name + '_frame_' + str(frame_count) + '.jpg')
        out.release()
        vidFile.release()
        cv2.destroyAllWindows()


def cut_video_points(ref_df):
    # ref_df = pd.read_csv(ref_df)
    frame_pair = []
    for row in range(len(ref_df)):
        cut = list(ref_df.loc[row])
        for k in range(3):
            init_frame = cut[k*2]
            end_frame = cut[k*2 + 1] + 1
            frame_pair.append([init_frame, end_frame])
        # pair_tmp = [cut[0], cut[-1] + 1]
        # frame_pair.append(pair_tmp)
    return frame_pair  # in seconds

def cut_video_noref(ref_df, sync):
    # ref_df = pd.read_csv(ref_df)
    frame_pair = []
    for row in range(len(ref_df)):
        cut = list(ref_df.loc[row])
        pair_tmp = [cut[1], cut[2]]
        time_second = []
        for cut in pair_tmp:
            time_split = cut.split(':')
            if len(time_split) == 3 and time_split[-1] == '00':
                time_second.append(_to_sec(0, time_split[0], time_split[1])) # - sync
            else:
                print('Video Time is incorrect!')
        frame_pair.append(time_second)
    return frame_pair


# def concat_video(seg_ref, video_path):
#
#     vidFile = cv2.VideoCapture(video_path)
#     if not vidFile.isOpened():
#         print("Error opening video file")
#         exit()
#
#     h = int(vidFile.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     w = int(vidFile.get(cv2.CAP_PROP_FRAME_WIDTH))
#     fps = vidFile.get(cv2.CAP_PROP_FPS)
#     print(fps)
#     print(w, h)
#
#     name = video_path.split('/')[-1]
#     name = name.split('.')[0]
#     name = name.split('_')
#     output_name = 's01'
#     for i in name[1:]:
#         output_name = output_name + '_' + i
#     output_path = os.path.join(os.path.dirname(video_path), output_name + '_concat2.mp4')
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
#
#     frame_pair = cut_points(seg_ref)
#
#
#     i = 0
#     # prev = 0
#     frame_pair_idx = 0
#     while frame_pair_idx < len(frame_pair):
#         print(frame_pair_idx)
#         init_frame = frame_pair[frame_pair_idx][0]
#         end_frame = frame_pair[frame_pair_idx][1]
#         while i <= end_frame:
#             ret, frame = vidFile.read()
#             if i >= init_frame:
#                 # time_elapsed = time.time() - prev
#                 # # print(time_elapsed)
#                 #
#                 # if time_elapsed > 1. / 30:
#                 #     prev = time.time()
#
#                     # cv2.imshow('Frame', frame)
#                     # cv2.waitKey(1)
#                 out.write(frame)
#             i += 1
#         frame_pair_idx += 1
#
#     # Store this frame to an image
#
#         # cv2.imwrite(video_name + '_frame_' + str(frame_count) + '.jpg')
#
#     vidFile.release()
#     cv2.destroyAllWindows()


# def concate_2(seg_point, video_path):
#     seg_point = int(seg_point * 30)
#
#     vidFile = cv2.VideoCapture(video_path)
#     end_frame = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT))
#     h = int(vidFile.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     w = int(vidFile.get(cv2.CAP_PROP_FRAME_WIDTH))
#     fps = vidFile.get(cv2.CAP_PROP_FPS)
#
#     name = video_path.split('/')[-1]
#     name = name.split('.')[0]
#     output_path = os.path.join(os.path.dirname(video_path), name + '_4.mp4')
#
#     # writer = skvideo.io.FFmpegWriter(output_path, outputdict={
#     #     '-vcodec': 'libx264',  # use the h.264 codec
#     #     '-crf': '0',  # set the constant rate factor to 0, which is lossless
#     #     '-preset': 'veryslow'  # the slower the better compression, in princple, try
#     #     # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
#     # })
#
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     # fourcc = cv2.VideoWriter_fourcc(*'AVC1')
#     # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
#     out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
#
#     i = 0
#     while i < end_frame+1:
#         ret, frame = vidFile.read()
#         # print(i)
#
#         if i >= seg_point:
#             # cv2.imshow('Frame', frame)
#             # cv2.waitKey(1)
#             out.write(frame)
#             # writer.writeFrame(frame[:, :, ::-1])
#         i += 1
#
#     # writer.close()  # close the writer
#     vidFile.release()
#     cv2.destroyAllWindows()


def _to_sec(h, m, s):
        secs = int(h) * 3600 + int(m) * 60 + int(s)
        return secs

def video_check(video):
    vidFile = cv2.VideoCapture(video)
    if not vidFile.isOpened():
        print("Error opening video file")
        exit()

    fps = vidFile.get(cv2.CAP_PROP_FPS)
    frame_count = vidFile.get(cv2.CAP_PROP_FRAME_COUNT)
    print(fps)
    print(frame_count)
    return fps, frame_count


if __name__ == "__main__":
    subject = "Caroline"
    session = 4
    exp = False
    if subject == "Caroline":
        sbj = 's01'
        sync_time = video_sync_s01[session - 1]
    else:
        sbj = 's02'
        sync_time = video_sync_s02[session - 1]
    if sbj == 's01':
        exp = True

    path_format = subject + "_data/"
    video_ref_path = path_format + subject + " video check.csv"
    video_ref_path = os.path.join(package_directory, video_ref_path)
    video_ref = VideoReference(video_ref_path, session)
    if exp:
        if session != 1 and session != 5:
            t_list = video_ref.ref_time()
            data_path_format = path_format + subject + " Session " + str(session) + "/IMU_data"
            data_file_path = os.path.join(package_directory, data_path_format)
            files = os.listdir(data_file_path)
            for i in files:
                char = i.split('_')
                if char[-1] == 'd':
                    t = os.path.join(data_file_path, i)
                    cut_points = os.path.join(t, 'cut_points_f.csv')

    else:
        t_list = video_ref.ref_time()
        data_path_format = path_format + subject + " Session " + str(session) + "/IMU_data"
        data_file_path = os.path.join(package_directory, data_path_format)
        files = os.listdir(data_file_path)
        for i in files:
            char = i.split('_')
            if char[-1] == 'd':
                t = os.path.join(data_file_path, i)
                cut_points = os.path.join(t, 'cut_points_f.csv')


    folder_name = sbj + '_Session_' + str(session)
    ori_root = '/Users/jiyangli/Documents/UB/research/SmartRing/datasets/video_asl/inputs/' + folder_name

    # **************************************************************************************
    ori_video = ['s01_front_s4_20210410_fps.mp4'] #, 'Hanna_side_s7_20210411.mp4']
    # **************************************************************************************

    for vv in ori_video:
        ori_file = os.path.join(ori_root, vv)
        which_view = vv.split('_')
        which_view = which_view[1]
        if which_view == 'front':
            sync_time = [0, 0]
        if exp:
            if session == 1 or session == 5:
                ref_file = os.path.join(path_format + subject + " Session " + str(session), folder_name + ' video check.csv')
                ref_file = pd.read_csv(os.path.join(package_directory, ref_file))
                ref_seconds = cut_video_noref(ref_file, sync_time)
            else:
                ref_seconds = imu_seg_ref(t_list, sync_time, cut_points)
        else:
            ref_seconds = imu_seg_ref(t_list, sync_time, cut_points)
        #
        #     # seg_ref.to_csv(os.path.join(t, which_view + '_cut_frame.csv'), index=False)
        #
        # # ref_seconds = os.path.join(t, which_view + '_cut_frame.csv')
        fps_reslotion_adjustment(ref_seconds, sync_time, ori_file, which_view, crop=True, res=True)
        # fps_reslotion_adjustment(ref_seconds, sync_time, ori_file, which_view)
        # video_check(ori_file)