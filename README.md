# asl_3d_handjoint_coordinates

The sequence of codes to be executed to run the full pipeline is:

    STEP_MINUS1_VIDEOS_PREPROCESS.py (ONLY FOR SELF-COLLECTED DATA)
    STEP0_VIDEO_SYNC.ipynb (ONLY FOR SELF-COLLECTED DATA)
    STEP1_MEDIAPIPE_AUTO.py
    STEP2_MATLAB_PREREQUISITES.py
    STEP3_TRIANGULATION.mlx
    STEP4_KEYPOINTS_TO_IMAGES.py
    STEP5_IMAGES_TO_VIDEO.py (OPTIONAL)

The directory structure that the codes use is as shown in directory_structure.png.
