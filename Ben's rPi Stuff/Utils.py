import cv2
import numpy as np


def getConfig(path):
    configFile = open(path)
    configText = configFile.read()
    configFile.close()
    config = eval(configText)
    return config


def getAngle(homography):
    print('homography: ' + str(homography))
    print('0th column: ' + str(homography[:, 0]))
    print('1st column: ' + str(homography[:, 1]))
    pose = np.eye(3, 4)  # 3x4 matrix, possibly not right size
    norm1 = np.linalg.norm(homography[:, 0])
    norm2 = np.linalg.norm(homography[:, 1])
    tnorm = (norm1+norm2) / 2

    cv2.normalize(homography[:, 0], pose[:, 0])
    cv2.normalize(homography[:, 1], pose[:, 1])

    print(pose[:, 0])
    v3 = np.cross(pose[:, 0], pose[:, 1])
    np.copyto(pose[2, 2], v3)
    pose[3, 3] = homography[2, 2] / tnorm
    print(pose[3, 3])

    return pose
