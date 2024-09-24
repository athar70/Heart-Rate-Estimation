import cv2
import face_recognition
import numpy as np
import csv
import h5py

def cropFace(image):
    '''
    Detects and crops faces from the input image using CNN model.
    '''
    faceLocations = face_recognition.face_locations(image, model="cnn")
    for top, right, bottom, left in faceLocations:
        faceImage = image[top:bottom, left:right]
        n_px = 128
        faceImage = cv2.resize(faceImage, (n_px, n_px))
        faceImage = faceImage / 255  # Normalize pixel values
        faceImage = faceImage.ravel() #reshape the image - roll it up into a column vector
        return faceImage

def assignHRToFrames(user, videoIDs, hrData):
    '''
    Reads video frames and assigns heart rate to each face.
    '''
    listImages, listLabels = [], []

    for videoID in videoIDs:
        hr = hrData[videoID]
        cap = cv2.VideoCapture(f'./video/{user}_{videoID}_face.mov')

        frameNum = 0
        while True:
            ret, frame = cap.read()
            if frame is None or frameNum >= len(hr):
                break

            hrInFrame = hr[frameNum]
            if not np.isnan(hrInFrame):
                face = cropFace(frame)
                listImages.append(face)
                listLabels.append(hrInFrame)

            frameNum += 1

        cap.release()

    return listImages, listLabels

def writeDataToH5File(destFilePath, listImages, listLabels):
    '''
    Writes cropped face images and corresponding heart rate labels to an H5 file.
    '''
    with h5py.File(destFilePath, 'a') as f:
        f.create_dataset('input_data', data=np.array(listImages, dtype=np.float32))
        f.create_dataset('input_labels', data=np.array(listLabels, dtype=np.float32))
