import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

def filterAndVisualize(data, sampleRate):
    '''
    Filters ECG data using baseline wander removal and visualizes it.
    '''
    filtered = hp.remove_baseline_wander(data, sampleRate)
    plt.figure(figsize=(12, 3))
    plt.title('Filtered ECG Signal')
    plt.plot(filtered)
    plt.show()
    return filtered

def processEcgData(userData, numVideos=16, sampleRate=128):
    '''
    Processes ECG data for multiple videos, calculates heart rate, and saves results.
    '''
    videoFramePerSec = 25
    MaxHR = 200 #maximum heart rate that we assume: more than that is an error

    mat = loadmat(userData)
    data = mat['joined_data']
    videoIDs = mat['VideoIDs']

    minHR, maxHR, error, videoList = [], [], [], []

    for v in range(numVideos):
        videoID = videoIDs[0, v][0]
        ecgSignal = data[0, v][:, 15]  # ECG signal from channel 15
        filteredEcg = filterAndVisualize(ecgSignal, sampleRate)

        bpm, numError = [], 0
        windowSize = 300 * (sampleRate / videoFramePerSec)  # ~6-second window

        for i in range(1, len(filteredEcg), round(sampleRate / videoFramePerSec)):
            start = max(0, i - windowSize)
            end = min(len(filteredEcg), i + windowSize)
            segment = filteredEcg[int(start):int(end)]

            wd, m = hp.process(hp.scale_data(segment), sampleRate)
            if m['bpm'] < MaxHR:
                bpm.append(m['bpm'])
            else:
                bpm.append(np.nan)
                numError += 1

        minHR.append(np.nanmin(bpm))
        maxHR.append(np.nanmax(bpm))
        error.append(numError)
        videoList.append(videoID)

        print(f"Video {videoID} - Min HR: {minHR[-1]}, Max HR: {maxHR[-1]}, Errors: {numError}")

    return minHR, maxHR, error, videoList
