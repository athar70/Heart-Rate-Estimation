# Heart Rate Estimation From Video via CNN

This repository contains the code and methodology for estimating heart rate from video feeds using a Convolutional Neural Network (CNN). The project explores non-contact video-based physiological measurements, focusing on heart rate estimation by leveraging transfer learning with the DenseNet-161 model and multiple regression.

## Project Overview

Traditional heart rate measurements typically involve physical sensors such as ECG or PPG, which can be inconvenient or intrusive. In this project, we propose a non-contact approach that estimates heart rate from facial videos. By using deep learning models and remote video processing, this project demonstrates how heart rate can be predicted from face images while participants watch video clips.

### Key Features

- **CNN-based heart rate estimation**: A CNN model with transfer learning from DenseNet-161 is used to predict heart rate from video frames.
- **ECG signal processing**: The project denoises ECG signals and calculates heart rate using a windowing approach.
- **Face recognition and normalization**: Faces are detected, cropped, and normalized from each video frame.
- **Dataset**: Uses the AMIGOS dataset, which includes ECG signals, facial videos, and participants' emotions.
- **Two evaluation approaches**: Train on participants and test on unseen participants, or train on a subset of videos and test on the remaining videos from the same participants.

## Methodology

The methodology consists of several key steps:

1. **ECG Signal Denoising**: ECG signals are filtered using a Notch filter to remove baseline wander. Heart rate is extracted using a sliding window approach.
2. **Face Recognition and Normalization**: Faces are detected and cropped from each video frame using the `face-recognition` package. The cropped face images are normalized to a size of 128x128 pixels.
3. **Convolutional Neural Network (CNN)**: A pre-trained DenseNet-161 model is used for transfer learning. The CNN outputs are passed to a multiple regression model with three hidden layers to predict heart rate.
4. **Loss Function and Optimizer**: The model is trained using Mean Squared Error (MSE) as the loss function and the Adam optimizer.
5. **Evaluation**: The model is evaluated using two approaches:
   - Train on 35 participants and test on 5 unseen participants.
   - Train on 75% of the videos of each participant and test on the remaining 25%.

## Results

The following results were obtained:
- **Test on Unseen Participants**: The average RMSE is 27.17 bpm, and the median is 26.41 bpm.
- **Test on Unseen Videos**: The average RMSE is 16.96 bpm, and the median is 12.32 bpm.

The results indicate that heart rate estimation is feasible, but performance can vary depending on the variability in participants’ characteristics, such as age and gender.

## Dataset

The AMIGOS dataset was used in this project. It contains video recordings of participants watching emotional videos, along with physiological signals such as ECG, EEG, and GSR. Each video is accompanied by annotated emotional states (arousal and valence). The dataset can be accessed [here](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/index.html).

### Video Clips

Participants watched a series of short video clips that elicited various emotional responses. The clips varied in arousal and valence, and heart rate was estimated while participants were watching the videos.

## How to Run the Code

1. **Clone the repository**:
   ```bash
   git clone https://github.com/athar70/Heart-Rate-Estimation.git
   cd Heart-Rate-Estimation
   ```
2. **Install the required libraries:**:

    ```bash
    pip install -r requirements.txt
    ```
3. **Download the dataset:**:
  The AMIGOS dataset is required for this project. Download it from [here](https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/) and place the data in the appropriate directory.

4. **Run the preprocessing and Train the model:**:

    ```bash
    pip install -r requirements.txt
    ```
## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Future Work

Possible future directions include:

- Reframing the heart rate estimation task as a classification problem (e.g., classifying heart rate as normal, slow, or fast).
- Exploring more sophisticated datasets that include annotations of emotional states.
- Investigating other model architectures, such as RNNs or attention mechanisms, to further improve the accuracy of heart rate estimation.

## References

1. X. Niu, S. Shan, H. Han, and X. Chen, “RhythmNet: End-to-end heart rate estimation from face via spatial-temporal representation,” *IEEE Transactions on Image Processing*, vol. 29, pp. 2409–2423, 2019.
2. W. Chen and D. McDuff, “DeepPhys: Video-based physiological measurement using convolutional attention networks,” in *Proceedings of ECCV*, 2018.
3. E. Lee, E. Chen, and C.-Y. Lee, “Meta-rPPG: Remote heart rate estimation using a transductive meta-learner,” in *ECCV*, 2020.
4. Z. Yu, W. Peng, X. Li, X. Hong, and G. Zhao, “Remote heart rate measurement from highly compressed facial videos,” *ICCV*, 2019.
