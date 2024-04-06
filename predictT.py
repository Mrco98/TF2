from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from recording_helper import record_audio
from librosa.core import resample

src_dir = '/home/mark/TF2/enhanced-sirenmodel/validation/T1'
sr = 16000
dt = 1
model_fn = '/home/mark/TF2/enhanced-sirenmodel/models/con2d.h5'
threshold = 20

def scan_siren(index):

    model = load_model(model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    classes = sorted(os.listdir(src_dir))
    #print('classes: ', classes)
    audio = record_audio(index)
    #print(audio)
    rate, wav = downsample_mono(audio, sr)
    mask, env = envelope(wav, rate, threshold=threshold)
    clean_wav = wav[mask]
    step = int(sr*dt)
    batch = []

    for i in range(0, clean_wav.shape[0], step):
        sample = clean_wav[i:i+step]
        sample = sample.reshape(-1, 1)

        if sample.shape[0] < step:
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)
            tmp[:clean_wav.shape[0],:] = clean_wav.flatten().reshape(-1, 1)
            sample = tmp
        batch.append(sample)
    X_batch = np.array(batch, dtype=np.float32)

    #print("X_batch SHAPE: ", X_batch.shape)
    y_pred = model.predict(X_batch, verbose=0)
    #print('model output: ', y_pred)
    y_mean = np.mean(y_pred, axis=0)
    #print('yMean: ', y_mean)
    y_pred = np.argmax(y_mean)
    #print('yPred: ', y_pred)
    #print('predictedClass: ', classes[y_pred])

    if classes[y_pred] == 'siren' and y_mean[y_pred] >= 0.8:
        detected = True
        #print(y_mean[y_pred].shape)
    else:
        detected = False
    return detected, y_mean[y_pred]
    #return detected









'''
        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch, verbose=0)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)
        y_predictions.append(y_pred)
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        y_truee.append(classes.index(real_class))
        print('File Name: '+wav_fn, 'Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
        results.append(y_mean)

    np.save(os.path.join('logs', args.pred_fn), np.array(results))
    return y_truee, y_predictions
'''    

 
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/conv2d.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='T1',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='D:/SirenNeuralNetwork/enhanced-sirenmodel/validation/T1',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    y_predictions = []
    y_truee = []
    make_prediction(args)

    #ACCURACY
    acc_score = accuracy_score(y_true=y_truee, y_pred=y_predictions)
    print(f'Accuracy: {acc_score:.3f}')

    #PRECISION
    precision = precision_score(y_true=y_truee, y_pred=y_predictions, average='micro')
    print(f"Precision: {precision:.3f}")

    #RECALL
    recall = recall_score(y_truee, y_predictions, average='micro')
    print(f"Recall: {recall:.3f}")

    #F1 SCORE
    f1 = f1_score(y_true=y_truee, y_pred=y_predictions, average='micro')
    print(f"F1 Score: {f1:.3f}")

    print(y_truee, len(y_truee))
    print(y_predictions, len(y_predictions))

    cm = confusion_matrix(y_truee, y_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Siren', 'Traffic'], yticklabels=['Siren', 'Traffic'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix - Gunshot')
    plt.show()

