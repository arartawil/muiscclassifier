import sys

import librosa
import math
import tensorflow as tf
import numpy as np






def save_mfcc(dataset_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
		print(dataset_path)
		SAMPLE_RATE = 22050
		TRACK_DURATION = 30  # measured in seconds
		SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

		samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
		num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)


		signal, sample_rate = librosa.load(dataset_path, sr=SAMPLE_RATE)
		# process all segments of audio file
		for d in range(num_segments):
			start = samples_per_segment * d
			finish = start + samples_per_segment
			mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,hop_length=hop_length)
			mfcc = mfcc.T

			if len(mfcc) == num_mfcc_vectors_per_segment:
				return (mfcc.tolist())
            
def predict(model, X):

    X = X[np.newaxis, ...]  # array shape (1, 130, 13, 1)
    X = X[...,np.newaxis]
    print(X.shape)
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    if predicted_index == 2 :
        return("موشحات")
    elif predicted_index == 3:
        return("قصائد")
    elif predicted_index ==  0:
        return("تخت شرفي")
    elif predicted_index== 1:
        return("موال")
    elif predicted_index== 4:
        return("راي")
    print(" Predicted label: {}".format( predicted_index))

def predict_E(model, X):

    X = X[np.newaxis, ...]  # array shape (1, 130, 13, 1)
    X = X[...,np.newaxis]
    print(X.shape)
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    if predicted_index == 0 :
        return("blues")
    if predicted_index == 1 :
        return("classical")
    if predicted_index == 2 :
        return("country")
    if predicted_index == 3 :
        return("disco")
    if predicted_index == 4 :
        return("hiphop")
    if predicted_index == 5 :
        return("jazz")
    if predicted_index == 6 :
        return("metal")
    if predicted_index == 7 :
        return("pop")
    if predicted_index == 8 :
        return("reggae")
    if predicted_index == 9 :
        return("rock")
    print(" Predicted label: {}".format( predicted_index))
    
def load_file(file_name,lang):
    X=np.array(save_mfcc(file_name, num_segments=10))
    model=tf.keras.models.load_model("firstapp/LSTM_CNN_Arabic.h5")
    print(lang)
    if(lang=="2"):
        model = tf.keras.models.load_model("firstapp/LSTM_CNN_Arabic.h5" )
        return predict(model, X)
    elif(lang=="1"):
        model = tf.keras.models.load_model("firstapp/LSTM_CNN_E.h5")
        return predict_E(model, X)




