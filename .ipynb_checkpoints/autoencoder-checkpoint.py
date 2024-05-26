import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import build
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dropout
from tensorflow.keras import Model


def get_term_index(df):
    unique_terms = set(term for sublist in df['artist_terms'] for term in sublist)
    term_index = {term: idx for idx, term in enumerate(unique_terms)}
    return term_index

def create_inputdata_ws(df, term_index):
    udf = df[['hottness', 'loudness', 'tempo']]
    aadf = df[['artist_terms', 'artist_terms_weights']]
    
    # Process numerical features
    data_train_u = np.array(udf.values)

    # Process artist terms
    

    data_train_aa = np.zeros((len(df), len(term_index)))
    
    for i, (terms, weights) in enumerate(zip(df['artist_terms'], df['artist_terms_weights'])):
        if 'nan' not in terms:
            for term, weight in zip(terms, weights):
                    weight_float = float(weight)
                    idx = term_index[term]
                    data_train_aa[i, idx] = weight_float
            
    data_train = np.concatenate((data_train_u, data_train_aa), axis=1)

    return data_train


def creating_autoencoder(input_size, code_size, node_size):

    # Encoder
    encoder_input = Input((input_size,))
    encoder_nl = Dense(node_size, activation='relu')(encoder_input)
    encoder_encode = Dense(code_size, activation='relu')(encoder_nl)

    # Decoder
    decoder_nl = Dense(node_size, activation='relu')(encoder_encode)
    decoder_output = Dense(input_size)(decoder_nl)

    # Build the autoencoder model
    autoencoder = Model(encoder_input, decoder_output)
    autoencoder.compile(loss='MSE', optimizer=tf.optimizers.Adam(learning_rate=0.001))

    # Build the encoder model
    encoder = Model(encoder_input, encoder_encode)

    
    return autoencoder, encoder


def distance_encoder(song1, song2, encoder, term_index):
    """
    songs must be of the form ['artist_name', 'title', 'release', 'similar', 'hottness', 'artist_terms', 'artist_terms_weights', 'loudness', 'tempo']
    
    term_index has to be done on the full dataset 
    """
    
    hsong1, hsong2 = np.array([song1[4], song1[7], song1[8]]), np.array([song2[4], song2[7], song2[8]])

    atw1, atw2 = np.zeros(len(term_index)), np.zeros(len(term_index))
    
    if not any(pd.isna(song1[5])):  # Assuming song1[5] and song1[6] are lists of terms and weights
        for term, weight in zip(song1[5], song1[6]):
            weight_float = float(weight)
            idx = term_index[term]
            atw1[idx] = weight_float

    if not any(pd.isna(song2[5])):  # Assuming song2[5] and song2[6] are lists of terms and weights
        for term, weight in zip(song2[5], song2[6]):
            weight_float = float(weight)
            idx = term_index[term]
            atw2[idx] = weight_float
        
    hsong1 = np.concatenate([hsong1, atw1])
    hsong2 = np.concatenate([hsong2, atw2]) 
    
    encoded_song1, encoded_song2 = encoder.predict(np.array([hsong1, hsong2]), verbose=False)
    
    return np.linalg.norm(encoded_song1-encoded_song2)



