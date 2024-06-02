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


def distance_encoder(song1_indx, song2_indx, encoder, data):
    encoded_song1, encoded_song2 = encoder.predict(np.array([data[song1_indx], data[song2_indx]]), verbose=False)
    return np.linalg.norm(encoded_song1-encoded_song2)



