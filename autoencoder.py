import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import build
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


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


def creating_autoencoder(input_size):

    # Encoder
    encoder_input = Input((input_size,))
    encoder_nl = Dense(500, activation='relu')(encoder_input)
    encoder_nl2 = Dropout(0.2)(encoder_nl)
    encoder_nl3 = BatchNormalization()(encoder_nl2)
    encoder_encode = Dense(250, activation='relu')(encoder_nl3)
    encoder_encode2 = Dropout(0.2)(encoder_encode)
    encoder_encode3 = BatchNormalization()(encoder_encode2)
    encoder_encode4 = Dense(100, activation='relu')(encoder_encode3)
    encoder_encode5 = Dropout(0.2)(encoder_encode4)
    encoder_encode6 = BatchNormalization()(encoder_encode5)
    
    # Decoder
    decoder_nl = Dense(250, activation='relu')(encoder_encode2)
    decoder_nl2 = Dropout(0.2)(decoder_nl)
    decoder_nl3 = BatchNormalization()(decoder_nl2)
    decoder_nl4 = Dense(500, activation='relu')(decoder_nl3)
    decoder_nl5 = Dropout(0.2)(decoder_nl4)
    decoder_nl6 = BatchNormalization()(decoder_nl5)
    
    decoder_output = Dense(input_size, activation='sigmoid')(decoder_nl6)
    
    # Build the autoencoder model
    autoencoder = Model(encoder_input, decoder_output)
    autoencoder.compile(loss='MSE', optimizer=Adam(learning_rate=0.001))
    
    # Build the encoder model
    encoder = Model(encoder_input, encoder_encode6)

    
    return autoencoder, encoder


def distance_encoder(song1_indx, encoded_song, encoder, data):
    encoded_song1 = encoder.predict(np.array([data[song1_indx]]), verbose=False)
    #print(encoded_song1, encoded_song2)
    return np.linalg.norm(encoded_song1-encoded_song)



