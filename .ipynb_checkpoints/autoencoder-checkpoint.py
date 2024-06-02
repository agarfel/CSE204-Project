import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import build
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dropout
from tensorflow.keras import Model


def get_term_index(df):
    unique_terms = set(['00','40','50','60','70','80','90',
    'hip hop', 'house', 'jazz','acid','blues','acoustic','rock','metal','techno','punk','rap','ambient','alternative','pop','bachata','ballet',
'heavy','funk','chill','folk','reggae','indie','soul','country','latin','japan','dance','disco','german','classic','french','greek','brazil','irish','viking','turk','finish','celtic','mambo','rumba','merengue','karaoke','swing','norway','arab','chinese','japan','canad','scandi','salsa',
    'beach','contemporary', 'gospel', 'psych', 'melod'])
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
    encoder_nl = Dense(50, activation='relu')(encoder_input)
    encoder_encode = Dense(25, activation='relu')(encoder_nl)
    encoder_encode2 = Dense(10, activation='relu')(encoder_encode)

    # Decoder
    decoder_nl = Dense(25, activation='relu')(encoder_encode2)
    decoder_nl2 = Dense(50, activation='relu')(decoder_nl)

    decoder_output = Dense(input_size)(decoder_nl2)


    # Build the autoencoder model
    autoencoder = Model(encoder_input, decoder_output)
    autoencoder.compile(loss='MSE', optimizer=tf.optimizers.Adam(learning_rate=0.001))

    # Build the encoder model
    encoder = Model(encoder_input, encoder_encode2)

    
    return autoencoder, encoder


def distance_encoder(song1_indx, encoded_song, encoder, data):
    encoded_song1 = encoder.predict(np.array([data[song1_indx]]), verbose=False)
    #print(encoded_song1, encoded_song2)
    return np.linalg.norm(encoded_song1-encoded_song)



