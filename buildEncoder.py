import pandas as pd
import numpy as np
import alphabetical
import autoencoder as ae
import matplotlib.pyplot as plt
import build


def load_data(filename):
    """
    input: (string) file name
    output: (pandas dataframe) of csv data

    Builds pandas dataframe and normalizes loudness and tempo
    """
    data = pd.read_csv(filename)
    data['similar'] = data['similar'].apply(lambda x: x.split('+'))
    data['artist_terms'] = data['artist_terms'].apply(lambda x: str(x).split('+'))
    data['artist_terms_weights'] = data['artist_terms_weights'].apply(lambda x: str(x).split('+'))
    
    l_max = data['loudness'].max()
    l_min = data['loudness'].min()
    
    t_max = data['tempo'].max()
    t_min = data['tempo'].min()
    
    data['tempo'] = (data['tempo'] - t_min)/(t_max - t_min)
    data['loudness'] = (data['loudness'] - l_min)/(l_max - l_min)
    alphabetical.sort_dataframe(data)
    term_index = ae.get_term_index(data)

    return data, term_index


def build_encoder(df, term_index, tuning = False):
    """
    df: dataframe

    """
    data_train = ae.create_inputdata_ws(df, term_index)
    autoencoder, encoder = ae.creating_autoencoder(data_train.shape[1],128, 200)
    ae_history = autoencoder.fit(data_train, data_train, epochs=10)
    if tuning:
        plt.plot(ae_history.history['loss']);
        plt.xlabel('epochs')
        plt.ylabel('loss')
    return encoder


def build_distances(df: pd.core.frame.DataFrame, encoder, term_index):
    """
    input: dataFrame of songs
    output: list of lists

    Creates matrix of distances.
    """
    M = len(df)
    distances = [[0 for i in range(M)] for j in range(M)]
    for j in range(M):
        for i in range(j+1,M):
            distances[i][j] = ae.distance_encoder(df.iloc[j], df.iloc[i], encoder, term_index)
    return distances


def format_input(input: list, df: pd.core.frame.DataFrame):
    """
    input: list of song names, dataFrame of songs
    output: list of songs, list of indexes of songs in dataFrame
    """

    alphabetical.sort_dataframe(df)
    index_dict = alphabetical.get_alpha_dict(df)
    
    indexes = [0]*len(input)
    result = []
    for i in range(len(input)):
        index = alphabetical.get_index_song(input[i][0], input[i][1],input[i][2],df,index_dict)
        result.append(df.iloc[index])
        indexes[i]= index
            
    return result, indexes


def select_distance_to_playlist(playlist: list, indexes: list, distances: pd.core.frame.DataFrame = 0):
    """
    input: list of songs, list of indexes, distance matrix
    output: distance to playlist matrix, total distance vector
    """
    result = distances[indexes].copy()
    for i in indexes:
        result.drop(i, inplace=True)
    total = []
    for i in range(len(result)):
        total.append(sum(result.iloc[i]))
    return result, total


def compute_distance_to_playlist(playlist: list, indexes: list, df: pd.core.frame.DataFrame, encoder, term_index):
    """
    input: list of songs, list of indexes, distance matrix
    output: distance to playlist matrix, total distance vector
    """

    M = len(df)
    N = len(indexes)
    distances = [[0]*N]*M
    print(N,M)
    for j in range(M):
        if j%100 == 0:
            print("M: ", j)
        for i in range(N):
            distances[j][i] = ae.distance_encoder(df.iloc[j], df.iloc[i], encoder, term_index)
    total = []
    distances = pd.DataFrame(distances)
    for i in indexes:
        distances.drop(i, inplace=True)
    for i in range(len(distances)):
        total.append(sum(distances.iloc[i]))
    return distances, total


def get_k_recommendations(total_d, k, df):
    """
    total_d: array of total distances
    k: number of recommendations to give
    df: song dataframe
    """
    results = np.argpartition(total_d, k)[:k]
    output = []
    d = []
    for i in results:
        output.append(df.iloc[i])
        d.append(total_d[i])
    output_df = pd.DataFrame(output)
    output_df['total distance'] = d
    return output_df[['artist_name', 'title', 'release','total distance']]