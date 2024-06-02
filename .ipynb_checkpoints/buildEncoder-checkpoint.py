import pandas as pd
import numpy as np
import alphabetical
import autoencoder as ae
import matplotlib.pyplot as plt
import build
from keras.callbacks import EarlyStopping


def reduce_terms(df):
    terms = ['00','40','50','60','70','80','90',
    'hip hop', 'house', 'jazz','acid','blues','acoustic','rock','metal','techno','punk','rap','ambient','alternative','pop','bachata','ballet',
'heavy','funk','chill','folk','reggae','indie','soul','country','latin','japan','dance','disco','german','classic','french','greek','brazil','irish','viking','turk','finish','celtic','mambo','rumba','merengue','karaoke','swing','norway','arab','chinese','japan','canad','scandi','salsa',
    'beach','contemporary', 'gospel', 'psych', 'melod']
    
    for i in range(len(df)):
        t = []
        w = []
        for pterm in terms:
            avg = 0
            n = 0
            for sterm in df.iloc[i]['artist_terms']:
                if pterm in sterm or sterm in pterm:
                    n+=1
                    avg += float(df.iloc[i]['artist_terms_weights'][df.iloc[i]['artist_terms'].index(sterm)])
    
                    if pterm not in t:
                            t.append(pterm)
            if n != 0:
                w.append(avg/n)
    
        df.at[i, 'artist_terms'] = t
        df.at[i, 'artist_terms_weights'] = w
    
def build_batches(playlist: list, indexes: list, df: pd.core.frame.DataFrame, encoder, filename):
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
            distances[j][i] = ae.distance_encoder(j, i, encoder, df)
    if filename:
        df = pd.DataFrame(distances)
        df.to_csv(filename, index=False)
    return



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
    reduce_terms(data)

    return data, term_index


def build_encoder(df, term_index, tuning = False):
    """
    df: dataframe

    """
    data_train = ae.create_inputdata_ws(df, term_index)
    autoencoder, encoder = ae.creating_autoencoder(data_train.shape[1])
    earlyStop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    ae_history = autoencoder.fit(data_train, data_train, epochs=10, shuffle=True, batch_size=32, validation_split=0.2, callbacks=[earlyStop])
    if tuning:
        plt.plot(ae_history.history['loss']);
        plt.xlabel('epochs')
        plt.ylabel('loss')
    return data_train, encoder


def build_distances(df: pd.core.frame.DataFrame, encoder, filename = None):
    """
    input: dataFrame of songs
    output: list of lists = None

    Creates matrix of distances.
    """
    M = len(df)
    distances = [[0 for i in range(M)] for j in range(M)]
    for j in range(M):
        if (j%1000 == 0): print("M: ", j)
        for i in range(j+1,M):
            distances[i][j] = ae.distance_encoder(j, i, encoder, df)
    if filename:
        df.to_csv(filename, index=False)
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


def compute_distance_to_playlist(playlist: list, indexes: list, df: pd.core.frame.DataFrame, encoder):
    """
    input: list of songs, list of indexes, distance matrix
    output: distance to playlist matrix, total distance vector
    """
    M = len(df)
    N = len(indexes)
    distances = np.zeros((M, N))  # Correct initialization of distances matrix

    playlist_encoding = encoder.predict(np.array([df[i] for i in indexes]), verbose=False)

    for j in range(M):
        if j % 100 == 0:
            print("M: ", j)
        encoded_song2 = encoder.predict(np.array([df[j]]), verbose=False)

        for i in range(N):
            distances[j][i] = np.linalg.norm(playlist_encoding[i] - encoded_song2[0])

    total = distances.sum(axis=1)
    distances_df = pd.DataFrame(distances)

    for i in indexes:
        distances_df.drop(i, inplace=True)

    total = list(distances_df.sum(axis=1))
    return distances_df, total


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