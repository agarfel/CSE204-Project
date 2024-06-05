import pandas as pd
import numpy as np
import alphabetical

def reduce_terms(df):
    terms = ['00','40','50','60','70','80','90',
    'hip hop', 'house', 'jazz','acid','blues','acoustic','rock','metal','techno','punk','rap','ambient','alternative','pop','bachata','ballet',
    'heavy','funk','chill','folk','reggae','indie','soul','country','latin','japan','dance','disco','german','classic','french','greek','brazil',
    'irish','viking','turk','finish','celtic','mambo','rumba','merengue','karaoke','swing','norway','arab','chinese','japan','canad','scandi','salsa',
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

def load_data(filename, reduce = False, filterz = False):
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
    if reduce:
        reduce_terms(data)
    if filterz:
        df_filtered = data[(data['hottness'] != 0) & (data['loudness'] != 0) & (data['tempo'] != 0)]
        df_filtered.reset_index(drop=True, inplace=True)
        return df_filtered
    else:
        return data


def distance(song1, song2, alphas = [1,1,1,0.5,3], feature = 'all'):
    """
    song1, song2 : python native lists    format : [artist, title, album, similar, hottness, terms, terms-weights, loudness, tempo]
    alphas : python native list
    feature : string

    Calculates feature distance between song1 and song2. 
    """
    
    alpha_hot, alpha_loud, alpha_tempo, alpha_similar, alpha_terms = alphas
    artist, title1, album1, similar1, hot1, terms1, weights1, loud1, tempo1 = song1
    artist2, title2, album2, similar2, hot2, terms2, weights2, loud2, tempo2 = song2
    
    distance = 0
    if hot1 != 0 and hot2 != 0 and (feature == 'all' or feature == 'hottness'):
        distance += alpha_hot*abs(hot1-hot2)
    if loud1 != 0 and loud2 != 0 and (feature == 'all' or feature == 'loudness'):
        distance += alpha_loud*abs(loud1-loud2)
    if tempo1 != 0 and tempo2 != 0 and (feature == 'all' or feature == 'tempo'):
        distance += alpha_tempo*abs(tempo1-tempo2)

    if feature == 'all' or feature == 'similar':
        distance += alpha_similar*(1 - len([singer for singer in similar1 if singer in similar2])/100)


    if feature == 'all' or feature == 'terms':
        terms_set = { term for term in terms1+terms2 }
        shared_terms = [term for term in terms1 if term in terms2]
        shared_weights1 = []
        shared_weights2 = []
        weightsum = sum([float(w) for w in weights1])+sum([float(w) for w in weights2])
        for term in shared_terms:
            try:
                shared_weights1.append(float(weights1[terms1.index(term)]))
                shared_weights2.append(float(weights2[terms2.index(term)]))
            except:
                print(len(terms1), len(weights1), len(terms2), len(weights2), song2)
                print(weights1[terms1.index(term)],weights2[terms2.index(term)],term)
        try:
            distance += alpha_terms*(1 - sum([shared_weights1[i] + shared_weights2[i] for i in range(len(shared_weights1))])/weightsum)
        except:
            pass
    
    return distance


def build_distances(df: pd.core.frame.DataFrame, alphas = [1,1,1,1,1]):
    """
    input: dataFrame of songs
    output: list of lists

    Creates matrix of distances.
    """
    M = len(df)
    distances = [[0 for i in range(M)] for j in range(M)]
    for j in range(M):
        for i in range(j+1,M):
            distances[i][j] = distance(df.iloc[j], df.iloc[i], alphas)
    return distances


def get_song_index(df, title, artist_name, release):
    """
    Get the index of a song in the DataFrame based on its title, artist name, and release.
    """
    title_mask = df['title'] == title
    artist_mask = df['artist_name'] == artist_name
    release_mask = df['release'] == release
    combined_mask = title_mask & artist_mask & release_mask
    index = df.index[combined_mask].tolist()
    
    if index:
        return index[0]
    else:
        return None


def newformat(input: list, df: pd.core.frame.DataFrame):
    """
    input: list of song name, title and release and dataframe of songs
    output: list of song (data objects) and respective indices
    doesn't use alphabetization
    """
    indices = [0]*len(input)
    result = []
    
    for i in range(len(input)):
        index = get_song_index(df, input[i][0], input[i][1], input[i][2])
        result.append(df.iloc[index])
        indices[i] = index

    return result, indices

def format_input(input: list, df: pd.core.frame.DataFrame):
    """
    input: list of song names, dataFrame of songs
    output: list of songs, list of indexes of songs in dataFrame
    """
    d = df.copy()
    alphabetical.sort_dataframe(d)
    index_dict = alphabetical.get_alpha_dict(d)
    
    indexes = [0]*len(input)
    result = []
    for i in range(len(input)):
        index = alphabetical.get_song_index(input[i][0], input[i][1], input[i][2], d,index_dict)
        result.append(d.iloc[index])
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


def compute_distance_to_playlist1(playlist: list, indexes: list, df: pd.core.frame.DataFrame, alphas=[1,1,1,1,1]):
    """
    input: list of songs, list of indexes, distance matrix
    output: distance to playlist matrix, total distance vector
    """

    M = len(df)
    N = len(indexes)
    distances = [[0 for i in range(N)] for j in range(M)]
    for j in range(M):
        for i in range(N):
            distances[j][i] = distance(df.iloc[j], playlist[i], alphas = alphas)
    total = []
    distances = pd.DataFrame(distances)
    for i in indexes:
        distances.drop(i, inplace=True)
    for i in range(len(distances)):
        total.append(sum(distances.iloc[i]))
    return distances, total

def compute_distance_to_playlist(playlist: list, indexes: list, df: pd.core.frame.DataFrame, alphas=[1,1,1,1,1]):
    """
    input: list of songs, list of indexes, DataFrame of songs, alphas for distance calculation
    output: distance to playlist matrix, total distance vector
    """
    df_filtered = df.drop(indexes)
    M = len(df_filtered)
    N = len(playlist)

    distances = [[0 for i in range(N)] for j in range(M)]

    for j in range(M):
        for i in range(N):
            distances[j][i] = distance(df_filtered.iloc[j], playlist[i], alphas)

    distances_df = pd.DataFrame(distances)

    total = distances_df.sum(axis=1).tolist()

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