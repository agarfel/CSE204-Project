import pandas as pd
import numpy as np

def sort_dataframe(song_df):
        song_df.sort_values(['title', 'artist_name'], ascending=True, ignore_index=True, inplace=True)

def get_alpha_dict(sorted_song_df):
        alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M',\
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        index_dict = { a : sorted_song_df['title'].str.startswith(a).idxmax() for a in alpha }
        index_dict['*'] = sorted_song_df.index[-1]
        return index_dict


def get_index_song(title, artist,sorted_df,alpha_dict):

    title, artist = title.upper(), artist.upper()

    if (title[0] in alpha_dict.keys() ) and (title[0] != 'Z') :
        start_idx = alpha_dict[title[0]]
        end_idx = min([ idx for idx in alpha_dict.values() if idx>start_idx])
    else :
        start_idx = alpha_dict['Z']
        end_idx = alpha_dict['*'] + 1

    small_df = sorted_df.iloc[start_idx:end_idx]
    s_df = small_df.copy()
    s_df.loc[:, 'title'] = s_df['title'].str.upper()
    s_df.loc[:, 'artist_name'] = s_df['artist_name'].str.upper()

    idx = s_df.loc[(s_df['title'] == title) & (s_df['artist_name'] == artist)].index
    if idx.empty :
        print("Song not found")
        return -1
    
    return idx.to_list()[0]

