import os
import os.path as path
from shutil import copyfile, rmtree
import numpy as np
from . import cmd, document_parser
from argparse import ArgumentParser
"""
song_diversity is intended to change the number of songs to be trained with
songs should already be sampled
While the entire dataset does not have to be in the same audio format, this script assumes that all audio files of the same song are of the same format
"""

def array_parse(string, _type=str):
    if string[0] == "[" and string[-1] == "]":
        string = string[1:-1]
    array = string.split(",")
    for i in range(len(array)):
        array[i] = _type(array[i].strip("'\" "))
    return array


def song_diversity(**kwargs):
    # Parse Arrays
    num_songs = array_parse(kwargs["num_songs"], _type=int)
    target_paths = array_parse(kwargs["target_paths"])
    instrument_types = array_parse(kwargs["instrument_types"])

    src_path = kwargs["source_path"]

    # obtain song list
    song_list = sorted(os.listdir(path.join(src_path, instrument_types[0])))
    for num, target_path in zip(num_songs, target_paths):
        if path.exists(target_path):
            rmtree(target_path)
        os.mkdir(target_path)
        if len(song_list) < num:
            print("num_songs is greater than the number of total songs")
            print("Choosing all songs")
            num = len(song_list)
        selected = np.random.choice(len(song_list), num, replace=False)

        selected_songs = []
        for i in selected:
            selected_songs.append(song_list[i])
    
        for instrument in instrument_types:
            dst_instrument = path.join(target_path, instrument)
            os.mkdir(dst_instrument)
            src_instrument = path.join(src_path, instrument)
            for song in selected_songs:
                dst_song = path.join(dst_instrument, song)
                src_song = path.join(src_instrument, song)
                copyfile(src_song, dst_song)

@document_parser('song_diversity', 'scripts.song_diversity.song_diversity')
def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--num_songs",
        type=str,
        required=True,
        help="""Number of songs in each training set. Needs to be the same length as target_paths
        """
    )
    parser.add_argument(
        "--target_paths",
        type=str,
        required=True,
        help="""Target folders filepaths for each training set. Needs to be the same length has num_songs
        """
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="""Filepath with the current training set. Audio samples be sampled to 16k Hz, as well as organized to to be fed into scaper.
        """
    )
    parser.add_argument(
        "--instrument_types",
        type=str,
        required=True,
        help="""Expected instrument categories for the training set.
        """
    )
    return parser
                
if __name__ == "__main__":
    cmd(song_diversity, build_parser)
