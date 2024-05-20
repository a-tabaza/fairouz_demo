import hashlib
import networkx as nx

idify = lambda my_string: str(hashlib.md5(my_string.encode()).hexdigest())

def tracks_to_networkx(tracks_dict, graph):
    track_set = set()
    artist_set = set()
    album_set = set()
    genre_set = set()

    for track_id in tracks_dict:

        #################
        ### Add Track ###
        #################
        track = tracks_dict[track_id]

        track_title = track['track_title']
        artist_name = track['artist_name']
        album_name = track['album_name']
        album_image = track['image']

        if track_id not in track_set:
            graph.add_node(track_id, title=track_title, type='track')

        artist_id = 'id' + idify('artist' +track['artist_name'])
        album_id = 'id' + idify('album' +track['album_name'])
        genres_id  = {}

        for genre in track['genres']:
            if isinstance(genre, str):
                genre_id = 'id' + idify('genre' + genre)
                genres_id[genre_id] = genre
            else:
                genre_id = 'id' + idify('genre' + genre.name)
                genres_id[genre_id] = genre.name

        ##################
        ### Add Artist ###
        ##################

        if artist_id not in artist_set:
            graph.add_node(artist_id, name=artist_name, type='artist')

        #################
        ### Add Album ###
        #################
        if album_id not in album_set and album_name != '':
            graph.add_node(album_id, title=album_name, hasImage=album_image, type='album')
            graph.add_edge(album_id, artist_id, type='BY_ARTIST')

        ##################
        ### Add Genres ###
        ##################

        for genre_id in genres_id:
            if genre_id not in genre_set:
                genre_name = genres_id[genre_id]
                graph.add_node(genre_id, name=genre_name, type='genre')
                genre_set.add(genre_id)

        if track_id not in track_set:
            for genre_id in genres_id:
                graph.add_edge(track_id, genre_id, type='HAS_GENRE')
            graph.add_edge(track_id, artist_id, type='BY_ARTIST')
            if album_name != '':
                graph.add_edge(track_id, album_id, type='PART_OF_ALBUM')

        # graph.add_edge(track_id, user_id, type='LIKED_BY')

        track_set.add(track_id)
        artist_set.add(artist_id)
        album_set.add(album_id)

# def create_node_properties():
