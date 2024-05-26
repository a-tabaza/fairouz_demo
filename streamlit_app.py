import streamlit as st

from nomic_maps import md_string

st.set_page_config(
    page_title="Binding Text, Images, Graphs, and Audio for Music Representation Learning",
    page_icon="🎵",
)


st.title("Binding Text, Images, Graphs, and Audio for Music Representation Learning")
st.subheader("Abdulrahman Tabaza, Omar Quishawi, Abdelrahman Yaghi, Omar Qawasmeh")
st.write(
    "If you want to inspect the vector spaces that the model operates on, feel free to click the vector space button below."
)

with st.expander("Vector Spaces"):
    st.markdown(md_string, unsafe_allow_html=True)

with st.expander("Abstract"):
    st.markdown(
        """
        In the field of Information Retrieval and Natural Language Processing, text embeddings play a significant role in tasks such as classification, clustering, and topic modeling. However, extending these embeddings to abstract concepts such as music, which involves multiple modalities, presents a unique challenge. Our work addresses this challenge by integrating rich multi-modal data into a unified joint embedding space. This space includes textual, visual, acoustic, and graph-based modality features. By doing so, we mirror cognitive processes associated with music interaction and overcome the disjoint nature of individual modalities. The resulting joint low-dimensional vector space facilitates retrieval, clustering, embedding space arithmetic, and cross-modal retrieval tasks. Importantly, our approach carries implications for music information retrieval and recommendation systems. Furthermore, we propose a novel multi-modal model that integrates various data types—text, images, graphs, and audio—for music representation learning. Our model aims to capture the complex relationships between different modalities, enhancing the overall understanding of music. By combining textual descriptions, visual imagery, graph-based structures, and audio signals, we create a comprehensive representation that can be leveraged for a wide range of music-related tasks. Notably, our model demonstrates promising results in music classification and recommendation systems.
        """
    )

import networkx as nx
from streamlit_d3graph import d3graph
import random
import pandas as pd
import hashlib

st.subheader("Select an artist and song to get started")

from collections import namedtuple
import json
import numpy as np
from numpy.linalg import norm

import faiss


key_to_track_id = json.load(open("data/id_to_track_mapping.json"))
track_id_to_key = {v: k for k, v in key_to_track_id.items()}

tracks = json.load(open("data/tracks.json"))

# fairouz_index = Index.restore("data/fairouz_index.usearch")
# image_index = Index.restore("data/image_index.usearch")
# audio_index = Index.restore("data/audio_index.usearch")
# text_index = Index.restore("data/text_index.usearch")
# graph_index = Index.restore("data/graph_index.usearch")

fairouz_index = faiss.read_index("data/fairouz_index.faiss")
image_index = faiss.read_index("data/image_index.faiss")
audio_index = faiss.read_index("data/audio_index.faiss")
text_index = faiss.read_index("data/text_index.faiss")
graph_index = faiss.read_index("data/graph_index.faiss")

fairouz_embeddings = np.load("data/fairouz_np.npy")
image_embeddings = np.load("data/image_np.npy")
audio_embeddings = np.load("data/audio_np.npy")
text_embeddings = np.load("data/text_np.npy")
graph_embeddings = np.load("data/graph_np.npy")

Explanation = namedtuple(
    "Explanation",
    [
        "fairouz_similarity",
        "image_similarity",
        "audio_similarity",
        "text_similarity",
        "graph_similarity",
    ],
)

idify = lambda my_string: str(hashlib.md5(my_string.encode()).hexdigest())


def load_to_graph(tracks, graph):
    track_set = set()
    artist_set = set()
    album_set = set()
    genre_set = set()

    for track_id in tracks:

        track = tracks[track_id]

        track_title = track["track_title"]
        artist_name = track["artist_name"]
        album_name = track["album_name"]
        album_image = track["image"]

        if track_id not in track_set:
            graph.add_node(track_id, title=track_title, type="track")

        artist_id = "id" + idify("artist" + track["artist_name"])
        album_id = "id" + idify("album" + track["album_name"])
        genres_id = {}

        for genre in track["genres"]:
            if isinstance(genre, str):
                genre_id = "id" + idify("genre" + genre)
                genres_id[genre_id] = genre
            else:
                genre_id = "id" + idify("genre" + genre.name)
                genres_id[genre_id] = genre.name

        if artist_id not in artist_set:
            graph.add_node(artist_id, name=artist_name, type="artist")

        if album_id not in album_set and album_name != "":
            graph.add_node(
                album_id, title=album_name, hasImage=album_image, type="album"
            )
            graph.add_edge(album_id, artist_id, type="BY_ARTIST")

        for genre_id in genres_id:
            if genre_id not in genre_set:
                genre_name = genres_id[genre_id]
                graph.add_node(genre_id, name=genre_name, type="genre")
                genre_set.add(genre_id)

        if track_id not in track_set:
            for genre_id in genres_id:
                graph.add_edge(track_id, genre_id, type="HAS_GENRE")
            graph.add_edge(track_id, artist_id, type="BY_ARTIST")
            if album_name != "":
                graph.add_edge(track_id, album_id, type="PART_OF_ALBUM")

        track_set.add(track_id)
        artist_set.add(artist_id)
        album_set.add(album_id)


full_graph = nx.DiGraph()
load_to_graph(tracks, full_graph)


# The reason behind this is that you CANNOT return a graph,
#  nx didn't implement it i dont think. So I have to create a global graph,
#  then populate it inside the function.
# okay bro it's not that deep
def create_d3_graph(graph, id):
    d3 = d3graph()

    node_types_colors = {
        "track": "#F26419",
        "artist": "#F6AE2D",
        "album": "#86BBD8",
        "genre": "#E2EBF3",
    }

    chosen_nodes = set()
    chosen_edges = set()

    edge_properties = {}
    node_properties = {}
    base_graph = nx.DiGraph()
    other_songs = set()

    for node in nx.ego_graph(graph, id).nodes.data():
        if node[1]["type"] == "genre":
            poss_songs = random.sample(
                list(graph.in_edges(node[0])),
                (
                    len(list(graph.in_edges(node[0])))
                    if len(list(graph.in_edges(node[0]))) < 2
                    else 2
                ),
            )
            for poss_song in poss_songs:
                other_songs.add(poss_song[0])

        if node[1]["type"] == "artist":
            poss_songs = random.sample(
                list(graph.in_edges(node[0])),
                (
                    len(list(graph.in_edges(node[0])))
                    if len(list(graph.in_edges(node[0]))) < 2
                    else 2
                ),
            )
            for poss_song in poss_songs:
                other_songs.add(poss_song[0])

    base_graph = nx.ego_graph(graph, id)

    for pos in other_songs:
        base_graph = nx.compose(base_graph, nx.ego_graph(graph, pos))

    for i in base_graph.nodes():
        node_properties.update(
            {
                i: {
                    "name": i,
                    "marker": "circle",
                    "label": (
                        graph.nodes[i]["name"]
                        if "name" in graph.nodes[i]
                        else graph.nodes[i]["title"]
                    ),
                    "tooltip": i,
                    "color": node_types_colors[graph.nodes[i]["type"]],
                    "opacity": "0.99",
                    "fontcolor": "#F0F0F0",
                    "fontsize": 12,
                    "size": 13.0,
                    "edge_size": 1,
                    "edge_color": "#000000",
                    "group": 1,
                }
            }
        )

    for i in base_graph.edges():
        edge_properties.update(
            {
                (i[0], i[1]): {
                    "weight": 1.0,
                    "weight_scaled": 1.0,
                    "edge_distance": 50.0,
                    "edge_style": 0,
                    "color": "#808080",
                    "marker_start": "",
                    "marker_end": "arrow",
                    "marker_color": "#808080",
                    "label": "",
                    "label_color": "#808080",
                    "label_fontsize": 8,
                }
            }
        )

    d3.graph(
        pd.DataFrame(
            nx.adjacency_matrix(base_graph).toarray(),
            columns=base_graph.nodes(),
            index=base_graph.nodes(),
        )
    )
    d3.node_properties = node_properties
    d3.edge_properties = edge_properties
    return d3


def explainability(query_track: int, similar_track: int) -> Explanation:
    """Compute the cosine similarity between the query and similar tracks for each modality."""

    query_fairouz_embedding = fairouz_embeddings[int(query_track)]
    similar_fairouz_embedding = fairouz_embeddings[int(similar_track)]
    query_image_embedding = image_embeddings[int(query_track)]
    similar_image_embedding = image_embeddings[int(similar_track)]
    query_audio_embedding = audio_embeddings[int(query_track)]
    similar_audio_embedding = audio_embeddings[int(similar_track)]
    query_text_embedding = text_embeddings[int(query_track)]
    similar_text_embedding = text_embeddings[int(similar_track)]
    query_graph_embedding = graph_embeddings[int(query_track)]
    similar_graph_embedding = graph_embeddings[int(similar_track)]

    fairouz_similarity = np.dot(query_fairouz_embedding, similar_fairouz_embedding) / (
        norm(query_fairouz_embedding) * norm(similar_fairouz_embedding)
    )

    image_similarity = np.dot(query_image_embedding, similar_image_embedding) / (
        norm(query_image_embedding) * norm(similar_image_embedding)
    )
    audio_similarity = np.dot(query_audio_embedding, similar_audio_embedding) / (
        norm(query_audio_embedding) * norm(similar_audio_embedding)
    )

    # norm of the zero vector is zero, so we need to handle this case
    if (
        np.array_equal(
            np.array(query_text_embedding), np.zeros(query_text_embedding.shape)
        )
    ) or (
        np.array_equal(
            np.array(similar_text_embedding), np.zeros(similar_text_embedding.shape)
        )
    ):
        text_similarity = 0

    else:
        text_similarity = np.dot(query_text_embedding, similar_text_embedding) / (
            norm(query_text_embedding) * norm(similar_text_embedding)
        )

    graph_similarity = np.dot(query_graph_embedding, similar_graph_embedding) / (
        norm(query_graph_embedding) * norm(similar_graph_embedding)
    )

    return Explanation(
        fairouz_similarity,
        image_similarity,
        audio_similarity,
        text_similarity,
        graph_similarity,
    )


artists = list(set([item["artist_name"] for item in tracks.values()]))
selected_artist = st.selectbox("Select Artist", artists)
selected_song = st.selectbox(
    "Select Song",
    [
        track["track_title"]
        for track in tracks.values()
        if track["artist_name"] == selected_artist
    ],
)

id_and_track = [
    (id, track) for id, track in tracks.items() if track["track_title"] == selected_song
]
id = id_and_track[0][0]
track = id_and_track[0][1]

with st.expander(f"Lyrics"):
    lyrics = track["lyrics"]["lyrics"]
    if lyrics != "":
        emotional_tone = ", ".join(track["lyrics"]["emotional"])
        keywords = ", ".join(track["lyrics"]["context"])
        summary = track["lyrics"]["summary"]
        st.write(f"Emotional Tone: {emotional_tone}")
        st.write(f"Keywords: {keywords}")
        st.write(f"Summary: {summary}")

        if st.button("View Lyrics"):
            st.write(lyrics)
    else:
        st.write("No Lyrics Available")

with st.expander("Album"):
    st.write(f"{track['album_name']}")
    st.image(track["image"], caption="Album Art")

with st.expander(f"Graph"):
    if st.button("Show Sub Graph"):
        create_d3_graph(full_graph, id).show(
            figsize=(600, 500), show_slider=False, save_button=False
        )

with st.expander("Audio"):
    st.audio(track["preview_url"])

fairouz_tab, image_tab, audio_tab, text_tab, graph_tab = st.tabs(
    ["Fairouz", "Image", "Audio", "Text", "Graph"]
)

with fairouz_tab:
    key = track_id_to_key[id]
    query_key = track_id_to_key[id]
    fairouz_embedding = fairouz_embeddings[int(key)]
    f_D, f_I = fairouz_index.search(fairouz_embedding.reshape(1, -1), 5)
    st.write("Similar Tracks based on Fairouz Embeddings")
    for i, (key, score) in enumerate(zip(f_I[0], f_D[0])):
        if int(key) != int(track_id_to_key[id]):
            track_id = key_to_track_id[str(key)]
            track = tracks[track_id]
            st.write(f"{track['track_title']} by {track['artist_name']}")
            with st.expander(f"Lyrics"):
                lyrics = track["lyrics"]["lyrics"]
                if lyrics != "":
                    emotional_tone = ", ".join(track["lyrics"]["emotional"])
                    keywords = ", ".join(track["lyrics"]["context"])
                    summary = track["lyrics"]["summary"]
                    st.write(f"Emotional Tone: {emotional_tone}")
                    st.write(f"Keywords: {keywords}")
                    st.write(f"Summary: {summary}")

                    if st.button("View Lyrics", key=i + 52):
                        st.write(lyrics)
                else:
                    st.write("No Lyrics Available")
            with st.expander("Album"):
                st.write(f"{track['album_name']}")
                st.image(track["image"], caption="Album Art")
            with st.expander("Graph"):
                if st.button("Show Subgraph", key=key + 104):
                    d3_temp = create_d3_graph(full_graph, key_to_track_id[str(key)])
                    d3_temp.show(
                        figsize=(600, 500), show_slider=False, save_button=False
                    )
            with st.expander("Audio"):
                st.audio(track["preview_url"])
            with st.expander("Explainability"):
                sim_scores = explainability(query_key, key)
                st.write(
                    f"Cosine Similarity between {track['track_title']} and {selected_song}:"
                )
                st.write(f"Fairouz Similarity: {sim_scores.fairouz_similarity}")
                st.write(f"Image Similarity: {sim_scores.image_similarity}")
                st.write(f"Audio Similarity: {sim_scores.audio_similarity}")
                st.write(f"Text Similarity: {sim_scores.text_similarity}")
                st.write(f"Graph Similarity: {sim_scores.graph_similarity}")

with image_tab:
    key = track_id_to_key[id]
    image_embedding = image_embeddings[int(key)]
    i_D, i_I = image_index.search(image_embedding.reshape(1, -1), 5)
    query_key = track_id_to_key[id]
    st.write("Similar Tracks based on Image Embeddings")
    for i, (key, score) in enumerate(zip(i_I[0], i_D[0])):
        if int(key) != int(track_id_to_key[id]):
            track_id = key_to_track_id[str(key)]
            track = tracks[track_id]
            st.write(f"{track['track_title']} by {track['artist_name']}")
            with st.expander(f"Lyrics"):
                lyrics = track["lyrics"]["lyrics"]
                if lyrics != "":
                    emotional_tone = ", ".join(track["lyrics"]["emotional"])
                    keywords = ", ".join(track["lyrics"]["context"])
                    summary = track["lyrics"]["summary"]
                    st.write(f"Emotional Tone: {emotional_tone}")
                    st.write(f"Keywords: {keywords}")
                    st.write(f"Summary: {summary}")

                    if st.button("View Lyrics", key=i + 69):
                        st.write(lyrics)
                else:
                    st.write("No Lyrics Available")
            with st.expander("Album"):
                st.write(f"{track['album_name']}")
                st.image(track["image"], caption="Album Art")
            with st.expander("Graph"):
                if st.button("Show Subgraph", key=key + 130):
                    create_d3_graph(full_graph, key_to_track_id[str(key)]).show(
                        figsize=(600, 500), show_slider=False, save_button=False
                    )
            with st.expander("Audio"):
                st.audio(track["preview_url"])
            with st.expander("Explainability"):
                sim_scores = explainability(query_key, key)
                st.write(
                    f"Cosine Similarity between {track['track_title']} and {selected_song}:"
                )
                st.write(f"Fairouz Similarity: {sim_scores.fairouz_similarity}")
                st.write(f"Image Similarity: {sim_scores.image_similarity}")
                st.write(f"Audio Similarity: {sim_scores.audio_similarity}")
                st.write(f"Text Similarity: {sim_scores.text_similarity}")
                st.write(f"Graph Similarity: {sim_scores.graph_similarity}")

with audio_tab:
    key = track_id_to_key[id]
    audio_embedding = audio_embeddings[int(key)]
    a_D, a_I = audio_index.search(audio_embedding.reshape(1, -1), 5)
    query_key = track_id_to_key[id]
    st.write("Similar Tracks based on Audio Embeddings")
    for i, (key, score) in enumerate(zip(a_I[0], a_D[0])):
        if int(key) != int(track_id_to_key[id]):
            track_id = key_to_track_id[str(key)]
            track = tracks[track_id]
            st.write(f"{track['track_title']} by {track['artist_name']}")
            with st.expander(f"Lyrics"):
                lyrics = track["lyrics"]["lyrics"]
                if lyrics != "":
                    emotional_tone = ", ".join(track["lyrics"]["emotional"])
                    keywords = ", ".join(track["lyrics"]["context"])
                    summary = track["lyrics"]["summary"]
                    st.write(f"Emotional Tone: {emotional_tone}")
                    st.write(f"Keywords: {keywords}")
                    st.write(f"Summary: {summary}")

                    if st.button("View Lyrics", key=i + 26):
                        st.write(lyrics)
                else:
                    st.write("No Lyrics Available")
            with st.expander("Album"):
                st.write(f"{track['album_name']}")
                st.image(track["image"], caption="Album Art")
            with st.expander("Graph"):
                if st.button("Show Subgraph", key=key + 156):
                    create_d3_graph(full_graph, key_to_track_id[str(key)]).show(
                        figsize=(600, 500), show_slider=False, save_button=False
                    )
            with st.expander("Audio"):
                st.audio(track["preview_url"])
            with st.expander("Explainability"):
                sim_scores = explainability(query_key, key)
                st.write(
                    f"Cosine Similarity between {track['track_title']} and {selected_song}:"
                )
                st.write(f"Fairouz Similarity: {sim_scores.fairouz_similarity}")
                st.write(f"Image Similarity: {sim_scores.image_similarity}")
                st.write(f"Audio Similarity: {sim_scores.audio_similarity}")
                st.write(f"Text Similarity: {sim_scores.text_similarity}")
                st.write(f"Graph Similarity: {sim_scores.graph_similarity}")

with text_tab:
    key = track_id_to_key[id]
    text_embedding = text_embeddings[int(key)]
    t_D, t_I = text_index.search(text_embedding.reshape(1, -1), 5)
    query_key = track_id_to_key[id]
    st.write("Similar Tracks based on Text Embeddings")
    for i, (key, score) in enumerate(zip(t_I[0], t_D[0])):
        if int(key) != int(track_id_to_key[id]):
            track_id = key_to_track_id[str(key)]
            track = tracks[track_id]
            st.write(f"{track['track_title']} by {track['artist_name']}")
            with st.expander(f"Lyrics"):
                lyrics = track["lyrics"]["lyrics"]
                if lyrics != "":
                    emotional_tone = ", ".join(track["lyrics"]["emotional"])
                    keywords = ", ".join(track["lyrics"]["context"])
                    summary = track["lyrics"]["summary"]
                    st.write(f"Emotional Tone: {emotional_tone}")
                    st.write(f"Keywords: {keywords}")
                    st.write(f"Summary: {summary}")

                    if st.button("View Lyrics", key=i):
                        st.write(lyrics)
                else:
                    st.write("No Lyrics Available")
            with st.expander("Album"):
                st.write(f"{track['album_name']}")
                st.image(track["image"], caption="Album Art")
            with st.expander("Graph"):
                if st.button("Show Subgraph", key=key + 1850):
                    create_d3_graph(full_graph, key_to_track_id[str(key)]).show(
                        figsize=(600, 500), show_slider=False, save_button=False
                    )
            with st.expander("Audio"):
                st.audio(track["preview_url"])
            with st.expander("Explainability"):
                sim_scores = explainability(query_key, key)
                st.write(
                    f"Cosine Similarity between {track['track_title']} and {selected_song}:"
                )
                st.write(f"Fairouz Similarity: {sim_scores.fairouz_similarity}")
                st.write(f"Image Similarity: {sim_scores.image_similarity}")
                st.write(f"Audio Similarity: {sim_scores.audio_similarity}")
                st.write(f"Text Similarity: {sim_scores.text_similarity}")
                st.write(f"Graph Similarity: {sim_scores.graph_similarity}")

with graph_tab:
    key = track_id_to_key[id]
    graph_embedding = graph_embeddings[int(key)]
    g_D, g_I = graph_index.search(graph_embedding.reshape(1, -1), 5)
    query_key = track_id_to_key[id]
    st.write("Similar Tracks based on Graph Embeddings")
    for i, (key, score) in enumerate(zip(g_I[0], g_D[0])):
        if int(key) != int(track_id_to_key[id]):
            track_id = key_to_track_id[str(key)]
            track = tracks[track_id]
            st.write(f"{track['track_title']} by {track['artist_name']}")
            with st.expander(f"Lyrics"):
                lyrics = track["lyrics"]["lyrics"]
                if lyrics != "":
                    emotional_tone = ", ".join(track["lyrics"]["emotional"])
                    keywords = ", ".join(track["lyrics"]["context"])
                    summary = track["lyrics"]["summary"]
                    st.write(f"Emotional Tone: {emotional_tone}")
                    st.write(f"Keywords: {keywords}")
                    st.write(f"Summary: {summary}")

                    if st.button("View Lyrics", key=i + 5):
                        st.write(lyrics)
                else:
                    st.write("No Lyrics Available")
            with st.expander("Album"):
                st.write(f"{track['album_name']}")
                st.image(track["image"], caption="Album Art")
            with st.expander("Graph"):
                if st.button("Show Subgraph", key=key + 2085):
                    create_d3_graph(full_graph, key_to_track_id[str(key)]).show(
                        figsize=(600, 500), show_slider=False, save_button=False
                    )
            with st.expander("Audio"):
                st.audio(track["preview_url"])
            with st.expander("Explainability"):
                sim_scores = explainability(query_key, key)
                st.write(
                    f"Cosine Similarity between {track['track_title']} and {selected_song}:"
                )
                st.write(f"Fairouz Similarity: {sim_scores.fairouz_similarity}")
                st.write(f"Image Similarity: {sim_scores.image_similarity}")
                st.write(f"Audio Similarity: {sim_scores.audio_similarity}")
                st.write(f"Text Similarity: {sim_scores.text_similarity}")
                st.write(f"Graph Similarity: {sim_scores.graph_similarity}")

track_names = {track["track_title"]: id for id, track in tracks.items()}

st.title("Smart Shuffle")

st.write(
    "You: can I have spotify smart shuffle?\n\nMom: we have spotify smart shuffle at home\n\nSpotify smart shuffle at home:"
)

st.write("Select a song to start the Smart Shuffle!")
selected_track = st.multiselect("Select songs", list(track_names.keys()))

smart_shuffle_tracks = []
if st.button("Smart Shuffle"):
    for s_track in selected_track:
        s_id = track_names[s_track]
        s_key = track_id_to_key[s_id]
        s_fairouz_embedding = fairouz_embeddings[int(s_key)]
        sf_D, sf_I = fairouz_index.search(s_fairouz_embedding.reshape(1, -1), 3)
        for i, (key, score) in enumerate(zip(f_I[0], f_D[0])):
            if int(key) != int(track_id_to_key[s_id]):
                ss_id = key_to_track_id[str(key)]
                ss_track = tracks[ss_id]
                smart_shuffle_tracks.append(ss_track)

    st.write("### Smart Shuffle Results:")
    for track in smart_shuffle_tracks:
        st.write(track["track_title"], track["artist_name"], track["album_name"])
        with st.expander(f"Lyrics"):
            lyrics = track["lyrics"]["lyrics"]
            if lyrics != "":
                emotional_tone = ", ".join(track["lyrics"]["emotional"])
                keywords = ", ".join(track["lyrics"]["context"])
                summary = track["lyrics"]["summary"]
                st.write(f"Emotional Tone: {emotional_tone}")
                st.write(f"Keywords: {keywords}")
                st.write(f"Summary: {summary}")
                if st.button("View Lyrics", key=i + 5252 + np.random.randint(1000)):
                    st.write(lyrics)
            else:
                st.write("No lyrics available for this song.")

        with st.expander(f"Album Art"):
            st.image(track["image"])

        with st.expander(f"Audio"):
            st.audio(track["preview_url"], format="audio/mp3")

        st.write("----")
