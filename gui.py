import streamlit as st
from graph_function import tracks_to_networkx
import networkx as nx
from streamlit_d3graph import d3graph
import random
import pandas as pd

st.set_page_config(
    page_title="Binding Text, Images, Graphs, and Audio for Music Representation Learning",
    page_icon="🎵",
)

from collections import namedtuple
import json
import numpy as np
from numpy.linalg import norm
from usearch.index import Index

from nomic_maps import md_string

key_to_track_id = json.load(open("data/id_to_track_mapping.json"))
track_id_to_key = {v: k for k, v in key_to_track_id.items()}

tracks = json.load(open("data/tracks.json"))

fairouz_index = Index.restore("data/fairouz_index.usearch")
image_index = Index.restore("data/image_index.usearch")
audio_index = Index.restore("data/audio_index.usearch")
text_index = Index.restore("data/text_index.usearch")
graph_index = Index.restore("data/graph_index.usearch")

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


st.title("Binding Text, Images, Graphs, and Audio for Music Representation Learning")
st.header("Explore Music Similarity Across Multiple Modalities")
st.subheader("Select an artist and song to get started")
st.write(
    "If you want to inspect the vector spaces that the model operates on, feel free to click the vector space button below."
)

if st.button("View Vector Spaces"):
    st.markdown(md_string, unsafe_allow_html=True)

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

tracks_dict = json.load(open('tracks_contextualized.json', 'r'))

d3 = d3graph()
dg = nx.DiGraph()
tracks_to_networkx(tracks_dict, dg)

node_types_colors = {
    "track": "#F26419",
    "artist":"#F6AE2D",
    "album": "#86BBD8",
    "genre": "#E2EBF3",
}

chosen_nodes = set()
chosen_edges = set()

edge_properties = {}
node_properties = {}

with st.expander(f"Show Sub Graph"):
    
    base_graph = nx.DiGraph()
    other_songs = set()

    for node in nx.ego_graph(dg, id).nodes.data():
        if node[1]['type'] == 'genre':
            poss_songs = random.sample(list(dg.in_edges(node[0])), len(list(dg.in_edges(node[0]))) if len(list(dg.in_edges(node[0]))) < 2 else 2)
            for poss_song in poss_songs:
                other_songs.add(poss_song[0])

        if node[1]['type'] == 'artist':
            poss_songs = random.sample(list(dg.in_edges(node[0])), len(list(dg.in_edges(node[0]))) if len(list(dg.in_edges(node[0]))) < 2 else 2)
            for poss_song in poss_songs:
                other_songs.add(poss_song[0])

    base_graph = nx.ego_graph(dg, id)

    for pos in other_songs:
        base_graph = nx.compose(base_graph, nx.ego_graph(dg, pos))
        
    for i in base_graph.nodes():
        node_properties.update({i: 
            {'name': i,
            'marker': 'circle', 
            'label': dg.nodes[i]['name'] if 'name' in dg.nodes[i] else dg.nodes[i]['title'], 
            'tooltip': i, 
            'color': node_types_colors[dg.nodes[i]['type']], 
            'opacity': '0.99', 
            'fontcolor': '#F0F0F0', 
            'fontsize': 12, 
            'size': 13.0, 
            'edge_size': 1, 
            'edge_color': '#000000',
            'group': 1}}
            )

    for i in base_graph.edges():
        edge_properties.update({(i[0], i[1]):
                    {'weight': 1.0,
                    'weight_scaled': 1.0,
                    'edge_distance': 50.0, 
                    'edge_style': 0, 
                    'color': '#808080', 
                    'marker_start': '', 
                    'marker_end': 'arrow', 
                    'marker_color': '#808080', 
                    'label': '', 
                    'label_color': '#808080', 
                    'label_fontsize': 8}})

    d3.graph(pd.DataFrame(nx.adjacency_matrix(base_graph).toarray(), columns = base_graph.nodes(), index = base_graph.nodes()))
    d3.node_properties = node_properties
    d3.edge_properties = edge_properties

    d3.show(figsize=(600, 500))




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

with st.expander("Audio"):
    st.audio(track["preview_url"])

fairouz_tab, image_tab, audio_tab, text_tab, graph_tab = st.tabs(
    ["Fairouz", "Image", "Audio", "Text", "Graph"]
)

with fairouz_tab:
    key = track_id_to_key[id]
    query_key = track_id_to_key[id]
    fairouz_embedding = fairouz_embeddings[int(key)]
    top_fairouz = fairouz_index.search(fairouz_embedding, 5)
    st.write("Similar Tracks based on Fairouz Embeddings")
    for i, (key, score) in enumerate(top_fairouz.to_list()):
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
    top_image = image_index.search(image_embedding, 5)
    query_key = track_id_to_key[id]
    st.write("Similar Tracks based on Image Embeddings")
    for i, (key, score) in enumerate(top_image.to_list()):
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
    top_audio = audio_index.search(audio_embedding, 5)
    query_key = track_id_to_key[id]
    st.write("Similar Tracks based on Audio Embeddings")
    for i, (key, score) in enumerate(top_audio.to_list()):
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
    top_text = text_index.search(text_embedding, 5)
    query_key = track_id_to_key[id]
    st.write("Similar Tracks based on Text Embeddings")
    for i, (key, score) in enumerate(top_text.to_list()):
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
    top_graph = graph_index.search(graph_embedding, 5)
    query_key = track_id_to_key[id]
    st.write("Similar Tracks based on Graph Embeddings")
    for i, (key, score) in enumerate(top_graph.to_list()):
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
