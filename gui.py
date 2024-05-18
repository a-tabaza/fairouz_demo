import streamlit as st

st.set_page_config(
    page_title="Binding Text, Images, Graphs, and Audio for Music Representation Learning",
    page_icon="🎵",
)

import json
import numpy as np
from numpy.linalg import norm
from usearch.index import Index

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


def explainability(query_track, similar_track):
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

    return (
        fairouz_similarity,
        image_similarity,
        audio_similarity,
        text_similarity,
        graph_similarity,
    )


st.title("Binding Text, Images, Graphs, and Audio for Music Representation Learning")

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
                st.write(f"Fairouz Similarity: {sim_scores[0]}")
                st.write(f"Image Similarity: {sim_scores[1]}")
                st.write(f"Audio Similarity: {sim_scores[2]}")
                st.write(f"Text Similarity: {sim_scores[3]}")
                st.write(f"Graph Similarity: {sim_scores[4]}")

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
                st.write(f"Fairouz Similarity: {sim_scores[0]}")
                st.write(f"Image Similarity: {sim_scores[1]}")
                st.write(f"Audio Similarity: {sim_scores[2]}")
                st.write(f"Text Similarity: {sim_scores[3]}")
                st.write(f"Graph Similarity: {sim_scores[4]}")

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
                st.write(f"Fairouz Similarity: {sim_scores[0]}")
                st.write(f"Image Similarity: {sim_scores[1]}")
                st.write(f"Audio Similarity: {sim_scores[2]}")
                st.write(f"Text Similarity: {sim_scores[3]}")
                st.write(f"Graph Similarity: {sim_scores[4]}")

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
                st.write(f"Fairouz Similarity: {sim_scores[0]}")
                st.write(f"Image Similarity: {sim_scores[1]}")
                st.write(f"Audio Similarity: {sim_scores[2]}")
                st.write(f"Text Similarity: {sim_scores[3]}")
                st.write(f"Graph Similarity: {sim_scores[4]}")

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
                st.write(f"Fairouz Similarity: {sim_scores[0]}")
                st.write(f"Image Similarity: {sim_scores[1]}")
                st.write(f"Audio Similarity: {sim_scores[2]}")
                st.write(f"Text Similarity: {sim_scores[3]}")
                st.write(f"Graph Similarity: {sim_scores[4]}")
