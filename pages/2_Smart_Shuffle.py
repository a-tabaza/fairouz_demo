import streamlit as st
import json


tracks = json.load(open("../data/tracks.json"))

track_names = {track["track_title"]: id for id, track in tracks.items()}


from collections import namedtuple
import json
import numpy as np
from numpy.linalg import norm

import faiss


key_to_track_id = json.load(open("../data/id_to_track_mapping.json"))
track_id_to_key = {v: k for k, v in key_to_track_id.items()}

fairouz_index = faiss.read_index("../data/fairouz_index.faiss")
fairouz_embeddings = np.load("../data/fairouz_np.npy")


st.title("Smart Shuffle")

st.write(
    "You: can I have spotify smart shuffle?\n\nMom: we have spotify smart shuffle at home\n\nSpotify smart shuffle at home:"
)

st.write("Select a song to start the Smart Shuffle!")
selected_track = st.multiselect("Select songs", list(track_names.keys()))

for s_track in selected_track:
    s_id = track_names[s_track]
    key = track_id_to_key[s_id]
    fairouz_embedding = fairouz_embeddings[int(key)]
    f_D, f_I = fairouz_index.search(fairouz_embedding.reshape(1, -1), 3)
    st.write(f"### {s_track}")
    for i, (key, score) in enumerate(zip(f_I[0], f_D[0])):
        if int(key) != int(track_id_to_key[s_id]):
            ss_id = key_to_track_id[str(key)]
            ss_track = tracks[ss_id]
            st.write(
                ss_track["track_title"], ss_track["artist_name"], ss_track["album_name"]
            )
            with st.expander(f"Lyrics"):
                lyrics = ss_track["lyrics"]["lyrics"]
                if lyrics != "":
                    emotional_tone = ", ".join(ss_track["lyrics"]["emotional"])
                    keywords = ", ".join(ss_track["lyrics"]["context"])
                    summary = ss_track["lyrics"]["summary"]
                    st.write(f"Emotional Tone: {emotional_tone}")
                    st.write(f"Keywords: {keywords}")
                    st.write(f"Summary: {summary}")
                    if st.button("View Lyrics", key=i + 5252 + np.random.randint(1000)):
                        st.write(lyrics)
                else:
                    st.write("No lyrics available for this song.")

            with st.expander(f"Album Art"):
                st.image(ss_track["image"])

            with st.expander(f"Audio"):
                st.audio(ss_track["preview_url"], format="audio/mp3")

            st.write("----")

smart_shuffle_tracks = []
if st.button("Smart Shuffle"):
    for s_track in selected_track:
        s_id = track_names[s_track]
        key = track_id_to_key[s_id]
        fairouz_embedding = fairouz_embeddings[int(key)]
        f_D, f_I = fairouz_index.search(fairouz_embedding.reshape(1, -1), 3)
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
