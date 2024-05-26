import streamlit as st

from nomic_maps import md_string

st.set_page_config(
    page_title="Home",
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
