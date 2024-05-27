"""
This has a long string literal that I want to render in streamlit, but I want my streamlit script to look normal so I'm doing this. Sorry.
"""

vector_spaces = """
**Nomic Maps**

Text Embedding Maps
- [mixedbread-ai/mxbai-embed-large-v1](https://atlas.nomic.ai/data/omaralquishawi25/model-mxbai/map)

Image Embedding Maps
- [CLIP-ViT-B-32-laion2B](https://atlas.nomic.ai/data/omaralquishawi25/model-openclip-1/map)

Graph Embedding Maps
- [RandNE](https://atlas.nomic.ai/data/abd20200355/graph-embedding-map-for-model-randne/map)

Audio Embedding Maps
- [Vggish](https://atlas.nomic.ai/data/omaralquishawi25/all-music-embeddings-march-23rd---mean/map)

Fairouz Embedding Maps
- [51k Data Pairs](https://atlas.nomic.ai/data/tyqnology/fairouz-vggish-randne-openclip-mxbai-200-epochs-contracted-51k-datapoints-euclidian/map)
"""

abstract = """
In the field of Information Retrieval and Natural Language Processing, text embeddings play a significant role in tasks such as classification, clustering, and topic modeling. However, extending these embeddings to abstract concepts such as music, which involves multiple modalities, presents a unique challenge. Our work addresses this challenge by integrating rich multi-modal data into a unified joint embedding space. This space includes textual, visual, acoustic, and graph-based modality features. By doing so, we mirror cognitive processes associated with music interaction and overcome the disjoint nature of individual modalities. The resulting joint low-dimensional vector space facilitates retrieval, clustering, embedding space arithmetic, and cross-modal retrieval tasks. Importantly, our approach carries implications for music information retrieval and recommendation systems. Furthermore, we propose a novel multi-modal model that integrates various data types—text, images, graphs, and audio—for music representation learning. Our model aims to capture the complex relationships between different modalities, enhancing the overall understanding of music. By combining textual descriptions, visual imagery, graph-based structures, and audio signals, we create a comprehensive representation that can be leveraged for a wide range of music-related tasks. Notably, our model demonstrates promising results in music classification and recommendation systems.
"""

functionality = """
As soon as we trained our model, we embedded all the songs we had, resulting in an embedding per song, based on those embeddings, we added the following functionality:
- **Multimodal Search:** Given a song, we decompose the individual modalities that are present in that song, and explore similarities across these modalities, as well as their combined representation, formed by our model.
- **Recommendation:** Given a set of unique songs, we are able to generate a playlist of recommended tracks, most often, we find that our model generates excellent recommendations based on highly biased and subjective user feedback.
- **Explainablity:** Given a song, the model retrieves the most similar songs to the input song and explains why they are similar, based on each individual modality, and vector similarity.
- **Interactivity:** We added a component to interact with the knowledge graph, where you can explore the graph and see the songs that are connected to the selected song. We also link the `Nomic by Atlas` maps that we produced, these are by far the most interactive component of the demo, where you can explore the songs in a 2D space, and see the connections between them.

We use exhaustive flat L2 indexes from the `faiss` library to interface with any embeddings we use. 

All our data is attached to this demo (the GitHub repo) in the `data` directory, inlcuding the embeddings for all modalities.
"""