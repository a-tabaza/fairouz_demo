# Binding Text, Images, Graphs, and Audio for Music Representation Learning
This repo contains the code for a demo showcasing the model from our paper, [``Binding Text, Images, Graphs, and Audio for Music Representation Learning``](https://doi.org/10.1145/3660853.3660886)

For a detailed explanation of the model, please refer to the paper linked above.

# Functionality
As soon as we trained our model, we embedded all the songs we had, resulting in an embedding per song, based on those embeddings, we added the following functionality:
- **Multimodal Search:** Given a song, we decompose the individual modalities that are present in that song, and explore similarities across these modalities, as well as their combined representation, formed by our model.
- **Recommendation (Smart Shuffle):** Given a set of unique songs, we are able to generate a playlist of recommended tracks, most often, we find that our model generates excellent recommendations based on highly biased and subjective user feedback.
- **Explainablity:** Given a song, the model retrieves the most similar songs to the input song and explains why they are similar, based on each individual modality, and vector similarity, we use radar plots to represent similarity across 4-5 axes.
- **Interactivity:** We added a component to interact with the knowledge graph, where you can explore the graph and see the songs that are connected to the selected song. 

We also link the `Nomic by Atlas` maps that we produced, these are by far the most interactive component of the demo, where you can explore the songs in a 2D space, and see the connections between them.

We use exhaustive flat L2 indexes from the `faiss` library to interface with any embeddings we use, the exaustivity of the index is justified by the amount of data we have, but we plan on investigating approximate neighbor search in the future, once we train a full model with much more data. 

All our data is attached to this demo in the `data` directory, inlcuding the embeddings.

# Abstract
In the field of Information Retrieval and Natural Language Processing, text embeddings play a significant role in tasks such as classification, clustering, and topic modeling. However, extending these embeddings to abstract concepts such as music, which involves multiple modalities, presents a unique challenge. Our work addresses this challenge by integrating rich multi-modal data into a unified joint embedding space. This space includes textual, visual, acoustic, and graph-based modality features. By doing so, we mirror cognitive processes associated with music interaction and overcome the disjoint nature of individual modalities. The resulting joint low-dimensional vector space facilitates retrieval, clustering, embedding space arithmetic, and cross-modal retrieval tasks. Importantly, our approach carries implications for music information retrieval and recommendation systems. Furthermore, we propose a novel multi-modal model that integrates various data types—text, images, graphs, and audio—for music representation learning. Our model aims to capture the complex relationships between different modalities, enhancing the overall understanding of music. By combining textual descriptions, visual imagery, graph-based structures, and audio signals, we create a comprehensive representation that can be leveraged for a wide range of music-related tasks. Notably, our model demonstrates promising results in music classification and recommendation systems.

# Try Online
The demo is hosted at [https://fairouz.streamlit.app/](https://fairouz.streamlit.app/)

It is running the same code as in this repo, but hosted online.

# Running Locally
To run the demo locally, you can either use a virtual environment or docker.
To run using the virtual environment, you need to have Python installed.
For ease of access, we provide scripts to run the demo using either method in the `scripts` directory.
For Linux, you may need to give the scripts execution permissions using `chmod +x <script_name>.sh`.
For Windows, run the `.bat` files, and for Linux, run the `.sh` files.

## venv
1. Clone the repo
```bash
git clone https://github.com/a-tabaza/fairouz_demo.git
cd fairouz_demo
```

2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install the requirements
```bash
pip install -r requirements.txt
```

4. Run the demo
```bash
streamlit run gui.py
```

Access the demo at [http://localhost:8501/](http://localhost:8501/)

## Docker (Recommended)
1. Clone the repo
```bash
git clone https://github.com/a-tabaza/fairouz_demo.git
cd fairouz_demo
```

2. Build the docker image
```bash
docker build -t fairouz-demo .
```

3. Run the docker container
```bash
docker run -d -p 8501:8501 fairouz-demo
```

Access the demo at [http://localhost:8501/](http://localhost:8501/)

# Citation
```
@inproceedings{10.1145/3660853.3660886,
author = {Tabaza, Abdulrahman and Quishawi, Omar and Yaghi, Abdelrahman and Qawasmeh, Omar},
title = {Binding Text, Images, Graphs, and Audio for Music Representation Learning},
year = {2024},
isbn = {9798400716928},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3660853.3660886},
doi = {10.1145/3660853.3660886},
abstract = {Abstract In the field of Information Retrieval and Natural Language Processing, text embeddings play a significant role in tasks such as classification, clustering, and topic modeling. However, extending these embeddings to abstract concepts such as music, which involves multiple modalities, presents a unique challenge. Our work addresses this challenge by integrating rich multi-modal data into a unified joint embedding space. This space includes: (1) textual, (2) visual, (3) acoustic, and (4) graph-based modality features. By doing so, we mirror cognitive processes associated with music interaction and overcome the disjoint nature of individual modalities. The resulting joint low-dimensional vector space facilitates retrieval, clustering, embedding space arithmetic, and cross-modal retrieval tasks. Importantly, our approach carries implications for music information retrieval and recommendation systems. Furthermore, we propose a novel multi-modal model that integrates various data types—text, images, graphs, and audio—for music representation learning. Our model aims to capture the complex relationships between different modalities, enhancing the overall understanding of music. By combining textual descriptions, visual imagery, graph-based structures, and audio signals, we create a comprehensive representation that can be leveraged for a wide range of music-related tasks. Notably, our model demonstrates promising results in music classification, and recommendation systems. Code Availability: The source code for the multi-modal music representation model described in this paper is available on GitHub. Access and further details can be found at the following repository link: //github.com/a-tabaza/binding_music/},
booktitle = {Proceedings of the Cognitive Models and Artificial Intelligence Conference},
pages = {139–146},
numpages = {8},
location = {undefinedstanbul, Turkiye},
series = {AICCONF '24}
}
```
