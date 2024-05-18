# Binding Text, Images, Graphs, and Audio for Music Representation Learning
This repo contains the code for a demo showcasing the model from our paper, `Binding Text, Images, Graphs, and Audio for Music Representation Learning`.

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
