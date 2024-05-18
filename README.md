# Binding Text, Images, Graphs, and Audio for Music Representation Learning
This repo contains the code for a demo showcasing the model from our paper, `Binding Text, Images, Graphs, and Audio for Music Representation Learning`.

# Web Demo
The web demo is hosted at [https://fairouz-demo.herokuapp.com/](https://fairouz-demo.herokuapp.com/)

# Running Locally (venv)
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

# Running Locally (Docker)
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

Access the web demo at [http://localhost:8501/](http://localhost:8501/)
