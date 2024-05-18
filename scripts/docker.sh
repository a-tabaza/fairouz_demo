git clone https://github.com/a-tabaza/fairouz_demo.git
cd fairouz_demo
docker build -t fairouz-demo .
docker run -d -p 8501:8501 fairouz-demo