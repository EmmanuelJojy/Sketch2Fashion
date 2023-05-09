FROM python:slim
WORKDIR /app

RUN apt-get update && apt-get install -y git
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install streamlit

RUN git clone https://github.com/streamlit/streamlit-example.git docker .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]