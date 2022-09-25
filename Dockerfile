# STAGE 1 docker : machine learning stuff
FROM  python:3.10.5-buster AS pipeline

RUN mkdir dvc_pipeline


RUN mkdir dvc_pipeline/src
COPY src/ dvc_pipeline/src/

RUN mkdir dvc_pipeline/data
# COPY data/raw/text_emotion.csv dvc_pipeline/data/raw/text_emotion.csv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY params.yaml dvc_pipeline/params.yaml
COPY dvc.yaml dvc_pipeline/dvc.yaml


WORKDIR /dvc_pipeline
RUN dvc init --no-scm
RUN dvc repro 

# STAGE 2 docker : web application
FROM python:3.10.5-buster
# FROM python:alpine3.10
# FROM nacyot/fortran-gfortran:apt

RUN pip install --upgrade pip
# RUN apk update
# RUN apk add make automake gcc g++ subversion python3-dev
 
RUN mkdir web_app 

#COPY --from=pipeline /dvc_pipeline/data/raw/raw_scope.csv web_app/raw_scope.csv
COPY --from=pipeline /dvc_pipeline/params.yaml web_app/params.yaml 
COPY --from=pipeline /dvc_pipeline/src/streamingdata.py web_app/streamingdata.py

#COPY --from=pipeline /dvc_pipeline/config.ini web_app/config.ini
COPY --from=pipeline /dvc_pipeline/src/cleanscopedata.py web_app/cleanscopedata.py
COPY --from=pipeline /dvc_pipeline/src/app.py web_app/app.py
COPY --from=pipeline /dvc_pipeline/polarityclassifier.pkl web_app/polarityclassifier.pkl
COPY --from=pipeline /dvc_pipeline/sentimentclassifier.pkl web_app/sentimentclassifier.pkl



COPY requirements.txt .
RUN pip install -r requirements.txt
#RUN pip install flask
#RUN pip install numpy==1.23.0
#RUN pip install scipy
#RUN pip install sklearn

WORKDIR /web_app

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


#CMD ["python","app.py"]
