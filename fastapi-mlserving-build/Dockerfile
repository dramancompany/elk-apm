FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY ./app /app

WORKDIR /app

RUN pip install -r requirements.txt

ENV APM_SERVER_URL=172.31.9.105 \   ### onboarding private ip : 172.31.9.105
    APM_SERVER_PROTO=http
