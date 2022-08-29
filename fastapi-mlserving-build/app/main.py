import joblib
import re
import os

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from fastapi import FastAPI

from elasticapm.contrib.starlette import make_apm_client, ElasticAPM

agent_service_name = os.getenv("AGENT_SERVICE_NAME", default="spam-classifier")
apm_server_protocol = os.getenv("APM_SERVER_PROTO", default="http")
apm_server_url = os.getenv("APM_SERVER_URL", default="apm-server")

apm_config = {
    "SERVICE_NAME": f"{agent_service_name}",
    "SERVER_URL": f"{apm_server_protocol}://{apm_server_url}:8200",
    "ENVIRONMENT": "dev"
}

apm = make_apm_client(apm_config)
app = FastAPI()
app.add_middleware(ElasticAPM, client=apm)

model = joblib.load('spam_classifier.joblib')


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


def classify_message(model, message):

    message = preprocessor(message)
    label = model.predict([message])[0]
    spam_prob = model.predict_proba([message])

    return {'label': label, 'spam_probability': spam_prob[0][1]}


@app.get('/')
def get_root():
    return {'message': 'Welcome to the spam detection API'}


@app.get('/spam_detection_query/')
async def detect_spam_query(message: str):
    return classify_message(model, message)
    # e.g. http://localhost:8000/spam_detection_query/?message=attention:%20claim%20your%20free%20prize


@app.get('/spam_detection_query/{message}')
async def detect_spam_path(message: str):
    return classify_message(model, message)
    # e.g. http://localhost:8000/spam_detection_query/claim%20your%20free%20prize
