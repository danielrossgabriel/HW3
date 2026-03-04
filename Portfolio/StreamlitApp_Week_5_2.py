
import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from imblearn.pipeline import Pipeline

import shap

# Setup & Path Configuration
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import get_bitcoin_historical_prices

# Access secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint_bitcoin = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data
df_prices = get_bitcoin_historical_prices()

MIN_VAL = 0.5 * df_prices.iloc[:, 0].min()
MAX_VAL = 2.0 * df_prices.iloc[:, 0].max()
DEFAULT_VAL = df_prices.iloc[:, 0].mean()

MODEL_INFO = {
    "endpoint": aws_endpoint_bitcoin,
    "pipeline": "finalized_bitcoin_model.tar.gz",
    "keys": ["Close Price"],
    "inputs": [{
        "name": "Close Price",
        "type": "number",
        "min": MIN_VAL,
        "max": MAX_VAL,
        "default": DEFAULT_VAL,
        "step": 100.0
    }]
}

def load_pipeline(_session, bucket, key):

    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(joblib_file)


# Prediction
def call_model_api(input_df):

    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]

        mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}

        return mapping.get(pred_val, pred_val), 200

    except Exception as e:
        return f"Error: {str(e)}", 500


# SHAP explanation (fixed version)
def display_explanation(input_df, session, aws_bucket):

    full_pipeline = load_pipeline(session, aws_bucket, "sklearn-pipeline-deployment")

    preprocessing_pipeline = Pipeline(steps=full_pipeline.steps[:-1])
    model = full_pipeline.steps[-1][1]

    X_transformed = preprocessing_pipeline.transform(input_df)

    explainer = shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed)

    feature_names = preprocessing_pipeline.get_feature_names_out()

    exp = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_transformed[0],
        feature_names=feature_names
    )

    st.subheader("🔍 Decision Transparency (SHAP)")

    fig, ax = plt.subplots(figsize=(10,4))
    shap.plots.waterfall(exp)
    st.pyplot(fig)

    top_feature = pd.Series(exp.values, index=exp.feature_names).abs().idxmax()

    st.info(
        f"**Business Insight:** The most influential factor in this decision was **{top_feature}**."
    )


# Streamlit UI
st.set_page_config(page_title="Bitcoin ML Predictor", layout="wide")
st.title("₿ Bitcoin Buy / Hold / Sell Predictor")

with st.form("pred_form"):

    st.subheader("Inputs")

    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace('_', ' ').upper(),
                min_value=inp["min"],
                max_value=inp["max"],
                value=inp["default"],
                step=inp["step"]
            )

    submitted = st.form_submit_button("Run Prediction")


if submitted:

    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

    base_df = df_prices

    input_df = pd.concat([
        base_df,
        pd.DataFrame([data_row], columns=base_df.columns)
    ])

    res, status = call_model_api(input_df)

    if status == 200:

        st.metric("Prediction Result", res)

        display_explanation(input_df, session, aws_bucket)

    else:

        st.error(res)
