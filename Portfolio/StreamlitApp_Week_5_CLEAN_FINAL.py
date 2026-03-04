
import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
import boto3
import sagemaker
import tarfile
import joblib

from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

warnings.simplefilter("ignore")

# ----------------------------
# Import feature utilities
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import get_bitcoin_historical_prices


# ----------------------------
# AWS Secrets
# ----------------------------
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint_bitcoin = st.secrets["aws_credentials"]["AWS_ENDPOINT"]


# ----------------------------
# AWS Session
# ----------------------------
@st.cache_resource
def get_session():
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session = get_session()
sm_session = sagemaker.Session(boto_session=session)


# ----------------------------
# Load Bitcoin Data
# ----------------------------
df_prices = get_bitcoin_historical_prices()

MIN_VAL = float(0.5 * df_prices.iloc[:,0].min())
MAX_VAL = float(2.0 * df_prices.iloc[:,0].max())
DEFAULT_VAL = float(df_prices.iloc[:,0].mean())


MODEL_INFO = {
    "endpoint": aws_endpoint_bitcoin,
    "pipeline": "finalized_bitcoin_model.tar.gz",
    "keys": ["Close Price"],
    "inputs": [
        {
            "name": "Close Price",
            "min": MIN_VAL,
            "max": MAX_VAL,
            "default": DEFAULT_VAL,
            "step": 100.0
        }
    ]
}


# ----------------------------
# Load Pipeline
# ----------------------------
@st.cache_resource
def load_pipeline(bucket, prefix):

    s3 = session.client("s3")
    filename = MODEL_INFO["pipeline"]

    s3.download_file(
        Bucket=bucket,
        Key=f"{prefix}/{filename}",
        Filename=filename
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
        joblib_file = [f for f in tar.getnames() if f.endswith(".joblib")][0]

    return joblib.load(joblib_file)


# ----------------------------
# Safe Preprocessing
# ----------------------------
def safe_preprocess(pipeline, X):

    Xt = X

    for name, step in pipeline.steps[:-1]:

        if hasattr(step, "transform"):
            Xt = step.transform(Xt)

        elif hasattr(step, "fit_resample"):
            # skip samplers like SMOTE
            continue

    return Xt


# ----------------------------
# SageMaker Prediction
# ----------------------------
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

        mapping = {-1:"SELL",0:"HOLD",1:"BUY"}

        return mapping.get(pred_val,pred_val),200

    except Exception as e:

        return str(e),500


# ----------------------------
# SHAP Explanation
# ----------------------------
def display_explanation(input_df):

    full_pipeline = load_pipeline(aws_bucket,"sklearn-pipeline-deployment")

    model = full_pipeline.steps[-1][1]

    X_transformed = safe_preprocess(full_pipeline,input_df)

    X_array = np.array(X_transformed)

    x_one = X_array[-1:].copy()

    explainer = shap.Explainer(model,X_array[:200])

    shap_values = explainer(x_one)

    st.subheader("🔍 Decision Transparency (SHAP)")

    pred_class = np.argmax(model.predict_proba(x_one))

    fig, ax = plt.subplots(figsize=(10,4))

    shap.plots.waterfall(
        shap_values[0,:,pred_class],
        max_display=10,
        show=False
    )

    st.pyplot(fig)

    shap_vals = shap_values.values[0,:,pred_class]

    feature_names = list(input_df.columns)

    top_feature = feature_names[np.argmax(np.abs(shap_vals))]

    st.info(
        f"Most influential feature: **{top_feature}**"
    )


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Bitcoin Predictor", layout="wide")

st.title("₿ Bitcoin Buy / Hold / Sell Predictor")


with st.form("prediction_form"):

    st.subheader("Inputs")

    user_inputs = {}

    for inp in MODEL_INFO["inputs"]:

        user_inputs[inp["name"]] = st.number_input(
            inp["name"],
            min_value=inp["min"],
            max_value=inp["max"],
            value=inp["default"],
            step=inp["step"]
        )

    submitted = st.form_submit_button("Run Prediction")


if submitted:

    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

    base_df = df_prices.copy()

    input_df = pd.concat(
        [base_df,pd.DataFrame([data_row],columns=base_df.columns)]
    )

    result,status = call_model_api(input_df)

    if status == 200:

        st.metric("Prediction",result)

        display_explanation(input_df)

    else:

        st.error(result)
