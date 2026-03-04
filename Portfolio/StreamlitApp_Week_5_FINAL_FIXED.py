
import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import joblib
import tarfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

import shap

# ----------------------------
# Setup
# ----------------------------
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import get_bitcoin_historical_prices

# ----------------------------
# Secrets
# ----------------------------
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint_bitcoin = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ----------------------------
# AWS session
# ----------------------------
@st.cache_resource
def get_session(_aws_id, _aws_secret, _aws_token):
    return boto3.Session(
        aws_access_key_id=_aws_id,
        aws_secret_access_key=_aws_secret,
        aws_session_token=_aws_token,
        region_name="us-east-1"
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ----------------------------
# Data
# ----------------------------
df_prices = get_bitcoin_historical_prices()

# Basic bounds for UI input
MIN_VAL = float(0.5 * df_prices.iloc[:, 0].min())
MAX_VAL = float(2.0 * df_prices.iloc[:, 0].max())
DEFAULT_VAL = float(df_prices.iloc[:, 0].mean())

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

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_pipeline(_session, bucket, s3_prefix):
    """
    Downloads the tar.gz pipeline artifact from S3, extracts the .joblib, and loads it.
    """
    s3_client = _session.client("s3")
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{s3_prefix}/{os.path.basename(filename)}"
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith(".joblib")]
        if not joblib_file:
            raise FileNotFoundError("No .joblib found inside the model tar.gz")
        joblib_file = joblib_file[0]

    return joblib.load(joblib_file)


def safe_preprocess(pipeline, X):
    """
    Applies only transform-capable steps in order.
    Skips resamplers (fit_resample) and any estimator-like steps that don't transform.

    This avoids Streamlit crash:
    AttributeError: This 'Pipeline' has no attribute 'transform'
    """
    Xt = X
    # Some pipelines are imblearn.Pipeline or sklearn.Pipeline — both expose .steps
    for name, step in getattr(pipeline, "steps", [])[:-1]:
        if hasattr(step, "transform"):
            Xt = step.transform(Xt)
        elif hasattr(step, "fit_resample"):
            # SMOTE / sampling used only during training; skip at inference/explanation time
            continue
        else:
            # Unknown step type: ignore safely
            continue
    return Xt


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


def display_explanation(input_df, _session, bucket):
    """
    SHAP explanation built dynamically from the loaded pipeline (no saved explainer),
    and with robust preprocessing that doesn't assume .transform exists on the truncated pipeline.
    """
    full_pipeline = load_pipeline(_session, bucket, "sklearn-pipeline-deployment")
    model = getattr(full_pipeline, "steps", [])[-1][1]

    # Preprocess safely (transform steps only; skip samplers)
    X_transformed = safe_preprocess(full_pipeline, input_df)

    # For performance, use a small background sample
    if isinstance(X_transformed, pd.DataFrame):
        bg = X_transformed.sample(n=min(200, len(X_transformed)), random_state=0)
        x_one = X_transformed.tail(1)
        feature_names = list(X_transformed.columns)
        x_one_array = x_one.values
    else:
        X_arr = np.asarray(X_transformed)
        bg = X_arr[: min(200, len(X_arr))]
        x_one_array = X_arr[-1:].copy()
        feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    explainer = shap.Explainer(model, bg)
    shap_values = explainer(x_one_array)

    # Build a single-row Explanation for waterfall
    sv = shap_values.values[0]
    bv = shap_values.base_values[0] if np.ndim(shap_values.base_values) > 0 else shap_values.base_values

    exp = shap.Explanation(
        values=sv,
        base_values=bv,
        data=x_one_array[0],
        feature_names=feature_names
    )

    
    
    st.subheader("🔍 Decision Transparency (SHAP)")

    # choose explanation for predicted class (multiclass safe)
    pred_class = np.argmax(model.predict_proba(x_one_array))

    fig, ax = plt.subplots(figsize=(10,4))
    shap.plots.waterfall(
        shap_values[0,:,pred_class],
        max_display=12,
        show=False
    )
    st.pyplot(fig)

    # determine most influential feature
    shap_vals = shap_values.values[0,:,pred_class]
    top_feature = feature_names[np.argmax(np.abs(shap_vals))]

 pd.Series(exp.values, index=exp.feature_names).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Bitcoin ML Predictor", layout="wide")
st.title("₿ Bitcoin Buy / Hold / Sell Predictor")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " ").upper(),
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"]),
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

    # Append the user's price to the historical prices (keeps model expectations consistent)
    base_df = df_prices.copy()
    input_df = pd.concat([base_df, pd.DataFrame([data_row], columns=base_df.columns)], axis=0)

    res, status = call_model_api(input_df)

    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)
