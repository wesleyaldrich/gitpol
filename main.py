import joblib
import pandas as pd
import requests
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

def fetch_github_data(username: str) -> dict:
    """
    Fetches raw data for a user from GitHub REST API.
    """
    import requests

    response = requests.get(f"https://api.github.com/users/{username}")

    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="GitHub user not found.")
    
    raw_data = response.json()
    # only takes some fields from the raw data
    raw_data = {
        'hirable': raw_data.get('hireable', False),
        'public_repos': raw_data.get('public_repos', 0),
        'blog': raw_data.get('blog', ''),
        'followers': raw_data.get('followers', 0),
        'location': raw_data.get('location', ''),
        'type': raw_data.get('type', ''),
        'bio': raw_data.get('bio', ''),
        'company': raw_data.get('company', ''),
        'public_gists': raw_data.get('public_gists', 0),
        'following': raw_data.get('following', 0),
    }

    return raw_data


def fetch_commit_count(username: str) -> int:
    # make another API call to get commit count
    import requests
    response = requests.get(f"https://api.github.com/search/commits?q=author:{username}")

    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Failed to get commit count.")
    
    commit_count = response.json().get('total_count', 0)
    
    return commit_count


def encode_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a pandas DataFrame and returns a DataFrame with categorical data encoded.
    """

    df['company'] = df['company'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['hirable'] = df['hirable'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['location'] = df['location'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['bio'] = df['bio'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['type'] = df['type'].apply(lambda x: 0 if x == 'Bot' else (1 if x == 'User' else 2))
    df['blog'] = df['blog'].apply(lambda x: 0 if pd.isnull(x) or x == '' else 1)
    
    return df


def create_features_from_data(data: dict, username: str) -> pd.DataFrame:
    """
    Takes raw data and converts it into the features for our model.
    """

    df = pd.DataFrame([data])

    df = encode_categorical_data(df)

    # additional columns
    df['commits'] = fetch_commit_count(username)
    df['profile_completeness'] = df[['company', 'hirable', 'location', 'bio', 'blog']].sum(axis=1)
    df['commit_to_public_repo'] = df['commits'] / (df['public_repos'] + 1)

    # make sure the order of features is correct
    final_model_columns = [
        'hirable', 'public_repos', 'blog', 'followers', 'location', 'type', 'bio', 'commits', 'company', 'public_gists', 'following', 'profile_completeness', 'commit_to_public_repo'
    ]

    df = df.reindex(columns=final_model_columns, fill_value=0)

    return df

def get_starred_repo_count(username, token=os.getenv("GITHUB_TOKEN")):

    url = "https://api.github.com/graphql"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    query = """
    query ($login: String!) {
      user(login: $login) {
        starredRepositories {
          totalCount
        }
      }
    }
    """

    variables = {
        "login": username
    }

    response = requests.post(url, json={"query": query, "variables": variables}, headers=headers)

    if response.status_code != 200:
        print(f"HTTP error for {username}: {response.status_code}")
        return None

    result = response.json()

    # Check for GraphQL-level errors
    if "errors" in result:
        print(f"GraphQL error for {username}: {result['errors'][0].get('message')}")
        return None

    user_data = result.get("data", {}).get("user")

    if user_data is None:
        print(f"No user data for {username}")
        return None

    return user_data["starredRepositories"]["totalCount"]



# --- FastAPI Application ---

app = FastAPI(
    title="GitPol API",
    description="An API to predict if a GitHub user is potentially malicious."
)

try:
    model = joblib.load("model/gitpol_model_v3_xgboost_random_undersampling.pkl")
except FileNotFoundError:
    model = None

class UserRequest(BaseModel):
    username: str

@app.post("/sus")
async def check_suspicion(request: UserRequest):

    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load on the server.")

    # --- The Prediction Pipeline ---
    raw_data = fetch_github_data(request.username)

    features_df = create_features_from_data(raw_data, request.username)

    prediction = model.predict(features_df)
    prediction_proba = model.predict_proba(features_df)

    is_suspicious = bool(prediction[0])
    suspicion_score = float(prediction_proba[0][1])

    if is_suspicious:
        starred_count = get_starred_repo_count(request.username)
        commits = features_df['commits'].values[0]

        if commits > starred_count:
            is_suspicious = False
            suspicion_score = 0.0
            print(f"User {request.username} has more commits ({commits}) than starred repositories ({starred_count}). Marking as not suspicious.")
    
    return {
        "username": request.username,
        "is_suspicious": is_suspicious,
        "suspicion_score": f"{suspicion_score:.4f}",
        "status": "success"
    }

@app.post("/sus/test")
async def check_suspicion():

    final_features_mock = {
        'hirable': 0,
        'public_repos': 9999,
        'blog': 0,
        'followers': 0,
        'location': 0,
        'type': 1,
        'bio': 0,
        'commits': 1,
        'company': 0,
        'public_gists': 0,
        'following': 0,
        'profile_completeness': 0,
        'commit_to_public_repo': 0.0001 # commit / (public_repos + 1) 
    }

    features_df = pd.DataFrame([final_features_mock])

    prediction = model.predict(features_df)
    prediction_proba = model.predict_proba(features_df)

    # Extract the prediction and the probability for the "True" class
    is_suspicious = bool(prediction[0])
    suspicion_score = float(prediction_proba[0][1])

    return {
        "username": "super_bot",
        "is_suspicious": is_suspicious,
        "suspicion_score": f"{suspicion_score:.4f}",
        "status": "success"
    }

# uvicorn main:app --reload