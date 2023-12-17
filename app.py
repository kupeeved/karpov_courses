from typing import List
import os
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI

from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger

from schema import PostGet


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("C:/Users/79289/PycharmProjects/pythonProject/catboost_model2")
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model


def batch_load_sql(query: str):
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=100000):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {len(chunk_dataframe)}")
    conn.close()
    return pd.concat(chunks, ignore_index=True)





def load_features() -> pd.DataFrame:
    logger.info("loading liked posts")
    liked_posts_query = """SELECT DISTINCT post_id, user_id 
                           FROM public.feed_data 
                           WHERE action='like'"""
    liked_posts = batch_load_sql(liked_posts_query)
    # фичи по постам на основе tf-idf
    logger.info("loading posts features")
    post_features = pd.read_sql("""SELECT * FROM public.posts_info_features""",
                                con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
                                    "postgres.lab.karpov.courses:6432/startml"
                                )
    # фичи по юзерам
    logger.info("loading user features")
    user_features = pd.read_sql("""SELECT * FROM public.user_data""",
                                con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
                                    "postgres.lab.karpov.courses:6432/startml"
                                )
    return [liked_posts, post_features, user_features]

logger.info("loading model")
model = load_models()
logger.info("loading features")
features = load_features()
logger.info("service is up and running")

app = FastAPI()

def get_recommended_feed(id: int, time: datetime, limit: int):
    logger.info(f"user_id:{id}")
    logger.info("reading features")
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    logger.info("dropping columns")
    posts_features = features[1].drop(features[1].columns[[0, 2]], axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    logger.info("zipping everything")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))

    logger.info("assigning everything")
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    logger.info("add time info")
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    logger.info("predicting")
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    logger.info("deleting liked posts")
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = filtered_.sort_values(by='predicts', ascending=False)
    recommended_posts = recommended_posts.iloc[:5]
    recommended_posts = recommended_posts.index

    return [PostGet(**{"id": i, "text": content[content.post_id == i].text.values[0],
                       "topic": content[content.post_id == i].topic.values[0]
                       }) for i in recommended_posts
            ]


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)
