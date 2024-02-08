from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import pandas as pd
import logging
from dataclasses import dataclass, asdict
import json
import numpy as np
import sklearn.metrics


@dataclass
class Anime:
    id: int
    name: str
    image_url: str
    genres: str
    year: int


@dataclass
class OperationResult:
    id: int
    message: str


# importing data from csv to pandas
def from_csv(file: str) -> pd.DataFrame:
    return pd.read_csv(file)


pd.options.display.max_colwidth = 300
logging.basicConfig(level=logging.DEBUG)

anime_data = 'dataset/anime-dataset-2023.csv'
anime_df = from_csv(anime_data)
example_animes = set()
recommended_animes = []


class RatingGenerator:
    def __init__(self):
        self.users_count = None
        self.columns = ["user", "anime"]
        self.rankings = dict()

    def get_artifacts(self):
        p = "results/emb__"
        files = ["labels", "vects_iter"]
        suf = [".out.entities", ".out.npy"]

        return {f: f"{p}{self.columns[0]}__{self.columns[1]}{suf[idx]}"
                for idx, f in enumerate(files)}

    def load_artifacts(self):
        artifacts = self.get_artifacts()
        with open(artifacts['labels'], "r") as entities:
            self.labels = np.array([int(i) for i in json.load(entities)])
        # Load results to numpy
        self.vects_iter = np.load(artifacts['vects_iter'])

    def load_rankings(self, idx: int):
        real_id = np.where(self.labels == idx)[0][0]

        v = self.vects_iter[real_id]
        dist = sklearn.metrics.pairwise.cosine_similarity(v.reshape(1, -1),
                                                          self.vects_iter,
                                                          dense_output=True)
        ranking = (-dist).argsort()[0]

        self.rankings[self.labels[real_id]] = self.labels[ranking[:15]]

    def add_to_custom_ranking(self, custom_ranking, idx: int):
        anime_ranking = self.rankings[idx]

        for anime in anime_ranking:
            if anime in custom_ranking:
                custom_ranking[anime] += 1
            else:
                custom_ranking[anime] = 1

    def predict(self, already_watched):

        self.load_artifacts()
        custom_ranking = dict()

        for idx in already_watched:
            if idx not in self.rankings:
                self.load_rankings(idx)

            self.add_to_custom_ranking(custom_ranking, idx)

        return dict(sorted(custom_ranking.items(),
                           reverse=True,
                           key=lambda x: x[1]))


recommendations_model = RatingGenerator()


def pandas_extract_content(row, label):
    name = row[label].to_string()
    return name.split("    ")[1]


def extract_year(aired):
    try:
        return aired.split(",")[1].split(" ")[1]
    except IndexError:
        return 0


def is_anime_available(anime_id):
    return len(anime_df[anime_df.anime_id == anime_id]) > 0


def get_anime_dict(anime_id: int):
    anime_row = anime_df[anime_df.anime_id == anime_id] \
        .filter(items=["anime_id", "Name", "Genres", "Image URL", "Aired"])

    if len(anime_row) == 0:
        raise Exception("Anime not found!")

    anime = Anime(
        int(pandas_extract_content(anime_row, "anime_id")),
        pandas_extract_content(anime_row, "Name"),
        pandas_extract_content(anime_row, "Image URL"),
        pandas_extract_content(anime_row, "Genres"),
        int(extract_year(pandas_extract_content(anime_row, "Aired"))))

    return asdict(anime)


def startup():
    global all_data
    pass


class AnimeApp(Flask):
    def run(self,
            host=None,
            port=None,
            debug=None,
            load_dotenv=True,
            **options):
        with self.app_context():
            startup()
        super(AnimeApp, self).run(
                    host=host,
                    port=port,
                    debug=debug,
                    load_dotenv=load_dotenv,
                    **options
            )


app = AnimeApp(__name__)
CORS(app)


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'

        return response


@app.route("/")
def hello_world():
    return "<h1>Anime data server</h1>"


@app.route("/api/generate", methods=['GET'])
def generate_recommendations():
    global recommended_animes

    ranking = recommendations_model.predict(example_animes)
    recommended_animes = [i for i in ranking.keys() if is_anime_available(i)]

    return Response(), 200


@app.route("/api/Anime/<int:anime_id>", methods=['GET'])
def get_anime(anime_id: int):
    return jsonify(get_anime_dict(anime_id)), 200


@app.route("/api/Anime/<int:anime_id>", methods=['DELETE'])
def delete_anime(anime_id: int):
    if anime_id in example_animes:
        example_animes.remove(anime_id)
        return jsonify(asdict(
            OperationResult(anime_id, "Ok"))), 200
    else:
        return jsonify(asdict(
            OperationResult(anime_id, "Anime not selected"))), 200


@app.route("/api/Anime/<int:anime_id>", methods=['POST'])
def put_anime(anime_id: int):
    try:
        get_anime_dict(anime_id)
        
        example_animes.add(anime_id)
        return jsonify(asdict(OperationResult(anime_id, "Ok"))), 200
    except Exception:
        app.logger.warn("Anime not found")
        return jsonify(asdict(
            OperationResult(anime_id, "Anime not found"))
                       ), 200


@app.route("/api/Animes/selected")
def get_animes_selected():
    return jsonify([get_anime_dict(i) for i in example_animes]), 200


@app.route("/api/Animes/recommended")
def get_animes_recommended():
    return jsonify([get_anime_dict(i) for i in recommended_animes]), 200


if __name__ == '__main__':
    app.run()
