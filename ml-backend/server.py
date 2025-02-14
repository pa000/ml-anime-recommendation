from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import pandas as pd
import logging
from dataclasses import dataclass, asdict
import json
import numpy as np
import sklearn.metrics

from typing import Dict, Any

@dataclass
class Anime:
    id: int
    name: str
    image_url: str
    genres: str
    year: int

@dataclass
class Genre:
    id: int
    name: str

@dataclass
class OperationResult:
    id: int
    message: str


POPULARITY_THRESHOLD = 6000
ANIME_CSV = 'dataset/anime-dataset-2023.csv'

# importing data from csv to pandas
def from_csv(file: str) -> pd.DataFrame:
    return pd.read_csv(file)


pd.options.display.max_colwidth = 300
logging.basicConfig(level=logging.DEBUG)


anime_df = from_csv(ANIME_CSV)

genre_df = pd.read_csv("animes.tsv", sep="\t", names=["name", "animes"])['name'].reset_index().rename(columns={'index': 'genre_id'})

class RatingGenerator:
    def __init__(self):
        self.users_count = None
        self.columns = ["user", "anime"]
        self.rankings = dict()
        self.anime_by_genre = self.get_anime_by_genre()

    def get_anime_by_genre(self):
        anime_df = pd.read_csv(ANIME_CSV)
        return (anime_df
            .assign(genre=lambda x: x['Genres'].apply(lambda x: list(map(str.strip, x.split(",")))))
            .filter(items=['anime_id', 'genre'])
            .explode('genre')
            .groupby('genre')
            .agg(list))

    def get_artifacts(self) -> Dict[str, Dict[str, str]]:
        p = "results/emb__"
        files = ["labels", "vects_iter"]
        suf = [".out.entities", ".out.npy"]

        return {'users': {f: f"{p}{self.columns[0]}__{self.columns[1]}{suf[idx]}"
                            for idx, f in enumerate(files)},
                'genres': {f: f"{p}genre__anime{suf[idx]}"
                            for idx, f in enumerate(files)}}

    def get_dist_to_genre(self, genre: str):
            dists = []
            for anime_id in self.labels:
                dists.append(anime_id in self.anime_by_genre.loc[genre]['anime_id'])

            return np.array(dists).astype(float)

    def load_artifacts(self) -> None:
        artifacts = self.get_artifacts()
        with open(artifacts['users']['labels'], "r") as entities:
            self.labels = np.array([int(i) for i in json.load(entities) if i != ''])

        with open(artifacts['genres']['labels'], "r") as entities:
            glabels = np.array([int(i) for i in json.load(entities)])

        # Load results to numpy
        self.vects_iter = np.load(artifacts['users']['vects_iter'])
        self.gvects_iter = np.load(artifacts['genres']['vects_iter'])

        self.vects_iter = self.vects_iter[self.labels.argsort()]
        self.labels.sort()

        self.gvects_iter = self.gvects_iter[glabels.argsort()]
        glabels.sort()

        assert np.all(self.labels == glabels)


    def load_rankings(self, idx: int, preffered_genres: list[str]):
        real_id = np.where(self.labels == idx)[0][0]

        uv = self.vects_iter[real_id]
        udist = sklearn.metrics.pairwise.cosine_similarity(uv.reshape(1, -1),
                                                          self.vects_iter)

        gv = self.gvects_iter[real_id]
        gdist = sklearn.metrics.pairwise.cosine_similarity(gv.reshape(1, -1),
                                                           self.gvects_iter)

        if len(preffered_genres) == 0:
            dists_to_preffered_genres = udist[0]
        else:
            dists_to_preffered_genres = np.mean(
                [self.get_dist_to_genre(genre) for genre in preffered_genres],
                axis=0)

        dist = 0.88 * udist[0] + 0.1 * gdist[0] + 0.02 * dists_to_preffered_genres

        ranking = (-dist).argsort()

        self.rankings[idx] = self.labels[ranking[:15]]

    def add_to_custom_ranking(self, custom_ranking: Dict[int, int], idx: int) -> None:
        anime_ranking = self.rankings[idx]

        for anime in anime_ranking:
            if anime in custom_ranking:
                custom_ranking[anime] += 1
            else:
                custom_ranking[anime] = 1

    def predict(self, already_watched: list[int], preffered_genres: list[str] = []):

        self.load_artifacts()
        custom_ranking: Dict[int, int] = dict()

        for idx in already_watched:
            if idx not in self.rankings:
                self.load_rankings(idx, preffered_genres)

            self.add_to_custom_ranking(custom_ranking, idx)

        recommended = \
            {id: custom_ranking[id] for id in custom_ranking.keys()
             if id not in already_watched}

        return dict(sorted(recommended.items(),
                           reverse=True,
                           key=lambda x: x[1]))


recommendations_model = RatingGenerator()


def pandas_extract_content(row, label):
    name = row[label].to_string()
    return name.split("    ")[1]


def pandas_tuple_id(ptuple):
    return ptuple[0]


def extract_year(aired):
    try:
        return aired.split(",")[1].split(" ")[1]
    except IndexError:
        return 0


def english_name_exists(anime_row):
    return pandas_extract_content(anime_row, "English name") != "UNKNOWN"

def genre_filter(genre_id: int | None) -> pd.DataFrame:
    genre_df_local = genre_df
    if genre_id is not None:
        genre_df_local = genre_df_local[genre_df_local['genre_id'] == genre_id]

    return genre_df_local

def anime_filter(anime_id: int | None) -> pd.DataFrame:
    anime_df_local = anime_df
    if anime_id is not None:
        anime_df_local = anime_df_local[anime_df_local['anime_id'] == anime_id]

    return anime_df_local[(anime_df_local['Popularity'] > 0) &
                          (anime_df_local['Popularity'] <= POPULARITY_THRESHOLD)] # type: ignore


def is_anime_available(anime_id: int) -> bool:
    return len(anime_filter(anime_id)) > 0


def get_anime_dict(anime_id: int) -> Dict[str, Any]:
    anime_row: pd.DataFrame = anime_filter(anime_id)

    anime_row = anime_row.filter(items=["anime_id", "Name", "English name", "Genres", "Image URL", "Aired"])

    if len(anime_row) == 0:
        raise Exception("Anime not found!")

    anime = Anime(
        int(pandas_extract_content(anime_row, "anime_id")),
        pandas_extract_content(anime_row, "English name") if english_name_exists(anime_row) else pandas_extract_content(anime_row, "Name"),
        pandas_extract_content(anime_row, "Image URL"),
        pandas_extract_content(anime_row, "Genres"),
        int(extract_year(pandas_extract_content(anime_row, "Aired"))))

    return asdict(anime)

def get_genre_dict(genre_id: int) -> Dict[str, Any]:
    genre_row: pd.DataFrame = genre_filter(genre_id)

    if len(genre_row) == 0:
        raise Exception("Genre not found!")

    genre = Genre(
        int(pandas_extract_content(genre_row, "genre_id")),
        pandas_extract_content(genre_row, "name"))

    return asdict(genre)

def search_str(s: str, search: str) -> bool:
    return search.lower() in s.lower()

def find_anime_by_name(name: str) -> pd.DataFrame:
    mask = (anime_filter(None)
        .filter(items=["Name", "English name", "Studios"])
        .apply(lambda x: x.map(lambda s: search_str(s, name))))
    return anime_filter(None).loc[mask.any(axis=1)].sort_values("Popularity")

def find_genre_by_name(name: str) -> pd.DataFrame:
    mask = (genre_filter(None)
        .filter(items=["name"])
        .apply(lambda x: x.map(lambda s: search_str(s, name))))
    return genre_filter(None).loc[mask.any(axis=1)]

def search_animes_engine(phrase: str, selected_animes: list[int]) -> list[int]:
    return [pandas_tuple_id(i) for i in find_anime_by_name(phrase).itertuples(index=False)
            if pandas_tuple_id(i) not in selected_animes][:5]

def search_genres_engine(phrase: str, selected_genres: list[int]) -> list[int]:
    return [pandas_tuple_id(i) for i in find_genre_by_name(phrase).itertuples(index=False)
            if pandas_tuple_id(i) not in selected_genres][:5]

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


@app.route("/api/generate", methods=['POST'])
def generate_recommendations():

    data: Dict[str, list[Dict[str, Any]]] = request.json # type: ignore

    selected_animes = [anime['id'] for anime in data['animes']]
    selected_genres = [genre['name'] for genre in data['genres']]

    ranking = RatingGenerator().predict(selected_animes, selected_genres)
    recommended_animes = [i for i in ranking.keys() if is_anime_available(i)]

    return jsonify([get_anime_dict(i) for i in recommended_animes]), 200


@app.route("/api/Anime/<int:anime_id>", methods=['GET'])
def get_anime(anime_id: int):
    return jsonify(get_anime_dict(anime_id)), 200


@app.route("/api/search/anime/<string:phrase>", methods=['POST'])
def search_animes(phrase: str):

    data: list[Dict[str, Any]] = request.json # type: ignore
    selected_animes = [anime['id'] for anime in data]

    proposed_animes = search_animes_engine(phrase, selected_animes)

    return jsonify([get_anime_dict(i) for i in proposed_animes]), 200

@app.route("/api/Genre/<int:genre_id>", methods=['GET'])
def get_genre(genre_id: int):
    return jsonify(get_genre_dict(genre_id)), 200


@app.route("/api/search/genre/<string:phrase>", methods=['POST'])
def search_genres(phrase: str):
    data: list[Dict[str, Any]] = request.json # type: ignore
    selected_genres = [anime['id'] for anime in data]

    proposed_genres = search_genres_engine(phrase, selected_genres)

    return jsonify([get_genre_dict(i) for i in proposed_genres]), 200

if __name__ == '__main__':
    app.run()
