from __future__ import annotations

import pickle
from dataclasses import dataclass
from heapq import nlargest
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix
from surprise import SVD

from allergen_detector import get_allergen_details

EXPLICIT_MODEL_PATH = Path("model/explicit_recommender_model.pkl")
IMPLICIT_MODEL_PATH = Path("model/implicit_recommender_model.pkl")
EXPLICIT_FEEDBACK_PATH = Path("data_mining/data/feedback_explicit.csv")
IMPLICIT_FEEDBACK_PATH = Path("data_mining/data/feedback_implicit.csv")
USERS_PATH = Path("data_mining/data/users.csv")
PRODUCT_FEATURES_DIR = Path("data_mining/data/product_features")

EXPLICIT_MAP_AT_10 = 0.95
IMPLICIT_MAP_AT_10 = 0.15

_TOTAL_MAP = EXPLICIT_MAP_AT_10 + IMPLICIT_MAP_AT_10
EXPLICIT_WEIGHT = EXPLICIT_MAP_AT_10 / _TOTAL_MAP if _TOTAL_MAP else 0.5
IMPLICIT_WEIGHT = IMPLICIT_MAP_AT_10 / _TOTAL_MAP if _TOTAL_MAP else 0.5

EVENT_STRENGTH = {
    "view": 1.0,
    "query_search": 2.0,
    "click": 3.0,
    "compare": 4.0,
}


@dataclass(frozen=True)
class UserSummary:
    """Basic information about an available user."""

    user_id: str
    display_name: str

@dataclass(frozen=True)
class RecommendationResult:
    """Hybrid recommendation with supporting metadata."""

    product_code: str
    combined_score: float
    explicit_score: float | None
    implicit_score: float | None
    product_name: str | None = None
    brands: str | None = None
    categories: str | None = None
    allergens: tuple[str, ...] = ()


class HybridRecommender:
    """Combine explicit and implicit recommenders using MAP@10 weights."""

    def __init__(
        self,
        *,
        explicit_model_path: Path = EXPLICIT_MODEL_PATH,
        implicit_model_path: Path = IMPLICIT_MODEL_PATH,
        explicit_feedback_path: Path = EXPLICIT_FEEDBACK_PATH,
        implicit_feedback_path: Path = IMPLICIT_FEEDBACK_PATH,
        users_path: Path = USERS_PATH,
        product_features_dir: Path = PRODUCT_FEATURES_DIR,
    ) -> None:
        self.explicit_model = self._load_explicit_model(explicit_model_path)
        self.implicit_model = self._load_implicit_model(implicit_model_path)

        self._explicit_known_users = self._collect_explicit_users(explicit_feedback_path)
        (
            self._implicit_user_items,
            self._implicit_user_lookup,
            self._implicit_item_lookup,
        ) = self._build_implicit_structures(implicit_feedback_path)

        self._user_display_names = self._load_user_display_names(users_path)
        self._available_users = self._collect_available_users()

        self._product_catalog = self._load_product_catalog(product_features_dir)

    def list_users(self) -> list[UserSummary]:
        """Return the list of users with interactions in either model."""

        users: list[UserSummary] = []
        for user_id in self._available_users:
            display_name = self._user_display_names.get(user_id, user_id)
            users.append(UserSummary(user_id, display_name))
        return users

    def recommend(self, user_id: str, top_k: int = 10) -> list[RecommendationResult]:
        """Return top-N hybrid recommendations for the requested user."""

        if top_k <= 0:
            return []

        candidate_pool = max(top_k * 5, 10000)

        explicit_scores = self._explicit_scores(user_id, candidate_pool)
        implicit_scores = self._implicit_scores(user_id, candidate_pool)

        if not explicit_scores and not implicit_scores:
            return []

        #explicit_norm = self._normalise_scores(explicit_scores)
        #implicit_norm = self._normalise_scores(implicit_scores)

        candidate_rows: list[tuple[str, float, float | None, float | None, str | None, str | None, str | None]] = []
        all_codes = set(explicit_scores) | set(implicit_scores)
        for code in all_codes:
            combined = (
                explicit_scores.get(code, 0.0) / 5 * EXPLICIT_WEIGHT
                + implicit_scores.get(code, 0.0) * IMPLICIT_WEIGHT
            )
            info = self._product_catalog.get(code, {})
            candidate_rows.append(
                (
                    code,
                    combined,
                    explicit_scores.get(code, 0.0) / 5,
                    implicit_scores.get(code),
                    info.get("product_name"),
                    info.get("brands"),
                    info.get("categories"),
                )
            )

        candidate_rows.sort(key=lambda rec: rec[1], reverse=True)
        top_candidates = candidate_rows[:top_k]

        recommendations: list[RecommendationResult] = []
        for (
            code,
            combined,
            explicit_score,
            implicit_score,
            product_name,
            brands,
            categories,
        ) in top_candidates:
            allergen_info = get_allergen_details(user_id, code)
            detected_allergens = allergen_info.get("detected_allergens") or ()
            allergens = tuple(str(a) for a in detected_allergens)
            recommendations.append(
                RecommendationResult(
                    product_code=code,
                    combined_score=combined,
                    explicit_score=explicit_score,
                    implicit_score=implicit_score,
                    product_name=product_name,
                    brands=brands,
                    categories=categories,
                    allergens=allergens,
                )
            )

        return recommendations

    def _explicit_scores(
        self,
        user_id: str,
        max_candidates: int,
    ) -> dict[str, float]:
        model = self.explicit_model
        trainset = getattr(model, "trainset", None)
        if trainset is None or max_candidates <= 0:
            return {}

        raw_uid = str(user_id)
        try:
            inner_uid = trainset.to_inner_uid(raw_uid)
        except ValueError:
            return {}

        rated_items = {item_id for item_id, _ in trainset.ur[inner_uid]}
        if len(rated_items) >= trainset.n_items:
            return {}

        to_score = [
            (raw_uid, trainset.to_raw_iid(inner_iid), 0.0)
            for inner_iid in trainset.all_items()
            if inner_iid not in rated_items
        ]
        if not to_score:
            return {}

        predictions = model.test(to_score)
        if not predictions:
            return {}

        top_predictions = nlargest(
            min(max_candidates, len(predictions)),
            predictions,
            key=lambda pred: pred.est,
        )
        return {prediction.iid: float(prediction.est) for prediction in top_predictions}

    def _implicit_scores(
        self,
        user_id: str,
        max_candidates: int,
    ) -> dict[str, float]:
        user_idx = self._implicit_user_lookup.get(str(user_id))
        if user_idx is None:
            return {}

        user_row = self._implicit_user_items[user_idx]
        rec_ids, scores = self.implicit_model.recommend(
            userid=user_idx,
            user_items=user_row,
            N=max_candidates,
            filter_already_liked_items=True,
        )

        results: dict[str, float] = {}
        for item_idx, score in zip(rec_ids, scores, strict=False):
            product_code = self._implicit_item_lookup.get(int(item_idx))
            if product_code is None:
                continue
            results[product_code] = float(score)
        return results

    def _collect_available_users(self) -> list[str]:
        explicit_users = set(self._explicit_known_users)
        implicit_users = set(self._implicit_user_lookup)
        all_users = sorted(explicit_users | implicit_users)
        return all_users

    def _collect_explicit_users(self, feedback_path: Path) -> set[str]:
        if not feedback_path.exists():
            return set()

        frame = pd.read_csv(
            feedback_path,
            usecols=["UserID"],
            dtype={"UserID": str},
        )
        return set(frame["UserID"].dropna().astype(str))

    def _build_implicit_structures(
        self,
        feedback_path: Path,
    ) -> tuple[csr_matrix, dict[str, int], dict[int, str]]:
        frame = pd.read_csv(feedback_path)
        frame["Event"] = frame["Event"].astype(str)
        frame["Strength"] = frame["Event"].map(EVENT_STRENGTH).fillna(1.0)

        user_codes, user_index = pd.factorize(frame["UserID"], sort=True)
        item_codes, item_index = pd.factorize(frame["ProductCode"], sort=True)

        interactions = coo_matrix(
            (
                frame["Strength"].to_numpy(dtype=np.float32),
                (user_codes, item_codes),
            ),
            shape=(len(user_index), len(item_index)),
        ).tocsr()

        user_lookup = {str(user): idx for idx, user in enumerate(user_index)}
        item_lookup = {idx: str(code) for idx, code in enumerate(item_index)}

        return interactions, user_lookup, item_lookup

    def _load_explicit_model(self, model_path: Path) -> SVD:
        with open(model_path, "rb") as handle:
            model: SVD = pickle.load(handle)
        return model

    def _load_implicit_model(self, model_path: Path) -> AlternatingLeastSquares:
        with open(model_path, "rb") as handle:
            model: AlternatingLeastSquares = pickle.load(handle)
        return model

    def _load_user_display_names(self, users_path: Path) -> dict[str, str]:
        if not users_path.exists():
            return {}
        frame = pd.read_csv(users_path, usecols=["UserID", "Name"])
        return dict(zip(frame["UserID"].astype(str), frame["Name"].astype(str)))

    def _load_product_catalog(
        self,
        directory: Path,
        *,
        upto: int | None = None,
    ) -> dict[str, dict[str, str | None]]:
        if not directory.exists():
            return {}

        records: list[pd.DataFrame] = []
        columns = ["code", "product_name", "brands", "categories"]

        for chunk_no, path in enumerate(sorted(directory.glob("*.parquet")), start=1):
            try:
                frame = pd.read_parquet(path, columns=columns)
            except (ValueError, KeyError):
                frame = pd.read_parquet(path)
            records.append(frame)
            if upto is not None and chunk_no >= upto:
                break

        if not records:
            return {}

        merged = (
            pd.concat(records, ignore_index=True)
            .dropna(subset=["code"])
            .drop_duplicates(subset="code", keep="first")
        )

        catalog = {}
        for row in merged.itertuples(index=False):
            code = str(getattr(row, "code", "")).strip()
            if not code:
                continue
            catalog[code] = {
                "product_name": getattr(row, "product_name", None),
                "brands": getattr(row, "brands", None),
                "categories": getattr(row, "categories", None),
            }
        return catalog

    @staticmethod
    def _normalise_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        values = np.array(list(scores.values()), dtype=float)
        min_val = float(values.min())
        max_val = float(values.max())

        if np.isclose(max_val, min_val):
            return {code: 1.0 for code in scores}

        if min_val < 0:
            values = values - min_val
            max_val -= min_val
            min_val = 0.0

        normalised = values / max_val if max_val else values
        return {
            code: float(value)
            for code, value in zip(scores.keys(), normalised, strict=False)
        }
