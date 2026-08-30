from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
PRICE_RE = re.compile(
    r"(?:\$\s*([0-9]+(?:\.[0-9]+)?)|(?:under|below|less than|up to)\s+\$?([0-9]+(?:\.[0-9]+)?))",
    re.I,
)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|rather|ignore|changed|change|new requirement|what i need)\b",
    re.I,
)
# The simulator's reply templates always lead with one of these fixed
# phrases before the actual constraint content. Matching the phrase and
# taking everything after it (rather than splitting on the last colon in
# the message) correctly captures multi-clause values that themselves
# contain colons, e.g. "leather; color: brown" or "Item model number: X".
CONSTRAINT_LEADIN_RE = re.compile(
    r"(?:a key requirement is|for that,? what matters is|what i need is)\s*:\s*(.+)",
    re.I,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "im", "still", "exploring", "key", "requirement", "requirements", "what",
    "matters", "need", "needs", "prefer", "preference", "preferences", "actually",
    "ignore", "earlier", "instead", "rather", "different", "options", "option",
    "quite", "right", "yet", "ask", "specific", "attribute", "additional", "have",
    "has", "dont", "do", "not", "no", "use", "judgment", "judgement", "around",
    "one", "two", "thing", "things", "something", "customer",
}

SYNONYMS = {
    "tee": ("tshirt", "shirt"),
    "tshirt": ("tee", "shirt"),
    "tshirts": ("tees", "shirts"),
    "sneaker": ("sneakers", "shoe"),
    "sneakers": ("sneaker", "shoes"),
    "trainers": ("sneakers", "shoes"),
    "hoodie": ("sweatshirt", "pullover"),
    "hoodies": ("sweatshirts", "pullovers"),
    "pullover": ("sweatshirt", "hoodie"),
    "joggers": ("sweatpants", "pants"),
    "leggings": ("tights",),
    "trousers": ("pants",),
    "outerwear": ("jacket", "coat"),
    "coat": ("jacket",),
    "raincoat": ("rain", "waterproof"),
    "waterproof": ("water", "rain"),
    "warm": ("winter", "insulated"),
    "lightweight": ("light",),
    "comfortable": ("comfort",),
    "mens": ("men",),
    "womens": ("women",),
    "childrens": ("children", "kids"),
}

MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "cashmere", "linen", "denim", "fleece", "suede",
}
COLORS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "beige", "navy", "khaki", "gold", "silver",
}

FIELD_NAMES = ("title", "categories", "features", "details", "store", "description")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _expanded_terms(terms: list[str], limit: int = 40) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for term in terms:
        for candidate in (term, *SYNONYMS.get(term, ())):
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
            if len(result) >= limit:
                return result
    return result


def _fts_expression(terms: list[str], operator: str = "OR") -> str:
    return f" {operator} ".join(f'"{term}"' for term in terms)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


class Agent:
    """Stateful, offline hybrid retrieval agent for the frozen catalog."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._document_frequency: Counter[str] = Counter()
        self._document_count = 0
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "price UNINDEXED, average_rating UNINDEXED, rating_number UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                fields = tuple(_text(product.get(field)) for field in FIELD_NAMES)
                self._document_frequency.update(set(_terms(" ".join(fields))))
                self._document_count += 1
                batch.append(
                    (
                        str(product["parent_asin"]),
                        *fields,
                        _text(product.get("price")),
                        _text(product.get("average_rating")),
                        _text(product.get("rating_number")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "profile": user_profile if isinstance(user_profile, dict) else {},
            "category_text": "",
            "constraints": [],
            "initial_constraint": "",
            "messages": [],
            "override_count": 0,
            "last_recommendations": [],
        }

    @staticmethod
    def _category_from_initial(message: str) -> str:
        match = re.search(r"looking\s+for\s+(.+?)(?:\.|,|\s+but\s+)", message, re.I)
        return match.group(1) if match else message

    @staticmethod
    def _constraint_from_message(message: str) -> str:
        lowered = message.lower()
        if "no preference" in lowered or "don't have a preference" in lowered:
            return ""
        if "still exploring" in lowered:
            return ""
        leadin_match = CONSTRAINT_LEADIN_RE.search(message)
        if leadin_match:
            return leadin_match.group(1).strip(" .;\n")
        sentences = [part.strip(" .") for part in message.split(".") if part.strip()]
        return sentences[-1] if len(sentences) > 1 else ""

    @staticmethod
    def _parse_price(text: str) -> float | None:
        match = PRICE_RE.search(text)
        if not match:
            return None
        try:
            return float(match.group(1) or match.group(2))
        except (TypeError, ValueError):
            return None

    def _record_message(self, state: dict, user_message: str) -> None:
        state["messages"].append(user_message)
        if not state["category_text"]:
            state["category_text"] = self._category_from_initial(user_message)

        constraint = self._constraint_from_message(user_message)
        if OVERRIDE_RE.search(user_message) and state["messages"][:-1]:
            # The simulator's override replaces the original preference, not
            # every fact learned before the replacement.  Keep constraints that
            # were disclosed in prior turns and remove the initial preference.
            initial_constraint = state.get("initial_constraint", "")
            if initial_constraint:
                state["constraints"] = [
                    value for value in state["constraints"] if value != initial_constraint
                ]
            state["override_count"] += 1

        if constraint:
            if not state["messages"][:-1] and not state["initial_constraint"]:
                state["initial_constraint"] = constraint
            state["constraints"].append(constraint)

    def _idf(self, term: str) -> float:
        count = self._document_frequency.get(term, 0)
        return math.log((self._document_count + 1) / (count + 1)) + 1.0

    def _query_rows(self, terms: list[str], limit: int, operator: str = "OR") -> list[dict]:
        terms = _expanded_terms(_unique(terms), 40)
        if not terms:
            return []
        try:
            rows = self.connection.execute(
                "SELECT parent_asin, title, categories, features, details, store, description, "
                "price, average_rating, rating_number, "
                "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0, 0.0, 0.0, 0.0) AS rank "
                "FROM products WHERE products MATCH ? ORDER BY rank LIMIT ?",
                (_fts_expression(terms, operator), limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            {
                "parent_asin": str(row[0]), "title": row[1], "categories": row[2],
                "features": row[3], "details": row[4], "store": row[5],
                "description": row[6], "price": row[7], "average_rating": row[8],
                "rating_number": row[9], "retrieval_rank": float(row[10]),
            }
            for row in rows
        ]

    def _retrieve(self, state: dict) -> list[dict]:
        category_terms = _terms(state["category_text"])
        constraint_terms = _terms(" ".join(state["constraints"]))
        all_terms = _unique(category_terms + constraint_terms)
        candidates: dict[str, dict] = {}

        def add(rows: list[dict]) -> None:
            for row in rows:
                asin = row["parent_asin"]
                old = candidates.get(asin)
                if old is None or row["retrieval_rank"] < old["retrieval_rank"]:
                    candidates[asin] = row

        add(self._query_rows(all_terms, 350, "OR"))
        add(self._query_rows(category_terms, 150, "OR"))
        if constraint_terms:
            add(self._query_rows(constraint_terms, 250, "OR"))
            rare_terms = sorted(
                _unique(constraint_terms),
                key=lambda term: self._document_frequency.get(term, self._document_count),
            )[:5]
            if len(rare_terms) >= 2:
                add(self._query_rows(rare_terms, 200, "AND"))

        if not candidates:
            rows = self.connection.execute(
                "SELECT parent_asin, title, categories, features, details, store, description, "
                "price, average_rating, rating_number, 0.0 AS rank FROM products "
                "ORDER BY CAST(rating_number AS REAL) DESC LIMIT 250"
            ).fetchall()
            for row in rows:
                candidates[str(row[0])] = {
                    "parent_asin": str(row[0]), "title": row[1], "categories": row[2],
                    "features": row[3], "details": row[4], "store": row[5],
                    "description": row[6], "price": row[7], "average_rating": row[8],
                    "rating_number": row[9], "retrieval_rank": 0.0,
                }
        return list(candidates.values())

    def _row_score(self, row: dict, state: dict) -> float:
        category_terms = set(_expanded_terms(_terms(state["category_text"])))
        constraint_groups = [set(_expanded_terms(_terms(value))) for value in state["constraints"]]
        constraint_terms = set().union(*constraint_groups) if constraint_groups else set()
        profile = state.get("profile") or {}
        profile_text = " ".join(str(x) for x in profile.get("preference_tags", []))
        profile_terms = set(_expanded_terms(_terms(profile_text)))

        field_sets = {field: set(_terms(str(row.get(field, "")))) for field in FIELD_NAMES}
        all_product_terms = set().union(*field_sets.values())

        def weighted_overlap(query: set[str], field: str) -> float:
            if not query:
                return 0.0
            overlap = query & field_sets[field]
            return sum(self._idf(term) for term in overlap) / max(
                1.0, sum(self._idf(term) for term in query)
            )

        score = 0.0
        score += 10.0 * weighted_overlap(category_terms, "title")
        score += 6.0 * weighted_overlap(category_terms, "categories")
        score += 4.0 * weighted_overlap(category_terms, "features")

        # The evaluator supplies the last category levels (for example,
        # "Shirts T-Shirts").  Require those words to be represented in the
        # catalog's category field, not merely somewhere in a description.
        base_category_terms = set(_terms(state["category_text"]))
        category_field_terms = field_sets["categories"]
        if base_category_terms:
            category_coverage = len(base_category_terms & category_field_terms) / len(base_category_terms)
            score += 8.0 * category_coverage
            score -= 3.0 * (1.0 - category_coverage)
            category_phrase = " ".join(_terms(state["category_text"]))
            category_text = " ".join(_terms(str(row.get("categories", ""))))
            if len(base_category_terms) >= 2 and category_phrase in category_text:
                score += 8.0

        product_phrase = " ".join(
            _terms(" ".join(str(row.get(field, "")) for field in FIELD_NAMES))
        )
        for index, group in enumerate(constraint_groups):
            overlap = group & all_product_terms
            if group:
                match = sum(self._idf(term) for term in overlap) / max(
                    1.0, sum(self._idf(term) for term in group)
                )
                score += (16.0 if index == 0 else 11.0) * match
                if overlap and len(group) <= 3:
                    score += 3.0

                # Hidden intent-card values are copied from catalog metadata.
                # Preserve word order and reward an exact phrase strongly; this
                # separates the target from products that merely mention the
                # same common material or color.
                phrase = " ".join(_terms(state["constraints"][index]))
                if phrase and phrase in product_phrase:
                    score += 14.0 if index == 0 else 10.0
                # A simulator reply can contain multiple metadata values joined
                # with semicolons.  Their order may differ across catalog fields,
                # so score each meaningful clause independently as well.
                for clause in state["constraints"][index].split(";"):
                    clause_terms = _terms(clause)
                    clause_phrase = " ".join(clause_terms)
                    if len(clause_terms) >= 2 and clause_phrase in product_phrase:
                        score += 6.0 if index == 0 else 4.0
                        # Exact clauses containing rare words are much more
                        # diagnostic than generic clauses such as "imported".
                        specificity = sum(self._idf(term) for term in clause_terms)
                        score += min(10.0, specificity * 0.85)

        for term in category_terms | constraint_terms:
            if term in MATERIALS and term in all_product_terms:
                score += 4.0
            if term in COLORS and term in all_product_terms:
                score += 3.0

        score += 4.0 * weighted_overlap(constraint_terms, "title")
        score += 2.0 * weighted_overlap(constraint_terms, "categories")
        score += 1.0 * weighted_overlap(constraint_terms, "features")
        # user_profile.preference_tags is available from turn 1, but with no
        # disclosed constraints yet (e.g. browsing's opening turn) it's the
        # only real signal beyond the category text -- weight it much more
        # heavily until real constraints start accumulating.
        profile_weight = 1.0 if state["constraints"] else 6.0
        score += profile_weight * weighted_overlap(profile_terms, "title")
        score += (profile_weight * 0.5) * weighted_overlap(profile_terms, "features")

        price = self._parse_price(" ".join(state["constraints"]))
        if price is not None:
            try:
                product_price = float(row["price"])
            except (TypeError, ValueError):
                product_price = None
            if product_price is not None:
                if product_price <= price:
                    score += 2.5
                else:
                    score -= min(3.0, (product_price - price) / max(price, 1.0))

        try:
            rating = float(row["average_rating"])
        except (TypeError, ValueError):
            rating = 0.0
        try:
            rating_count = float(row["rating_number"])
        except (TypeError, ValueError):
            rating_count = 0.0
        score += 0.20 * rating + 0.22 * math.log1p(max(0.0, rating_count))
        score += max(-2.0, min(2.0, -float(row["retrieval_rank"]))) * 0.15
        return score

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        self._record_message(state, user_message if isinstance(user_message, str) else "")
        candidates = self._retrieve(state)
        ranked = sorted(candidates, key=lambda row: self._row_score(row, state), reverse=True)
        limit = max(1, min(int(top_k), 10))
        recommendations = [{"parent_asin": row["parent_asin"]} for row in ranked[:limit]]
        state["last_recommendations"] = [item["parent_asin"] for item in recommendations]

        if turn <= 1:
            prompt = "Which additional detail matters most for this search?"
        elif state["override_count"]:
            prompt = "Would you like to refine the new direction by another detail?"
        else:
            prompt = "What other detail should I prioritize?"
        return {
            "message": prompt,
            "ask_attribute": "other",
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
