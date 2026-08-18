"""
Fonctions partagees entre evaluation_baseline.py et production_baseline.py.

Toute correction du pipeline (troncature, baseline, construction des
features) doit etre faite ICI UNIQUEMENT -- eviter que plusieurs scripts du
projet ne coexistent avec des versions differentes de la meme logique.
"""
from collections import Counter

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SENTINELLES_CLASSIFIEUR = ("AUCUNE_SUGGESTION", "NON_CODABLE")

COLS_CLASSIFIEURS = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
COL_TTC = "ttc_code_1"

COLS_CATEGORIELLES = [
    "lcs_code_n1", "rag_code_n1", "ragann_code_n1", "ttc_code_1_n1",
    "pred_baseline_n1", "shop_type_name", "source",
]
COLS_NUMERIQUES = [
    "nb_accords_max", "unanimite",
    "lcs_distance", "rag_confidence", "ragann_confidence", "ttc_conf_1",
    "budget",
]


# =============================================================================
# Troncature des codes COICOP (niveau = nombre de chiffres significatifs - 1)
# =============================================================================

def tronquer_niveau(code, niveau=4):
    """
    Tronque un code COICOP au niveau demande.

    Gere les NaN et les sentinelles de preprocessing ("AUCUNE_SUGGESTION",
    "NON_CODABLE"), preservees telles quelles. Un "0" terminal ne correspond
    a aucune vraie sous-categorie (convention COICOP "pas de sous-classe/
    groupe plus precise", ex. 13.9.0 == 13.9) : il est retire en cascade.
    """
    if pd.isna(code):
        return code
    s = str(code)
    if s in SENTINELLES_CLASSIFIEUR:
        return s

    n_chiffres_cible = niveau + 1
    chiffres = 0
    out = []
    for c in s:
        if c.isdigit():
            if chiffres == n_chiffres_cible:
                break
            chiffres += 1
            out.append(c)
        else:
            if chiffres < n_chiffres_cible:
                out.append(c)
    resultat = "".join(out)
    while resultat.endswith(".0"):
        resultat = resultat[:-2]
    return resultat


# =============================================================================
# Baseline : vote majoritaire, TTC arbitre
# =============================================================================

def baseline_majorite_ttc(df, cols_votants=COLS_CLASSIFIEURS, col_ttc=COL_TTC, niveau=4):
    """
    Pour chaque ligne, choisit un code COICOP :
      - code majoritaire parmi les classifieurs votants (au niveau `niveau`)
      - en cas d'egalite (plusieurs codes ex aequo), on prend TTC
      - si aucun vote possible (tous NaN/sentinelle), on prend TTC

    Peut renvoyer NaN si TTC lui-meme n'a rien propose (aucun candidat
    exploitable pour ce produit) -- a gerer explicitement en aval, ne
    jamais laisser une ligne NaN disparaitre silencieusement.
    """
    ttc_tronq = df[col_ttc].map(lambda x: tronquer_niveau(x, niveau))
    votants_tronq = df[cols_votants].apply(
        lambda col: col.map(lambda x: tronquer_niveau(x, niveau))
    )

    def predire(row_votants, ttc_code):
        votes = [v for v in row_votants if pd.notna(v) and v not in SENTINELLES_CLASSIFIEUR]
        if not votes:
            return ttc_code
        compteur = Counter(votes)
        max_count = max(compteur.values())
        codes_majoritaires = [code for code, n in compteur.items() if n == max_count]
        if len(codes_majoritaires) > 1:
            return ttc_code
        return codes_majoritaires[0]

    pred = [
        predire(votants_tronq.iloc[i].tolist(), ttc_tronq.iloc[i])
        for i in range(len(df))
    ]
    return pd.Series(pred, index=df.index, name="pred_baseline")


# =============================================================================
# Features du modele de confiance (RF)
# =============================================================================

def calculer_nb_accords_max(df, cols_pred, niveau=4):
    """Pour chaque ligne : taille du plus gros groupe d'accord (NaN ignores)."""
    tronq = df[cols_pred].apply(lambda col: col.map(lambda x: tronquer_niveau(x, niveau)))

    def max_accords(row):
        codes = [v for v in row if pd.notna(v) and v not in SENTINELLES_CLASSIFIEUR]
        return max(Counter(codes).values()) if codes else 0

    return tronq.apply(max_accords, axis=1)


def construire_features(df, y_pred_baseline):
    """Construit la matrice X des features pour le modele de confiance."""
    X = pd.DataFrame(index=df.index)

    for c in COLS_CLASSIFIEURS:
        X[f"{c}_n1"] = df[c].map(lambda x: tronquer_niveau(x, niveau=1))
    X["pred_baseline_n1"] = y_pred_baseline.map(lambda x: tronquer_niveau(x, niveau=1))

    X["nb_accords_max"] = calculer_nb_accords_max(df, COLS_CLASSIFIEURS, niveau=4)
    X["unanimite"] = (X["nb_accords_max"] == 4).astype(int)

    X["lcs_distance"] = df["lcs_distance"]
    X["rag_confidence"] = df["rag_confidence"]
    X["ragann_confidence"] = df["ragann_confidence"]
    X["ttc_conf_1"] = df["ttc_conf_1"]

    X["budget"] = df["budget"]
    X["shop_type_name"] = df["shop_type_name"]
    X["source"] = df["source"]

    return X


def imputer_valeurs_manquantes(X):
    """
    Traite les NaN pour les colonnes dont la sentinelle est une constante
    connue a priori (donc sans risque de fuite train/test) :
    - lcs_distance : 1.5 (distance bornee a 1 -> hors intervalle = "tres loin")
    - confidences : -1 (hors intervalle [0, 1])
    - categorielles : "INCONNU"

    `budget` (mediane, dependante des donnees) N'EST PAS imputee ici : elle
    est geree par un SimpleImputer a l'interieur du Pipeline sklearn
    (construire_pipeline), fitte uniquement sur le train pour eviter toute
    fuite train/test.
    """
    X = X.copy()
    X["lcs_distance"] = X["lcs_distance"].fillna(1.5)
    for c in ["rag_confidence", "ragann_confidence", "ttc_conf_1"]:
        X[c] = X[c].fillna(-1)
    for c in COLS_CATEGORIELLES:
        X[c] = X[c].fillna("INCONNU").astype(str)
    return X


# =============================================================================
# Modele de confiance : Pipeline sklearn (encodage + RF)
# =============================================================================

def construire_pipeline(random_state=42, **rf_kwargs):
    """
    Pipeline : OrdinalEncoder sur les categorielles (categorie inconnue au
    scoring -> -1, gere nativement, pas besoin de re-lister les niveaux a la
    main), passthrough sur le numerique, imputation par mediane du train
    uniquement pour `budget`, puis RandomForestClassifier.

    Hyperparametres par defaut retenus apres tuning (RandomizedSearchCV, 60
    tirages, cf. site/modelisation/index.qmd) : le tuning n'a pas ameliore le
    rappel sur la classe qui compte operationnellement (baseline_fausse) par
    rapport a ces valeurs.
    """
    defaults = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    defaults.update(rf_kwargs)

    cols_num_sans_budget = [c for c in COLS_NUMERIQUES if c != "budget"]
    encodeur_cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    preprocess = ColumnTransformer([
        ("cat", encodeur_cat, COLS_CATEGORIELLES),
        ("num", "passthrough", cols_num_sans_budget),
        ("budget", SimpleImputer(strategy="median"), ["budget"]),
    ])
    return Pipeline([("preprocess", preprocess), ("rf", RandomForestClassifier(**defaults))])
