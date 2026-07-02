"""
Stats descriptives sur l'accord des classifieurs de codification COICOP.

Analyse :
- des cas d'accord unanime des classifieurs de base (et faux positifs associés)
- de la dissociation selon le comportement du LLM-judge
- des cas où un seul classifieur a raison contre tous les autres
- de l'évolution de ces statistiques selon le niveau de troncature COICOP

Convention de niveau (nombre de chiffres significatifs = niveau + 1) :
    niveau 1 → "XX"        (division)
    niveau 2 → "XX.X"      (groupe)
    niveau 3 → "XX.X.X"    (classe)
    niveau 4 → "XX.X.X.X"  (sous-classe)
"""
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

from src.coicop import tronquer_niveau

# Toutes les fonctions de stats acceptent un paramètre `verbose` :
#   - verbose=True (défaut)  → affiche un résumé texte, pratique en exploration interactive
#   - verbose=False          → silencieux
# Dans les deux cas, un résumé "propre" (DataFrame) est toujours attaché en
# `.attrs` de l'objet retourné, pour un affichage tabulaire (ex. dans un rapport
# Quarto) sans dépendre du texte imprimé sur stdout.


# =============================================================================
# Utilitaires de base
# =============================================================================

def accord_classifieurs(df, cols_pred, niveau=4):
    """
    Pour chaque ligne, indique si tous les classifieurs donnent le même code
    une fois tronqués au niveau demandé.

    Returns
    -------
    pd.DataFrame avec les colonnes :
        - code_consensus : code partagé si accord unanime, NaN sinon
        - tous_accord    : booléen
    """
    tronq = df[cols_pred].apply(
        lambda col: col.map(lambda x: tronquer_niveau(x, niveau))
    )
    premier = tronq.iloc[:, 0]
    tous_accord = tronq.eq(premier, axis=0).all(axis=1)
    code_consensus = premier.where(tous_accord)
    return pd.DataFrame({"code_consensus": code_consensus, "tous_accord": tous_accord})


# =============================================================================
# Stats d'accord des classifieurs de base
# =============================================================================

def stats_accord(df, cols_pred, col_vrai, niveau=4, verbose=True):
    """
    Stats globales sur les cas d'accord unanime au niveau donné :
    proportion d'accord, dont corrects vs faux positifs.

    Returns
    -------
    pd.DataFrame enrichi des colonnes 'code_consensus', 'tous_accord',
    'vrai_tronq', et une colonne tronquée par classifieur (`{col}_tronq`).
    Le résumé chiffré est disponible en `.attrs["resume"]` (DataFrame 1 ligne).
    """
    acc = accord_classifieurs(df, cols_pred, niveau)
    out = df.copy()
    out["code_consensus"] = acc["code_consensus"]
    out["tous_accord"] = acc["tous_accord"]
    out["vrai_tronq"] = out[col_vrai].map(lambda x: tronquer_niveau(x, niveau))

    # Versions tronquées des prédictions (pour affichage cohérent)
    for c in cols_pred:
        out[f"{c}_tronq"] = out[c].map(lambda x: tronquer_niveau(x, niveau))

    n_total = len(out)
    n_accord = int(out["tous_accord"].sum())

    df_acc = out[out["tous_accord"]]
    n_correct = int((df_acc["code_consensus"] == df_acc["vrai_tronq"]).sum())
    n_fp = n_accord - n_correct

    resume = pd.DataFrame([{
        "niveau": niveau,
        "n_total": n_total,
        "n_accord": n_accord,
        "pct_accord": n_accord / n_total,
        "n_correct": n_correct,
        "pct_correct_des_accords": n_correct / max(n_accord, 1),
        "n_fp": n_fp,
        "pct_fp_des_accords": n_fp / max(n_accord, 1),
        "pct_fp_du_total": n_fp / n_total,
    }])
    out.attrs["resume"] = resume

    if verbose:
        print(f"=== Accord unanime au niveau {niveau} ({len(cols_pred)} classifieurs) ===")
        print(f"Total observations              : {n_total}")
        print(f"Accord unanime                  : {n_accord:>6}  ({n_accord/n_total:.1%})")
        print(f"  ├─ corrects                   : {n_correct:>6}  ({n_correct/max(n_accord,1):.1%} des accords)")
        print(f"  └─ faux positifs (FP)         : {n_fp:>6}  ({n_fp/max(n_accord,1):.1%} des accords)")
        print(f"Taux de FP / total              : {n_fp/n_total:.1%}")
    return out


def analyse_faux_positifs(df_stats, niveau=4, top_n=15, verbose=True):
    """
    Sur les FP (accord unanime mais code faux), regarde :
      - les vrais codes (niveau donné) les plus concernés
      - les codes prédits à tort
      - les confusions vrai → prédit les plus fréquentes
      - si dispo : la division COICOP (niveau 1) concernée

    À appeler sur la sortie de stats_accord().

    Returns
    -------
    pd.DataFrame des lignes en faux positif. Le détail (tops et confusions)
    est disponible en `.attrs` : "top_vrais", "top_predits", "confusions",
    et "confusions_division" (si niveau > 1).
    """
    df_fp = df_stats[
        df_stats["tous_accord"]
        & (df_stats["code_consensus"] != df_stats["vrai_tronq"])
    ].copy()

    top_vrais = df_fp["vrai_tronq"].value_counts().head(top_n).rename_axis("code").reset_index(name="n")
    top_predits = df_fp["code_consensus"].value_counts().head(top_n).rename_axis("code").reset_index(name="n")
    confusions = (
        df_fp.groupby(["vrai_tronq", "code_consensus"])
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .rename("n")
        .reset_index()
    )
    df_fp.attrs["top_vrais"] = top_vrais
    df_fp.attrs["top_predits"] = top_predits
    df_fp.attrs["confusions"] = confusions

    # Vue agrégée à la division (niveau 1)
    if niveau > 1:
        df_fp["vrai_div"] = df_fp["vrai_tronq"].str[:2]
        df_fp["pred_div"] = df_fp["code_consensus"].str[:2]
        confusions_div = (
            df_fp.groupby(["vrai_div", "pred_div"])
            .size()
            .sort_values(ascending=False)
            .head(top_n)
            .rename("n")
            .reset_index()
        )
        df_fp.attrs["confusions_division"] = confusions_div

    if verbose:
        print(f"\n=== {len(df_fp)} faux positifs ===\n")
        print(f"→ Top vrais codes (niveau {niveau}) parmi les FP :")
        print(top_vrais.to_string(index=False))
        print("\n→ Top codes prédits à tort (consensus erroné) :")
        print(top_predits.to_string(index=False))
        print("\n→ Top confusions (vrai → prédit) :")
        print(confusions.to_string(index=False))
        if niveau > 1:
            print("\n→ Confusions agrégées au niveau 1 (division) :")
            print(df_fp.attrs["confusions_division"].to_string(index=False))

    return df_fp


# =============================================================================
# Dissociation selon le comportement du LLM
# =============================================================================

def stats_accord_avec_llm(df, cols_base, col_llm, col_vrai, niveau=4, verbose=True):
    """
    Sur les cas où les classifieurs de base sont unanimes au niveau donné,
    dissocie selon que le LLM suit ce consensus ou non, et regarde la justesse.

    Returns
    -------
    pd.DataFrame restreint aux lignes d'accord unanime des classifieurs de base,
    enrichi de 'code_consensus', 'vrai_tronq', 'llm_tronq', 'llm_suit',
    'consensus_correct', 'llm_correct'. Le récap chiffré est disponible en
    `.attrs["recap"]`.
    """
    acc = accord_classifieurs(df, cols_base, niveau)
    out = df.copy()
    out["code_consensus"] = acc["code_consensus"]
    out["tous_accord"] = acc["tous_accord"]
    out["vrai_tronq"] = out[col_vrai].map(lambda x: tronquer_niveau(x, niveau))
    out["llm_tronq"] = out[col_llm].map(lambda x: tronquer_niveau(x, niveau))

    df_acc = out[out["tous_accord"]].copy()
    df_acc["llm_suit"] = df_acc["llm_tronq"] == df_acc["code_consensus"]
    df_acc["consensus_correct"] = df_acc["code_consensus"] == df_acc["vrai_tronq"]
    df_acc["llm_correct"] = df_acc["llm_tronq"] == df_acc["vrai_tronq"]

    n_acc = len(df_acc)
    n_suit = int(df_acc["llm_suit"].sum())
    n_nosuit = n_acc - n_suit

    suit = df_acc[df_acc["llm_suit"]]
    n_suit_ok = int(suit["consensus_correct"].sum())
    n_suit_fp = n_suit - n_suit_ok

    nosuit = df_acc[~df_acc["llm_suit"]]
    n_base_ok = int(nosuit["consensus_correct"].sum())
    n_llm_ok = int(nosuit["llm_correct"].sum())
    n_aucun_ok = len(nosuit) - n_base_ok - n_llm_ok

    recap = pd.DataFrame({
        "cas": [
            "LLM suit, consensus correct",
            "LLM suit, FP partagé",
            "LLM ne suit pas, base correcte",
            "LLM ne suit pas, LLM correct",
            "LLM ne suit pas, personne correct",
        ],
        "n": [n_suit_ok, n_suit_fp, n_base_ok, n_llm_ok, n_aucun_ok],
    })
    recap["pct_accord"] = recap["n"] / n_acc
    df_acc.attrs["recap"] = recap

    if verbose:
        print(f"=== Accord unanime des {len(cols_base)} classifieurs de base (niveau {niveau}) : {n_acc} cas ===\n")
        print(f"┌─ LLM SUIT le consensus           : {n_suit:>6}  ({n_suit/n_acc:.1%})")
        print(f"│    ├─ tout le monde a raison     : {n_suit_ok:>6}  ({n_suit_ok/max(n_suit,1):.1%})")
        print(f"│    └─ FP partagés (5/5 faux)     : {n_suit_fp:>6}  ({n_suit_fp/max(n_suit,1):.1%})")
        print(f"│")
        print(f"└─ LLM NE SUIT PAS le consensus    : {n_nosuit:>6}  ({n_nosuit/n_acc:.1%})")
        print(f"     ├─ base a raison, LLM tort    : {n_base_ok:>6}  ({n_base_ok/max(n_nosuit,1):.1%})")
        print(f"     ├─ LLM a raison, base tort    : {n_llm_ok:>6}  ({n_llm_ok/max(n_nosuit,1):.1%})")
        print(f"     └─ personne n'a raison        : {n_aucun_ok:>6}  ({n_aucun_ok/max(n_nosuit,1):.1%})")
        print("\n", recap.to_string(index=False))

    return df_acc


# =============================================================================
# Cas où un seul classifieur a raison contre tous les autres
# =============================================================================

def stats_classifieur_seul_correct(df, cols_pred, col_vrai, niveau=4, verbose=True):
    """
    Pour chaque ligne, identifie si exactement UN classifieur a raison
    (au niveau donné) et que tous les autres ont tort.

    Returns
    -------
    pd.DataFrame enrichi des colonnes :
        - vrai_tronq         : vrai code tronqué
        - n_corrects         : nb de classifieurs corrects
        - seul_correct       : booléen (True si exactement 1 correct)
        - classifieur_seul   : nom du classifieur seul correct (NaN sinon)
        - {col}_tronq        : version tronquée de chaque prédiction
    La répartition par classifieur sauveur est disponible en `.attrs["repart"]`.
    """
    out = df.copy()
    out["vrai_tronq"] = out[col_vrai].map(lambda x: tronquer_niveau(x, niveau))

    corrects = pd.DataFrame({
        c: out[c].map(lambda x: tronquer_niveau(x, niveau)) == out["vrai_tronq"]
        for c in cols_pred
    })

    out["n_corrects"] = corrects.sum(axis=1)
    out["seul_correct"] = out["n_corrects"] == 1

    def _qui(row):
        return row[row].index[0] if row.sum() == 1 else None
    out["classifieur_seul"] = corrects.apply(_qui, axis=1)

    # Versions tronquées pour affichage cohérent
    for c in cols_pred:
        out[f"{c}_tronq"] = out[c].map(lambda x: tronquer_niveau(x, niveau))

    n_total = len(out)
    n_seul = int(out["seul_correct"].sum())

    repart = out.loc[out["seul_correct"], "classifieur_seul"].value_counts()
    repart_pct = repart / max(n_seul, 1)  # fraction 0-1, cohérent avec les autres colonnes pct_*
    repart_df = pd.DataFrame({"n": repart, "pct": repart_pct}).reset_index(names="classifieur")
    out.attrs["resume"] = pd.DataFrame([{
        "niveau": niveau, "n_total": n_total, "n_seul_correct": n_seul,
        "pct_seul_correct": n_seul / n_total,
    }])
    out.attrs["repart"] = repart_df

    if verbose:
        print(f"=== Cas où UN SEUL classifieur sur {len(cols_pred)} a raison (niveau {niveau}) ===")
        print(f"Total observations           : {n_total}")
        print(f"Un seul correct              : {n_seul}  ({n_seul/n_total:.1%})\n")
        print("→ Répartition par classifieur sauveur :")
        print(repart_df.to_string(index=False, formatters={"pct": "{:.1%}".format}))

    return out


def analyse_classifieur_seul(df_seul, classifieur, top_n=15, verbose=True):
    """
    Codes sur lesquels `classifieur` a raison quand tous les autres ont tort.

    Returns
    -------
    pd.DataFrame (sous-ensemble de df_seul pour ce classifieur), avec le top
    des codes concernés disponible en `.attrs["top_codes"]`.
    """
    sub = df_seul[df_seul["classifieur_seul"] == classifieur].copy()
    top_codes = sub["vrai_tronq"].value_counts().head(top_n).rename_axis("code").reset_index(name="n")
    sub.attrs["top_codes"] = top_codes
    sub.attrs["classifieur"] = classifieur

    if verbose:
        print(f"\n=== {classifieur} seul correct : {len(sub)} cas ===")
        print(top_codes.to_string(index=False))
    return sub


# =============================================================================
# Cas où une majorité (n-1) de classifieurs est d'accord contre 1 dissident
# =============================================================================

def stats_majorite_3_1(df, cols_pred, col_vrai, niveau=4, verbose=True):
    """
    Pour chaque ligne, identifie les cas où tous les classifieurs de
    `cols_pred` votent (aucun NaN) et se répartissent en un groupe majoritaire
    de taille `len(cols_pred) - 1` et un dissident isolé (ex. 3 contre 1 pour
    4 classifieurs), une fois les codes tronqués au niveau demandé.

    Returns
    -------
    pd.DataFrame enrichi des colonnes :
        - vrai_tronq          : vrai code tronqué
        - {col}_tronq         : version tronquée de chaque prédiction
        - code_majorite       : code partagé par la majorité (NaN si pas de cas 3v1)
        - code_minoritaire    : code du dissident (NaN si pas de cas 3v1)
        - classifieur_dissident : nom du classifieur isolé (NaN sinon)
        - cas_3_1             : booléen
        - majorite_correcte   : la majorité a-t-elle raison ? (NaN hors cas 3v1)
        - minorite_correcte   : le dissident a-t-il raison ? (NaN hors cas 3v1)
    Le résumé chiffré est disponible en `.attrs["resume"]`. La répartition
    par classifieur dissident est disponible en `.attrs["repart_dissident"]`
    (colonnes `classifieur`, `n`, `pct` pour la part de chaque classifieur
    parmi les dissidents, puis `n_majorite_correcte`/`pct_majorite_correcte`,
    `n_minorite_correcte`/`pct_minorite_correcte`,
    `n_aucun_correct`/`pct_aucun_correct` pour la répartition, propre à ce
    classifieur, des cas où la majorité/lui/personne a raison).
    """
    out = df.copy()
    out["vrai_tronq"] = out[col_vrai].map(lambda x: tronquer_niveau(x, niveau))

    tronq = df[cols_pred].apply(lambda col: col.map(lambda x: tronquer_niveau(x, niveau)))
    for c in cols_pred:
        out[f"{c}_tronq"] = tronq[c]

    taille_majorite = len(cols_pred) - 1

    def _analyse_ligne(row):
        votes = {c: v for c, v in row.items() if pd.notna(v)}
        if len(votes) != len(cols_pred):
            return pd.Series([False, None, None, None])
        compteur = Counter(votes.values())
        if sorted(compteur.values()) != sorted([1] + [taille_majorite]):
            return pd.Series([False, None, None, None])
        code_maj = [code for code, n in compteur.items() if n == taille_majorite][0]
        code_min = [code for code, n in compteur.items() if n == 1][0]
        dissident = [c for c, v in votes.items() if v == code_min][0]
        return pd.Series([True, code_maj, code_min, dissident])

    resultats = tronq.apply(_analyse_ligne, axis=1)
    resultats.columns = ["cas_3_1", "code_majorite", "code_minoritaire", "classifieur_dissident"]
    out = pd.concat([out, resultats], axis=1)

    df_31 = out[out["cas_3_1"]]
    out["majorite_correcte"] = pd.NA
    out["minorite_correcte"] = pd.NA
    out.loc[df_31.index, "majorite_correcte"] = df_31["code_majorite"] == df_31["vrai_tronq"]
    out.loc[df_31.index, "minorite_correcte"] = df_31["code_minoritaire"] == df_31["vrai_tronq"]

    n_total = len(out)
    n_3_1 = int(out["cas_3_1"].sum())
    df_31 = out[out["cas_3_1"]]
    n_maj_ok = int(df_31["majorite_correcte"].sum())
    n_min_ok = int(df_31["minorite_correcte"].sum())
    n_aucun_ok = n_3_1 - n_maj_ok - n_min_ok

    resume = pd.DataFrame([{
        "niveau": niveau,
        "n_total": n_total,
        "n_3_1": n_3_1,
        "pct_3_1": n_3_1 / n_total,
        "n_majorite_correcte": n_maj_ok,
        "pct_majorite_correcte": n_maj_ok / max(n_3_1, 1),
        "n_minorite_correcte": n_min_ok,
        "pct_minorite_correcte": n_min_ok / max(n_3_1, 1),
        "n_aucun_correct": n_aucun_ok,
        "pct_aucun_correct": n_aucun_ok / max(n_3_1, 1),
    }])
    out.attrs["resume"] = resume

    repart = df_31["classifieur_dissident"].value_counts()
    repart_pct = repart / max(n_3_1, 1)
    repart_df = pd.DataFrame({"n": repart, "pct": repart_pct}).reset_index(names="classifieur")

    # Pour chaque classifieur dissident, qui a raison quand c'est lui qui diverge ?
    par_dissident = df_31.groupby("classifieur_dissident")
    n_par_dissident = par_dissident.size()
    n_maj_par_dissident = par_dissident["majorite_correcte"].sum().astype(int)
    n_min_par_dissident = par_dissident["minorite_correcte"].sum().astype(int)
    n_aucun_par_dissident = n_par_dissident - n_maj_par_dissident - n_min_par_dissident

    detail = pd.DataFrame({
        "n_majorite_correcte": n_maj_par_dissident,
        "pct_majorite_correcte": n_maj_par_dissident / n_par_dissident,
        "n_minorite_correcte": n_min_par_dissident,
        "pct_minorite_correcte": n_min_par_dissident / n_par_dissident,
        "n_aucun_correct": n_aucun_par_dissident,
        "pct_aucun_correct": n_aucun_par_dissident / n_par_dissident,
    }).reindex(repart_df["classifieur"]).reset_index(drop=True)

    repart_df = pd.concat([repart_df, detail], axis=1)
    out.attrs["repart_dissident"] = repart_df

    if verbose:
        print(f"=== Cas {taille_majorite} contre 1 parmi {len(cols_pred)} classifieurs (niveau {niveau}) ===")
        print(f"Total observations           : {n_total}")
        print(f"Cas {taille_majorite} vs 1                 : {n_3_1}  ({n_3_1/n_total:.1%})\n")
        print(f"  ├─ majorité a raison       : {n_maj_ok:>6}  ({n_maj_ok/max(n_3_1,1):.1%})")
        print(f"  ├─ dissident a raison      : {n_min_ok:>6}  ({n_min_ok/max(n_3_1,1):.1%})")
        print(f"  └─ personne n'a raison     : {n_aucun_ok:>6}  ({n_aucun_ok/max(n_3_1,1):.1%})\n")
        print("→ Répartition par classifieur dissident (et qui a raison quand c'est lui) :")
        print(repart_df.to_string(index=False, formatters={
            c: "{:.1%}".format for c in repart_df.columns if str(c).startswith("pct")
        }))

    return out


# =============================================================================
# Wrappers multi-niveaux
# =============================================================================

def stats_accord_multi_niveaux(df, cols_base, col_llm, col_vrai, niveaux=(1, 2, 3, 4), verbose=True):
    """
    Lance stats_accord puis stats_accord_avec_llm pour chaque niveau demandé.

    Returns
    -------
    dict {niveau: {"df_stats": ..., "df_acc": ...}}
    """
    resultats = {}
    for n in niveaux:
        df_stats = stats_accord(df, cols_base, col_vrai, niveau=n, verbose=verbose)
        df_acc = stats_accord_avec_llm(df, cols_base, col_llm, col_vrai, niveau=n, verbose=verbose)
        resultats[n] = {"df_stats": df_stats, "df_acc": df_acc}
    return resultats


def stats_seul_multi_niveaux(df, cols_tous, col_vrai, niveaux=(1, 2, 3, 4), verbose=True):
    """
    Lance stats_classifieur_seul_correct pour chaque niveau.

    Returns
    -------
    dict {niveau: df_seul}
    """
    resultats = {}
    for n in niveaux:
        resultats[n] = stats_classifieur_seul_correct(df, cols_tous, col_vrai, niveau=n, verbose=verbose)
    return resultats


def stats_majorite_3_1_multi_niveaux(df, cols_pred, col_vrai, niveaux=(1, 2, 3, 4), verbose=True):
    """
    Lance stats_majorite_3_1 pour chaque niveau.

    Returns
    -------
    dict {niveau: df_31}
    """
    resultats = {}
    for n in niveaux:
        resultats[n] = stats_majorite_3_1(df, cols_pred, col_vrai, niveau=n, verbose=verbose)
    return resultats


def recap_3_1_multi_niveaux(df, cols_pred, col_vrai, niveaux=(1, 2, 3, 4), verbose=True):
    """
    DataFrame récap synthétique des stats de majorité (n-1 vs 1) à plusieurs niveaux.

    Colonnes :
      - n_3_1                 : nb de cas n-1 vs 1
      - pct_3_1               : part sur le total
      - pct_majorite_correcte : parmi les cas n-1 vs 1, part où la majorité a raison
      - pct_minorite_correcte : parmi les cas n-1 vs 1, part où le dissident a raison
    """
    rows = []
    for n in niveaux:
        resume = stats_majorite_3_1(df, cols_pred, col_vrai, niveau=n, verbose=False).attrs["resume"]
        rows.append(resume.iloc[0].to_dict())

    recap = pd.DataFrame(rows).set_index("niveau")
    if verbose:
        print(recap.to_string(formatters={
            "pct_3_1": "{:.1%}".format,
            "pct_majorite_correcte": "{:.1%}".format,
            "pct_minorite_correcte": "{:.1%}".format,
            "pct_aucun_correct": "{:.1%}".format,
        }))
    return recap


def rapport_complet_multi_niveaux(df, cols_base, col_llm, col_vrai,
                                  niveaux=(1, 2, 3, 4), top_n=10, verbose=True):
    """
    Rapport détaillé complet pour chaque niveau :
    accord global + analyse des FP + dissociation LLM.

    Returns
    -------
    dict {niveau: {"df_stats": ..., "df_fp": ..., "df_acc": ...}}
    """
    resultats = {}
    for n in niveaux:
        df_stats = stats_accord(df, cols_base, col_vrai, niveau=n, verbose=verbose)
        df_fp = analyse_faux_positifs(df_stats, niveau=n, top_n=top_n, verbose=verbose)
        df_acc = stats_accord_avec_llm(df, cols_base, col_llm, col_vrai, niveau=n, verbose=verbose)
        resultats[n] = {"df_stats": df_stats, "df_fp": df_fp, "df_acc": df_acc}
    return resultats


# =============================================================================
# Récap + dataviz multi-niveaux
# =============================================================================

def recap_multi_niveaux(df, cols_base, col_llm, col_vrai, niveaux=(1, 2, 3, 4), verbose=True):
    """
    DataFrame récap synthétique des stats d'accord à plusieurs niveaux.

    Colonnes :
      - n_accord     : nb d'accords unanimes des 4 base
      - pct_accord   : part sur le total
      - n_correct    : parmi les accords, combien sont justes
      - pct_correct  : taux de justesse des accords
      - n_fp_base    : faux positifs unanimes des 4 base
      - n_fp_5_5     : FP partagés 5/5 (4 base + LLM)
      - n_llm_sauve  : 4 base faux, LLM rattrape
    """
    rows = []
    n_total = len(df)

    for n in niveaux:
        acc = accord_classifieurs(df, cols_base, niveau=n)
        vrai_tronq = df[col_vrai].map(lambda x: tronquer_niveau(x, n))
        llm_tronq = df[col_llm].map(lambda x: tronquer_niveau(x, n))

        tous_accord = acc["tous_accord"]
        consensus = acc["code_consensus"]

        n_accord = int(tous_accord.sum())
        correct = (consensus == vrai_tronq) & tous_accord
        n_correct = int(correct.sum())
        n_fp_base = n_accord - n_correct

        fp_mask = tous_accord & ~correct
        n_fp_5_5 = int(((llm_tronq == consensus) & fp_mask).sum())
        n_llm_sauve = int(((llm_tronq == vrai_tronq) & fp_mask).sum())

        rows.append({
            "niveau": n,
            "n_accord": n_accord,
            "pct_accord": n_accord / n_total,
            "n_correct": n_correct,
            "pct_correct": n_correct / max(n_accord, 1),
            "n_fp_base": n_fp_base,
            "n_fp_5_5": n_fp_5_5,
            "n_llm_sauve": n_llm_sauve,
        })

    recap = pd.DataFrame(rows).set_index("niveau")
    if verbose:
        print(recap.to_string(formatters={
            "pct_accord": "{:.1%}".format,
            "pct_correct": "{:.1%}".format,
        }))
    return recap


def plot_recap_multi_niveaux(recap, n_total):
    """
    Graphique empilé : pour chaque niveau, décomposition des observations en
    [accord correct | FP 5/5 | base faux mais LLM sauve |
     base faux, LLM diverge mais faux aussi | pas d'accord].
    """
    niveaux = recap.index.tolist()

    n_correct = recap["n_correct"].values
    n_fp_5_5 = recap["n_fp_5_5"].values
    n_llm_sauve = recap["n_llm_sauve"].values
    n_fp_autre = recap["n_fp_base"].values - n_fp_5_5 - n_llm_sauve
    n_pas_accord = n_total - recap["n_accord"].values

    pct_correct = n_correct / n_total * 100
    pct_fp_5_5 = n_fp_5_5 / n_total * 100
    pct_llm_sauve = n_llm_sauve / n_total * 100
    pct_fp_autre = n_fp_autre / n_total * 100
    pct_pas_accord = n_pas_accord / n_total * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    x = [str(n) for n in niveaux]

    bars = [
        ("Accord correct",                              pct_correct,    "#2ca02c"),
        ("FP 5/5 (base+LLM faux)",                      pct_fp_5_5,     "#d62728"),
        ("Base faux, LLM rattrape",                     pct_llm_sauve,  "#1f77b4"),
        ("Base faux, LLM diverge\nmais faux aussi",     pct_fp_autre,   "#ff7f0e"),
        ("Pas d'accord des 4 base",                     pct_pas_accord, "#bbbbbb"),
    ]

    bottom = [0] * len(x)
    for label, vals, color in bars:
        ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white")
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xlabel("Niveau de troncature COICOP")
    ax.set_ylabel("Part des observations (%)")
    ax.set_title("Décomposition des observations selon le niveau d'accord et la justesse")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    return fig


def stats_seul_par_division(df_seul, cols_pred, niveau_analyse=4, top_n=None, verbose=True):
    """
    Pour chaque classifieur, ventile les cas où il est seul à avoir raison
    (au niveau `niveau_analyse`) par division COICOP (niveau 1, les 2 premiers
    chiffres du vrai code).

    Parameters
    ----------
    df_seul : pd.DataFrame
        Sortie de stats_classifieur_seul_correct(niveau=niveau_analyse).
    cols_pred : list[str]
        Liste des classifieurs à analyser.
    niveau_analyse : int
        Niveau de troncature utilisé pour construire df_seul (pour l'affichage).
    top_n : int, optional
        Nombre de divisions à afficher par classifieur (les plus fréquentes).
        None = toutes.

    Returns
    -------
    pd.DataFrame croisé (divisions × classifieurs) avec effectifs et parts.
    Les parts (par classifieur et par division) ainsi que le détail par
    classifieur sont disponibles en `.attrs` : "parts_col", "parts_lig",
    "detail_par_classifieur" (dict {classifieur: DataFrame}).
    """
    df_only = df_seul[df_seul["seul_correct"]].copy()
    df_only["division"] = df_only["vrai_tronq"].map(
        lambda x: tronquer_niveau(x, niveau=1) if pd.notna(x) else None
    )

    # Tableau croisé effectifs : divisions × classifieur seul correct
    cross = pd.crosstab(df_only["division"], df_only["classifieur_seul"])
    # Réordonne les colonnes selon cols_pred (au cas où certains n'apparaissent pas)
    cross = cross.reindex(columns=cols_pred, fill_value=0)

    # Ajout des totaux
    cross["TOTAL"] = cross.sum(axis=1)
    cross.loc["TOTAL"] = cross.sum(axis=0)

    # Parts en colonne : pour chaque classifieur, quelle part de ses "sauvetages"
    # concerne chaque division ?
    parts_col = cross.drop(index="TOTAL").div(cross.loc["TOTAL"]).drop(columns="TOTAL") * 100

    # Parts en ligne : pour chaque division, quel classifieur sauve le plus ?
    parts_lig = cross.drop(columns="TOTAL").div(cross["TOTAL"], axis=0).drop(index="TOTAL") * 100

    # Détail par classifieur
    detail = {}
    for c in cols_pred:
        sub = df_only[df_only["classifieur_seul"] == c]
        if len(sub) == 0:
            detail[c] = pd.DataFrame(columns=["division", "n", "pct"])
            continue
        repart = sub["division"].value_counts()
        if top_n is not None:
            repart = repart.head(top_n)
        pct = repart / len(sub)  # fraction 0-1, cohérent avec les autres colonnes pct_*
        detail[c] = pd.DataFrame({"n": repart, "pct": pct}).rename_axis("division").reset_index()

    cross.attrs["parts_col"] = parts_col.round(1)
    cross.attrs["parts_lig"] = parts_lig.round(1)
    cross.attrs["detail_par_classifieur"] = detail

    if verbose:
        print(f"=== Répartition par division COICOP des cas 'un seul correct' (niveau {niveau_analyse}) ===\n")
        print("Effectifs :")
        print(cross.to_string())
        print("\nParts par classifieur (colonne, en %) :")
        print(parts_col.round(1).to_string())
        print("\nParts par division (ligne, en %) — 'qui sauve dans cette division ?' :")
        print(parts_lig.round(1).to_string())
        print(f"\n=== Détail par classifieur ===")
        for c in cols_pred:
            print(f"\n{c} — {len(detail[c])} division(s) :")
            print(detail[c].to_string(index=False, formatters={"pct": "{:.1%}".format}))

    return cross

