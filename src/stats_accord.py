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
import io
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.coicop import tronquer_niveau


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

def stats_accord(df, cols_pred, col_vrai, niveau=4):
    """
    Stats globales sur les cas d'accord unanime au niveau donné :
    proportion d'accord, dont corrects vs faux positifs.

    Returns
    -------
    pd.DataFrame enrichi des colonnes 'code_consensus', 'tous_accord',
    'vrai_tronq', et une colonne tronquée par classifieur (`{col}_tronq`).
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

    print(f"=== Accord unanime au niveau {niveau} ({len(cols_pred)} classifieurs) ===")
    print(f"Total observations              : {n_total}")
    print(f"Accord unanime                  : {n_accord:>6}  ({n_accord/n_total:.1%})")
    print(f"  ├─ corrects                   : {n_correct:>6}  ({n_correct/max(n_accord,1):.1%} des accords)")
    print(f"  └─ faux positifs (FP)         : {n_fp:>6}  ({n_fp/max(n_accord,1):.1%} des accords)")
    print(f"Taux de FP / total              : {n_fp/n_total:.1%}")
    return out


def analyse_faux_positifs(df_stats, niveau=4, top_n=15):
    """
    Sur les FP (accord unanime mais code faux), regarde :
      - les vrais codes (niveau donné) les plus concernés
      - les codes prédits à tort
      - les confusions vrai → prédit les plus fréquentes
      - si dispo : la division COICOP (niveau 1) concernée

    À appeler sur la sortie de stats_accord().
    """
    df_fp = df_stats[
        df_stats["tous_accord"]
        & (df_stats["code_consensus"] != df_stats["vrai_tronq"])
    ].copy()

    print(f"\n=== {len(df_fp)} faux positifs ===\n")

    print(f"→ Top vrais codes (niveau {niveau}) parmi les FP :")
    print(df_fp["vrai_tronq"].value_counts().head(top_n).to_string())

    print("\n→ Top codes prédits à tort (consensus erroné) :")
    print(df_fp["code_consensus"].value_counts().head(top_n).to_string())

    print("\n→ Top confusions (vrai → prédit) :")
    confusions = (
        df_fp.groupby(["vrai_tronq", "code_consensus"])
        .size()
        .sort_values(ascending=False)
    )
    print(confusions.head(top_n).to_string())

    # Vue agrégée à la division (niveau 1)
    if niveau > 1:
        df_fp["vrai_div"] = df_fp["vrai_tronq"].str[:2]
        df_fp["pred_div"] = df_fp["code_consensus"].str[:2]
        print("\n→ Confusions agrégées au niveau 1 (division) :")
        conf_div = (
            df_fp.groupby(["vrai_div", "pred_div"])
            .size()
            .sort_values(ascending=False)
        )
        print(conf_div.head(top_n).to_string())

    return df_fp


# =============================================================================
# Dissociation selon le comportement du LLM
# =============================================================================

def stats_accord_avec_llm(df, cols_base, col_llm, col_vrai, niveau=4):
    """
    Sur les cas où les classifieurs de base sont unanimes au niveau donné,
    dissocie selon que le LLM suit ce consensus ou non, et regarde la justesse.

    Returns
    -------
    pd.DataFrame restreint aux lignes d'accord unanime des classifieurs de base,
    enrichi de 'code_consensus', 'vrai_tronq', 'llm_tronq', 'llm_suit',
    'consensus_correct', 'llm_correct'.
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

    print(f"=== Accord unanime des {len(cols_base)} classifieurs de base (niveau {niveau}) : {n_acc} cas ===\n")
    print(f"┌─ LLM SUIT le consensus           : {n_suit:>6}  ({n_suit/n_acc:.1%})")
    print(f"│    ├─ tout le monde a raison     : {n_suit_ok:>6}  ({n_suit_ok/max(n_suit,1):.1%})")
    print(f"│    └─ FP partagés (5/5 faux)     : {n_suit_fp:>6}  ({n_suit_fp/max(n_suit,1):.1%})")
    print(f"│")
    print(f"└─ LLM NE SUIT PAS le consensus    : {n_nosuit:>6}  ({n_nosuit/n_acc:.1%})")
    print(f"     ├─ base a raison, LLM tort    : {n_base_ok:>6}  ({n_base_ok/max(n_nosuit,1):.1%})")
    print(f"     ├─ LLM a raison, base tort    : {n_llm_ok:>6}  ({n_llm_ok/max(n_nosuit,1):.1%})")
    print(f"     └─ personne n'a raison        : {n_aucun_ok:>6}  ({n_aucun_ok/max(n_nosuit,1):.1%})")

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
    print("\n", recap.to_string(index=False))

    return df_acc


# =============================================================================
# Cas où un seul classifieur a raison contre tous les autres
# =============================================================================

def stats_classifieur_seul_correct(df, cols_pred, col_vrai, niveau=4):
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

    print(f"=== Cas où UN SEUL classifieur sur {len(cols_pred)} a raison (niveau {niveau}) ===")
    print(f"Total observations           : {n_total}")
    print(f"Un seul correct              : {n_seul}  ({n_seul/n_total:.1%})\n")

    print("→ Répartition par classifieur sauveur :")
    repart = out.loc[out["seul_correct"], "classifieur_seul"].value_counts()
    repart_pct = (repart / max(n_seul, 1) * 100).round(1)
    print(pd.DataFrame({"n": repart, "pct": repart_pct}).to_string())

    return out


def analyse_classifieur_seul(df_seul, classifieur, top_n=15):
    """
    Codes sur lesquels `classifieur` a raison quand tous les autres ont tort.
    """
    sub = df_seul[df_seul["classifieur_seul"] == classifieur]
    print(f"\n=== {classifieur} seul correct : {len(sub)} cas ===")
    print(sub["vrai_tronq"].value_counts().head(top_n).to_string())
    return sub


# =============================================================================
# Wrappers multi-niveaux
# =============================================================================

def stats_accord_multi_niveaux(df, cols_base, col_llm, col_vrai, niveaux=(1, 2, 3, 4)):
    """
    Lance stats_accord puis stats_accord_avec_llm pour chaque niveau demandé.

    Returns
    -------
    dict {niveau: {"df_stats": ..., "df_acc": ...}}
    """
    resultats = {}
    for n in niveaux:
        print("\n" + "=" * 70)
        print(f"  NIVEAU {n}")
        print("=" * 70)
        df_stats = stats_accord(df, cols_base, col_vrai, niveau=n)
        print()
        df_acc = stats_accord_avec_llm(df, cols_base, col_llm, col_vrai, niveau=n)
        resultats[n] = {"df_stats": df_stats, "df_acc": df_acc}
    return resultats


def stats_seul_multi_niveaux(df, cols_tous, col_vrai, niveaux=(1, 2, 3, 4)):
    """
    Lance stats_classifieur_seul_correct pour chaque niveau.

    Returns
    -------
    dict {niveau: df_seul}
    """
    resultats = {}
    for n in niveaux:
        print("\n" + "=" * 70)
        print(f"  NIVEAU {n}")
        print("=" * 70)
        df_seul = stats_classifieur_seul_correct(df, cols_tous, col_vrai, niveau=n)
        resultats[n] = df_seul
    return resultats


def rapport_complet_multi_niveaux(df, cols_base, col_llm, col_vrai,
                                  niveaux=(1, 2, 3, 4), top_n=10):
    """
    Rapport détaillé complet pour chaque niveau :
    accord global + analyse des FP + dissociation LLM.

    Returns
    -------
    dict {niveau: {"df_stats": ..., "df_fp": ..., "df_acc": ...}}
    """
    resultats = {}
    for n in niveaux:
        print("\n" + "█" * 70)
        print(f"  NIVEAU {n}")
        print("█" * 70)
        df_stats = stats_accord(df, cols_base, col_vrai, niveau=n)
        print()
        df_fp = analyse_faux_positifs(df_stats, niveau=n, top_n=top_n)
        print()
        df_acc = stats_accord_avec_llm(df, cols_base, col_llm, col_vrai, niveau=n)
        resultats[n] = {"df_stats": df_stats, "df_fp": df_fp, "df_acc": df_acc}
    return resultats


# =============================================================================
# Récap + dataviz multi-niveaux
# =============================================================================

def recap_multi_niveaux(df, cols_base, col_llm, col_vrai, niveaux=(1, 2, 3, 4)):
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
    plt.show()


def stats_seul_par_division(df_seul, cols_pred, niveau_analyse=4, top_n=None):
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

    print(f"=== Répartition par division COICOP des cas 'un seul correct' (niveau {niveau_analyse}) ===\n")
    print("Effectifs :")
    print(cross.to_string())

    # Parts en colonne : pour chaque classifieur, quelle part de ses "sauvetages"
    # concerne chaque division ?
    parts_col = cross.drop(index="TOTAL").div(cross.loc["TOTAL"]).drop(columns="TOTAL") * 100
    print("\nParts par classifieur (colonne, en %) :")
    print(parts_col.round(1).to_string())

    # Parts en ligne : pour chaque division, quel classifieur sauve le plus ?
    parts_lig = cross.drop(columns="TOTAL").div(cross["TOTAL"], axis=0).drop(index="TOTAL") * 100
    print("\nParts par division (ligne, en %) — 'qui sauve dans cette division ?' :")
    print(parts_lig.round(1).to_string())

    # Détail par classifieur avec libellés indicatifs
    print(f"\n=== Détail par classifieur ===")
    for c in cols_pred:
        sub = df_only[df_only["classifieur_seul"] == c]
        if len(sub) == 0:
            print(f"\n{c} : aucun cas")
            continue
        print(f"\n{c} — {len(sub)} cas au total :")
        repart = sub["division"].value_counts()
        if top_n is not None:
            repart = repart.head(top_n)
        pct = (repart / len(sub) * 100).round(1)
        tab = pd.DataFrame({"n": repart, "pct": pct.astype(str) + "%"})
        print(tab.to_string())

    return cross


RACINE_PROJET = Path(__file__).resolve().parent.parent


def rapport_html(df, cols_base, col_llm, col_vrai, cols_tous,
                 niveaux=(1, 2, 3, 4), top_n=10,
                 col_libelle="l_pr_product",
                 chemin_sortie=None):
    """
    Génère un rapport HTML des stats descriptives.

    Par défaut, écrit dans <racine du projet>/outputs/rapport_stats_accord.html,
    quel que soit le répertoire de travail courant (évite de créer un dossier
    outputs/ à côté du notebook si celui-ci n'est pas lancé depuis la racine).
    """
    if chemin_sortie is None:
        chemin_sortie = RACINE_PROJET / "outputs" / "rapport_stats_accord.html"
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    # --- Capture des fonctions verbeuses ---
    buf_complet = io.StringIO()
    with redirect_stdout(buf_complet):
        res_complet = rapport_complet_multi_niveaux(
            df, cols_base, col_llm, col_vrai, niveaux=niveaux, top_n=top_n,
        )
    txt_complet = buf_complet.getvalue()

    buf_seul = io.StringIO()
    with redirect_stdout(buf_seul):
        res_seul = stats_seul_multi_niveaux(df, cols_tous, col_vrai, niveaux=niveaux)
    txt_seul = buf_seul.getvalue()

    buf_division = io.StringIO()
    with redirect_stdout(buf_division):
        stats_seul_par_division(res_seul[4], cols_tous, niveau_analyse=4, top_n=None)
    txt_division = buf_division.getvalue()

    # --- Récap synthétique ---
    recap = recap_multi_niveaux(df, cols_base, col_llm, col_vrai, niveaux=niveaux)
    recap_fmt = recap.copy()
    recap_fmt["pct_accord"] = recap_fmt["pct_accord"].map("{:.1%}".format)
    recap_fmt["pct_correct"] = recap_fmt["pct_correct"].map("{:.1%}".format)
    html_recap = recap_fmt.to_html(classes="pandas", border=0)

    # --- Focus niveau 4 ---
    df_stats_n4 = res_complet[4]["df_stats"]
    df_fp_n4 = df_stats_n4[
        df_stats_n4["tous_accord"]
        & (df_stats_n4["code_consensus"] != df_stats_n4["vrai_tronq"])
    ]
    cols_fp = [c for c in [col_libelle, "code", "vrai_tronq", "code_consensus",
                            *[f"{c}_tronq" for c in cols_base]] if c in df_fp_n4.columns]
    html_fp = df_fp_n4[cols_fp].to_html(classes="pandas", border=0, index=False)

    df_seul_n4 = res_seul[4]
    df_seul_only = df_seul_n4[df_seul_n4["seul_correct"]]
    cols_seul = [c for c in [col_libelle, "code", "vrai_tronq", "classifieur_seul",
                             *[f"{c}_tronq" for c in cols_tous]] if c in df_seul_only.columns]
    html_seul = df_seul_only[cols_seul].to_html(classes="pandas", border=0, index=False)

    # --- Assemblage HTML ---
    date = datetime.now().strftime("%d/%m/%Y %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport stats accord — codification COICOP</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1200px; margin: 2em auto; padding: 0 2em;
    color: #222; line-height: 1.5;
  }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
  h2 {{ color: #1f4e79; margin-top: 2em;
        border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
  h3 {{ color: #555; }}
  pre {{
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
    background: #f5f5f5; padding: 1em; border-radius: 6px;
    overflow-x: auto; font-size: 0.85em; line-height: 1.3;
    white-space: pre;
  }}
  table.pandas {{
    border-collapse: collapse; margin: 1em 0; font-size: 0.9em;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
  table.pandas th, table.pandas td {{
    padding: 6px 12px; text-align: left; border-bottom: 1px solid #eee;
  }}
  table.pandas th {{ background: #1f4e79; color: white; }}
  table.pandas tr:hover {{ background: #f9f9f9; }}
  .meta {{ color: #888; font-size: 0.9em; }}
  .toc {{ background: #f0f4f8; padding: 1em 2em; border-radius: 6px; }}
  .toc a {{ text-decoration: none; color: #1f4e79; }}
  .table-wrapper {{ max-height: 500px; overflow-y: auto;
                    border: 1px solid #ddd; border-radius: 6px; }}
</style>
</head>
<body>

<h1>Rapport stats accord — codification COICOP</h1>
<p class="meta">Généré le {date} — {len(df)} observations — niveaux : {list(niveaux)}</p>

<div class="toc">
  <strong>Sommaire</strong>
  <ul>
    <li><a href="#recap">1. Récap synthétique multi-niveaux</a></li>
    <li><a href="#detail">2. Rapport détaillé par niveau</a></li>
    <li><a href="#seul">3. Cas où un seul classifieur a raison (par niveau)</a></li>
    <li><a href="#focus4">4. Focus niveau 4 : lignes concernées</a></li>
    <li><a href="#division">5. Ventilation par division COICOP (niveau 4)</a></li>
  </ul>
</div>

<h2 id="recap">1. Récap synthétique multi-niveaux</h2>
{html_recap}

<h2 id="detail">2. Rapport détaillé par niveau</h2>
<pre>{txt_complet}</pre>

<h2 id="seul">3. Cas où un seul classifieur a raison (par niveau)</h2>
<pre>{txt_seul}</pre>

<h2 id="focus4">4. Focus niveau 4 : lignes concernées</h2>

<h3>Faux positifs unanimes (les 4 base d'accord mais code faux) — {len(df_fp_n4)} lignes</h3>
<div class="table-wrapper">
{html_fp}
</div>

<h3>Un seul classifieur a raison — {len(df_seul_only)} lignes</h3>
<div class="table-wrapper">
{html_seul}
</div>

<h2 id="division">5. Ventilation par division COICOP — cas 'un seul correct' au niveau 4</h2>
<pre>{txt_division}</pre>

</body>
</html>"""

    with open(chemin_sortie, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Rapport écrit : {chemin_sortie}")
    return chemin_sortie