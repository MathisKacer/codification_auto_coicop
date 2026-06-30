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
    niveau 4 → "XX.X.X.X"  (sous-classe, granularité max)
"""
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Utilitaires de base
# =============================================================================

def tronquer_niveau(code, niveau=4):
    """
    Tronque un code COICOP au niveau demandé.

    Gère les NaN et les sentinels du preprocessing
    ("AUCUNE_SUGGESTION", "NON_CODABLE") qui sont préservés tels quels.
    """
    if pd.isna(code):
        return code
    s = str(code)
    if s in ("AUCUNE_SUGGESTION", "NON_CODABLE"):
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
    return "".join(out)


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