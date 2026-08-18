# Baseline + RF de confiance pour les codes COICOP

Outil de codification automatique des produits de l'enquête budget des
familles. Quatre classifieurs de base (LCS, RAG, RAG-ANN, TTC) proposent
chacun un code COICOP candidat par produit ; une règle de vote majoritaire
(la « baseline ») choisit un code, et un modèle de confiance (Random
Forest) estime la fiabilité de ce choix, pour décider automatiquement
quels produits peuvent être codés sans intervention humaine et lesquels
doivent repasser en reprise manuelle / LLM-judge.

C'est une alternative — moins interprétable mais construite exactement
sur le même schéma d'entrée/sortie — au modèle SIRUS de `sirus_stage/`
(mêmes runs d'entraînement, même format de décision en sortie).

## Comment ça marche

```
predictions.parquet (run S3, un produit par ligne)
  ├─ lcs_code, rag_code, ragann_code, ttc_code_1  (4 classifieurs de base)
        │
        ▼  baseline_majorite_ttc()  [fonctions_baseline.py]
  code majoritaire parmi les votants ; égalité ou aucun vote -> TTC arbitre
        │
        ▼  construire_features() + Pipeline (encodage + RF)  [modele_baseline_production.joblib]
  probabilité que ce code choisi soit correct
        │
        ▼  proba >= seuil_decision (0.6) ?
   ┌────┴────┐
   ▼         ▼
 "Code      "Reprise
 automatique" manuelle / LLM"
```

### Les 4 classifieurs de base

Ce ne sont **pas** des composants de ce dépôt : ce sont 4 systèmes amont
(ailleurs dans le pipeline de codification) qui déposent chacun un code
candidat par produit dans le run S3. Ce dépôt ne fait que lire leurs
colonnes de sortie :

| Classifieur | Colonnes en entrée | Fonctionnement (déduit du code) |
|---|---|---|
| **LCS** | `lcs_code`, `lcs_distance` | Rapprochement du libellé produit et de la nomenclature COICOP par similarité de chaîne de caractères — `lcs_distance` plus bas = candidat plus proche. |
| **RAG** | `rag_code`, `rag_confidence` | Classifieur basé sur un LLM avec retrieval (Retrieval-Augmented Generation). |
| **RAG-ANN** | `ragann_code`, `ragann_confidence` | Variante de RAG s'appuyant sur une recherche par plus proches voisins sur des embeddings (Approximate Nearest Neighbor) plutôt que sur le LLM. |
| **TTC** | `ttc_code_1`, `ttc_conf_1` | Classifieur neuronal fine-tuné ; seul le rang 1 est utilisé par ce pipeline. |

Si aucun des 4 classifieurs (TTC compris) n'a de code exploitable pour un
produit, la baseline renvoie `NaN` — ce produit part automatiquement en
reprise manuelle (cf. "Limites connues").

### Comment lire une décision

Contrairement à SIRUS (qui choisit *parmi* plusieurs codes candidats), la
baseline propose un seul code par construction (le vote majoritaire, TTC en
arbitre). La question du modèle de confiance est donc plus simple : *« ce
code unique proposé par la baseline est-il le bon ? »* — une probabilité
entre 0 et 1, obtenue par un Random Forest entraîné sur les signaux déjà
disponibles sans LLM (accord entre classifieurs, scores de confiance,
budget, enseigne, source). C'est cette probabilité (`proba` dans le CSV de
sortie) qui est comparée à `seuil_decision`.

## Prérequis : accès S3

Tous les scripts lisent/écrivent sur le bucket S3 `projet-budget-famille`
(endpoint `minio.lab.sspcloud.fr`). **Avant de lancer quoi que ce soit**,
vérifier que l'accès fonctionne :

```python
import s3fs
s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}).ls("projet-budget-famille")
```

Si ça échoue (erreur d'authentification), c'est très probablement que les
credentials du service SSP Cloud (variables d'environnement `AWS_*` /
`MC_HOST_s3`, injectées automatiquement par le service) ont expiré —
**relancer le service** SSP Cloud régénère un token frais. Ce n'est pas
quelque chose que ce dépôt peut corriger lui-même.

## Installation

```bash
pip install -r requirements.txt
```

Contrairement à SIRUS (package R nécessitant un correctif de compilation),
toutes les dépendances Python sont des paquets PyPI standards — pas de
patch, pas de compilation depuis les sources.

## Démarrage rapide

```bash
cd baseline_stage/
pip install -r requirements.txt
python evaluation_baseline.py     # mesure la performance + sauvegarde le modèle de prod
python production_baseline.py data/workflow_runs/<date>/codif-xxxxx/decide-coicop/predictions.parquet
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `requirements.txt` | Dépendances Python (`pip install -r requirements.txt`). |
| `fonctions_baseline.py` | Fonctions partagées entre évaluation et production (troncature des codes, baseline, construction des features, pipeline sklearn). **Toute correction doit être faite ici uniquement.** |
| `evaluation_baseline.py` | Charge les runs labellisés (vrai code connu), mesure la performance de la baseline seule puis du modèle de confiance (split 80/20), réentraîne un modèle final sur 100% des données, sauvegarde `modele_baseline_production.joblib`. À relancer dès qu'une nouvelle donnée labellisée est disponible. |
| `production_baseline.py` | Charge `modele_baseline_production.joblib` et l'applique à un nouveau run (vrai code inconnu). Produit un CSV de décisions. |
| `modele_baseline_production.joblib` | Modèle actuellement en production (généré par `evaluation_baseline.py`, compressé). Ne pas éditer à la main — relancer `evaluation_baseline.py` pour le régénérer. |

## Utilisation courante

### Réentraîner le modèle (nouvelles données labellisées disponibles)

1. Ajouter le chemin S3 du nouveau run dans `RUNS_LABELLISES` en tête de
   `evaluation_baseline.py`.
2. `python evaluation_baseline.py`
3. Vérifier dans la sortie :
   - L'accuracy de la baseline seule, puis l'accuracy/ROC AUC du modèle de
     confiance — comparer à l'exécution précédente pour détecter une
     régression.
   - La table de calibration (% de baseline réellement correcte par
     tranche de confiance) — sert à vérifier que `SEUIL_DECISION` (dans
     `production_baseline.py`) reste pertinent. Si la calibration a dérivé,
     ajuster ce seuil.
4. `modele_baseline_production.joblib` est régénéré automatiquement — rien
   d'autre à faire, `production_baseline.py` le rechargera à son prochain
   lancement.

### Coder un nouveau run

```bash
python production_baseline.py data/workflow_runs/<date>/codif-xxxxx/decide-coicop/predictions.parquet
```

Produit `<date>_codif-xxxxx_codes_baseline.csv` (identifiant unique par
run, pour ne jamais écraser le résultat d'un run précédent) avec une ligne
par produit d'entrée :

| Colonne | Sens |
|---|---|
| `ligne` | Identifiant du produit dans le run d'entrée. |
| `code_propose` | Code COICOP (niveau 4) choisi par la baseline (vote majoritaire, TTC arbitre). `NaN` si aucun classifieur (TTC compris) n'a de code exploitable. |
| `proba` | Confiance estimée par le modèle RF dans ce choix (entre 0 et 1). `NaN` si `code_propose` est `NaN` (rien à évaluer). |
| `decision` | `"Code automatique"` si `proba >= SEUIL_DECISION`, sinon `"Reprise manuelle / LLM"` (ou `"... (aucun candidat exploitable)"` si aucun classifieur n'a rien proposé). |

### Ajuster le seuil de décision

`SEUIL_DECISION` (dans `production_baseline.py`, actuellement `0.6`)
détermine l'arbitrage volume codé automatiquement / fiabilité. Le monter
réduit le volume auto-codé mais augmente sa fiabilité, et inversement. Sur
les 3 runs labellisés actuels (test 20%) :

| Seuil | Volume auto-codé | Fiabilité de l'auto-codé |
|---|---|---|
| 0.6 (actuel) | 63.5% | 96.9% |
| 0.7 | 57.0% | 97.9% |
| 0.8 | 48.6% | 98.6% |
| 0.9 | 37.4% | 99.2% |

À re-vérifier (table de calibration de `evaluation_baseline.py`) à chaque
réentraînement du modèle : la calibration peut dériver d'un run à l'autre.

## Limites connues

- Le modèle est entraîné sur les runs listés dans `RUNS_LABELLISES`
  (`evaluation_baseline.py`) — actuellement 3 runs de juin-juillet 2026 (les
  mêmes que ceux utilisés pour `sirus_stage/`, pour des résultats
  comparables entre les deux modèles). Plus il y a de données labellisées
  diverses, plus l'estimation de performance et la calibration sont
  fiables.
- Le modèle suppose les 4 classifieurs de base disponibles. Sur un run où
  un ou plusieurs d'entre eux n'ont rien produit, la baseline se rabat sur
  TTC (voire `NaN` si TTC lui-même est absent) plutôt que de forcer un
  choix peu fiable — comportement voulu, pas un bug.
- La calibration n'a été mesurée que sur la plage de confiance
  effectivement observée dans les données de test (pas de garantie en
  dehors).
- Contrairement à SIRUS, la décision de ce modèle n'est pas lisible sous
  forme de règles explicites (boîte noire à ~300 arbres) : les importances
  de features (`evaluation_baseline.py`) donnent une lecture globale de ce
  qui compte, mais pas de règle individuelle vérifiable pour un cas donné.
