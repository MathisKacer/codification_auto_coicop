# Re-ranking des codes COICOP avec SIRUS

Outil de codification automatique des produits de l'enquête budget des
familles. Quatre classifieurs de base (LCS, RAG, RAG-ANN, TTC) proposent
chacun un code COICOP candidat par produit ; un modèle SIRUS (règles
interprétables) choisit le meilleur candidat et estime sa fiabilité, pour
décider automatiquement quels produits peuvent être codés sans intervention
humaine et lesquels doivent repasser en reprise manuelle / LLM-judge.

## Comment ça marche

```
predictions.parquet (run S3, un produit par ligne)
  ├─ lcs_code     (LCS)
  ├─ rag_code     (RAG)
  ├─ ragann_code  (RAG-ANN)
  └─ ttc_code_1   (TTC)
        │
        ▼  construire_table_long()  [fonctions_sirus.R]
  table "long" : 1 ligne par (produit, code candidat proposé par ≥1 classifieur),
  avec qui a voté pour lui et avec quelle confiance
        │
        ▼  sirus.predict()  [modele_sirus_production.rds]
  probabilité de fiabilité pour chaque candidat
        │
        ▼  un seul candidat retenu par produit (le plus probable)  [production_sirus.R]
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
| **TTC** | `ttc_code_1/2/3`, `ttc_conf_1/2/3` | Renvoie jusqu'à 3 codes candidats classés avec une confiance chacun — seul le mieux classé (`ttc_code_1`/`ttc_conf_1`) est utilisé par ce pipeline. |

Un produit peut avoir jusqu'à 4 candidats (un par classifieur, dédupliqués
si plusieurs proposent le même code) — voire aucun si les 4 classifieurs
échouent, auquel cas le produit part automatiquement en reprise manuelle
(cf. "Limites connues").

### SIRUS : comment lire une règle

[SIRUS](https://cran.r-project.org/package=sirus) (Stable and Interpretable
RUle Set) entraîne une forêt d'arbres de décision peu profonds puis n'en
garde que les règles les plus stables/fréquentes, sous forme lisible. La
sortie de `evaluation_sirus.R` (`sirus.print`) affiche des lignes comme :

```
if conf_ragann < 0.9 then 0.134 (n=16219) else 0.898 (n=11869)
```

Se lit : parmi les `n` candidats (de l'échantillon d'entraînement)
vérifiant la condition, la proportion observée de candidats réellement
corrects est de 13.4% (resp. 89.8% pour ceux qui ne vérifient pas la
condition). La probabilité finale donnée à un candidat par
`sirus.predict()` est une moyenne sur toutes les règles qui s'appliquent à
lui — c'est ce nombre (`proba` dans le CSV de sortie) qui est comparé à
`seuil_decision`.

## Prérequis : accès S3

Tous les scripts lisent/écrivent sur le bucket S3 `projet-budget-famille`
(endpoint `minio.lab.sspcloud.fr`). **Avant de lancer quoi que ce soit**,
vérifier que l'accès fonctionne :

```r
aws.s3::get_bucket(bucket = "projet-budget-famille", region = "", base_url = "minio.lab.sspcloud.fr")
```

Si ça échoue (erreur d'authentification), c'est très probablement que les
credentials du service SSP Cloud (variables d'environnement `AWS_*` /
`MC_HOST_s3`, injectées automatiquement par le service) ont expiré —
**relancer le service** SSP Cloud régénère un token frais. Ce n'est pas
quelque chose que ce dépôt peut corriger lui-même.

## Installation

Le package `sirus` (0.3.3, CRAN) ne compile plus tel quel avec les
toolchains R/gcc récents (deux erreurs de compilation C++ : ambiguïté sur
`make_unique`, et macros `Rinternals.h` qui cassent des headers standard
inclus indirectement). `sirus_0.3.3_patched.tar.gz` contient un correctif
minimal (aucun changement de comportement du modèle) qui compile
normalement.

Ce patch corrige un problème observé sur *cet* environnement précis (R
4.6.0, gcc 13, SSP Cloud). Sur un autre poste/toolchain, essayer d'abord
`install.packages("sirus")` tel quel — le patch peut être superflu. S'il
échoue différemment, les deux correctifs (détaillés en commentaire dans
`src/utility.h` et `src/Makevars` du tarball) donnent le principe pour
l'adapter.

```bash
Rscript install_dependencies.R
```

installe tout (`arrow`, `dplyr`, `tidyr`, `aws.s3`, `ROCR`, `glmnet`,
`RcppEigen`, `randomForest`, `stringr`, puis `sirus` depuis le tarball
patché). Ce script est idempotent : relancez-le sans crainte, il ne
réinstalle que ce qui manque.

(`randomForest` et `stringr` ne sont utilisés que par
`resultats_sirus_final.qmd` — inutiles pour `evaluation_sirus.R` /
`production_sirus.R`. Pour re-générer `resultats_sirus_final.html` après une
modification du `.qmd`, il faut aussi [Quarto](https://quarto.org) installé,
puis `quarto render resultats_sirus_final.qmd`.)

**Sur SSP Cloud, ces packages ne persistent généralement pas entre deux
relances du service** — il faut relancer `install_dependencies.R` (compte
quelques minutes, compilation de `RcppEigen`/`glmnet` incluse) à chaque
nouvelle session de travail.

## Démarrage rapide

```bash
cd stage/
Rscript install_dependencies.R   # une fois par environnement/service (voir ci-dessus)
Rscript evaluation_sirus.R       # mesure l'accuracy + sauvegarde le modèle de prod
Rscript production_sirus.R data/workflow_runs/<date>/codif-xxxxx/decide-coicop/predictions.parquet
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `install_dependencies.R` | Installe tous les packages R nécessaires. À relancer à chaque nouvel environnement (voir "Installation" ci-dessus). |
| `fonctions_sirus.R` | Fonctions partagées entre évaluation et production (troncature des codes, construction de la table candidats/votes). **Toute correction du pipeline doit être faite ici uniquement** — voir les commentaires en tête de fichier pour les pièges déjà identifiés. |
| `evaluation_sirus.R` | Charge les runs labellisés (vrai code connu), mesure l'accuracy (split 80/20), réentraîne un modèle final sur 100% des données, sauvegarde `modele_sirus_production.rds`. À relancer dès qu'une nouvelle donnée labellisée est disponible. |
| `production_sirus.R` | Charge `modele_sirus_production.rds` et l'applique à un nouveau run (vrai code inconnu). Produit un CSV de décisions. |
| `modele_sirus_production.rds` | Modèle actuellement en production (généré par `evaluation_sirus.R`). Ne pas éditer à la main — relancer `evaluation_sirus.R` pour le régénérer. |
| `sirus_0.3.3_patched.tar.gz` | Source du package `sirus` corrigée pour compiler avec les toolchains R/gcc récents (voir "Installation"). C'est le seul tarball du dossier — c'est celui-ci qu'il faut installer. |
| `stage.Rproj` | Projet RStudio, pour ouvrir le dossier directement dans RStudio. |
| `resultats_sirus_final.qmd` / `.html` | Document de référence unique : résultats obtenus et démarche méthodologique complète (modèle, généralisation à plusieurs runs, hyperparamètres, régression RAG-ANN, état actuel du pipeline de production). Ouvrir le `.html` pour lire sans avoir besoin de R. À consulter si une décision du pipeline semble surprenante — la raison y est probablement déjà documentée. Pas nécessaire pour l'usage courant (évaluer/coder un run). |

## Utilisation courante

### Réentraîner le modèle (nouvelles données labellisées disponibles)

1. Ajouter le chemin S3 du nouveau run dans `runs_labellises` en tête de
   `evaluation_sirus.R`.
2. `Rscript evaluation_sirus.R`
3. Vérifier dans la sortie :
   - Les 3 indicateurs d'accuracy (candidat-level, produit-level, borne
     haute) — comparer à l'exécution précédente pour détecter une
     régression.
   - La table de calibration (% correct par tranche de confiance) — sert à
     vérifier que `seuil_decision` (dans `production_sirus.R`) reste
     pertinent. Si la calibration a dérivé, ajuster ce seuil.
4. `modele_sirus_production.rds` est régénéré automatiquement — rien
   d'autre à faire, `production_sirus.R` le rechargera à son prochain
   lancement.

### Coder un nouveau run

```bash
Rscript production_sirus.R data/workflow_runs/<date>/codif-xxxxx/decide-coicop/predictions.parquet
```

Produit `<date>_codif-xxxxx_codes_sirus.csv` (identifiant unique par run,
pour ne jamais écraser le résultat d'un run précédent) avec une ligne par
produit d'entrée :

| Colonne | Sens |
|---|---|
| `ligne` | Identifiant du produit dans le run d'entrée. |
| `code_propose` | Code COICOP (niveau 4) choisi par SIRUS parmi les candidats des 4 classifieurs. `NA` si aucun classifieur n'a proposé de candidat exploitable. |
| `proba` | Confiance estimée par SIRUS dans ce choix (entre 0 et 1). |
| `decision` | `"Code automatique"` si `proba >= seuil_decision`, sinon `"Reprise manuelle / LLM"` (ou `"... (aucun candidat exploitable)"` si aucun classifieur n'a rien proposé). |

Le script signale explicitement (sans les masquer) :
- les catégories `code_candidat_n1` absentes du modèle de production (candidat
  non scorable, exclu),
- les produits sans aucun candidat proposé par les 4 classifieurs.

### Ajuster le seuil de décision

`seuil_decision` (dans `production_sirus.R`, actuellement `0.6`) détermine
l'arbitrage volume codé automatiquement / fiabilité. Le monter réduit le
volume auto-codé mais augmente sa fiabilité, et inversement. À re-vérifier
contre la table de calibration à chaque réentraînement du modèle (la
calibration peut dériver — cf. régression RAG-ANN documentée dans
`resultats_sirus_final.qmd`, section 3).

## Limites connues

- Le modèle est entraîné sur les runs listés dans `runs_labellises`
  (`evaluation_sirus.R`) — actuellement 3 runs de juin-juillet 2026. Plus il y
  a de données labellisées diverses, plus l'estimation d'accuracy et la
  calibration sont fiables.
- Le modèle suppose les 4 classifieurs de base disponibles. Sur un run où
  RAG et/ou RAG-ANN n'ont rien produit (ex. run antérieur à leur intégration
  au pipeline), SIRUS bascule structurellement la quasi-totalité du volume en
  reprise manuelle plutôt que de forcer un choix peu fiable — comportement
  voulu, pas un bug.
- La calibration n'a été mesurée que sur la plage de confiance effectivement
  observée dans les données de test (pas de garantie en dehors).
