# codification_auto_coicop

Travaux de stage explorant des alternatives moins coûteuses au LLM-as-a-judge
utilisé dans le pipeline de [codification automatique de la nomenclature
COICOP](https://inseefrlab.github.io/codif-coicop-bdf/) (enquête Budget de
Famille, INSEE) : une baseline (vote majoritaire des classifieurs de base) et
sa Random Forest de confiance d'un côté, **SIRUS** (règles de décision
lisibles) de l'autre, comparées entre elles pour savoir quand on peut se
passer d'un appel au LLM.

**Site du stage** (résultats détaillés, run par run) : publié automatiquement
sur GitHub Pages à chaque push sur `main` (voir
[.github/workflows/publish.yml](.github/workflows/publish.yml)).

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| [`site/`](site/) | Source Quarto du site publié : fiabilité des classifieurs, modélisation (baseline+RF, SIRUS, comparaison), interprétation figée run par run. |
| [`src/`](src/) | Modules Python partagés (`baseline/` : pipeline RF ; `coicop.py` : troncature/libellés de la nomenclature ; `stats_accord.py`, `rapport_utils.py`). |
| [`data/`](data/) | Chargement des données depuis le S3 (SSPCloud/MinIO) et chemins des runs utilisés par le site. |
| [`sirus_final/`](sirus_final/) | Pipeline SIRUS autonome, prêt pour la production (entraînement, évaluation, scoring de nouveaux runs) — voir son [README](sirus_final/README.md). |
| [`baseline_final/`](baseline_final/) | Même chose pour la baseline + Random Forest — voir son [README](baseline_final/README.md). |
| [`notebooks/`](notebooks/) | Notebooks de recherche (exploration, généralisation à de nouveaux runs, brouillons antérieurs à `site/`). |
| [`rapport/`](rapport/) | Rapport de stage (LaTeX / Quarto). |
| [`rapport_stat_des/`](rapport_stat_des/) | Générateur autonome du rapport de statistiques descriptives (hors site, rejouable sur un autre run via `CHEMIN_S3_STATS_DESCRIPTIVES`). |
| [`annotation_erreur_rf.ods`](annotation_erreur_rf.ods) | Source de la réannotation manuelle des cas manqués par la RF (cf. [interprétation](site/interpretation/2026-06-29.qmd)). |

## Installation

Dépendances Python (site, `src/`, notebooks) :

```bash
pip install -r requirements.txt
```

Les pages du site en R (`site/modelisation/sirus.qmd`,
`site/modelisation/comparaison.qmd`) et le pipeline `sirus_final/` nécessitent
en plus R et le package `sirus` **depuis le tarball patché du dépôt** (la
version CRAN ne compile plus avec les toolchains récents) :

```bash
cd sirus_final/
Rscript install_dependencies.R
```

L'accès aux données (S3 `projet-budget-famille` sur `minio.lab.sspcloud.fr`)
nécessite des identifiants AWS valides dans l'environnement
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` le cas
échéant).

## Rendre le site en local

```bash
cd site/
quarto render
```

Le rendu s'appuie sur le cache *freeze* de Quarto (`_freeze/`) : les pages non
modifiées ne sont pas ré-exécutées, donc pas besoin d'accès S3/R tant qu'on ne
touche pas à leurs fichiers `.qmd` sources.
