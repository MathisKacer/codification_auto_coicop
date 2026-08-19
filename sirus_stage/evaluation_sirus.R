#!/usr/bin/env Rscript
# ============================================================================
# EVALUATION -- mesure la performance du modele SIRUS de re-ranking COICOP sur
# des donnees dont le vrai code est CONNU (accuracy candidat/produit-level,
# borne haute, calibration), puis sauvegarde un modele final -- reentraine
# sur 100% des donnees labellisees -- pret pour la mise en production
# (cf. production_sirus.R, qui charge ce fichier .rds).
#
# A executer : (a) pour mesurer l'accuracy actuelle avant/apres un
# changement (nouveau run labellise, hyperparametres...), (b) pour
# rafraichir le modele de production avec de nouvelles donnees labellisees.
#
# Usage : Rscript evaluation_sirus.R   (depuis le dossier stage/)
# ============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(sirus)
})
source("fonctions_sirus.R")

# --- Parametres ---------------------------------------------------------------
niveau         <- 4
num_regles     <- 10   # nombre de regles SIRUS (num.rule) -- cf. resultats_sirus_final.qmd section 5 (deja quasi-optimal)
profondeur_max <- 2    # profondeur max des arbres (max.depth)
seed_split     <- 42

bucket_s3   <- "projet-budget-famille"
endpoint_s3 <- "minio.lab.sspcloud.fr"

# Runs labellises disponibles (vrai code connu) -- ajouter ici toute nouvelle
# donnee labellisee au fur et a mesure qu'elle devient disponible, puis
# relancer ce script pour rafraichir le modele de production.
#
# Pour l'instant, entrainement sur un seul run -- les deux autres restent
# commentes ci-dessous, pour pouvoir tester `production_sirus.R` sur un run
# que le modele n'a jamais vu (generalisation), plutot que sur un echantillon
# issu du meme run que l'entrainement.
runs_labellises <- c(
  "data/workflow_runs/2026-06-29/codif-vvkv9/decide-coicop/predictions.parquet"
  # "data/workflow_runs/2026-07-23/codif-8m8jn/decide-coicop/predictions.parquet",
  # "data/workflow_runs/2026-07-30/codif-x98xl/decide-coicop/predictions.parquet"
)

chemin_modele_sortie <- "modele_sirus_production.rds"

# --- Chargement -----------------------------------------------------------------
charger_run_labellise <- function(objet_s3) {
  df <- aws.s3::s3read_using(
    FUN = arrow::read_parquet, object = objet_s3, bucket = bucket_s3,
    opts = list(region = "", base_url = endpoint_s3),
  )
  if (!"ligne" %in% names(df)) df$ligne <- seq_len(nrow(df)) - 1L
  stopifnot(
    "run sans colonne `code` -- pas de vrai code, ce script attend des donnees labellisees" =
      "code" %in% names(df)
  )
  # vrai code : `code_lvl4` si present (calcule en amont depuis 2026-07-30),
  # sinon troncature manuelle avec cascade -- verifie equivalentes a 100%
  # (cf. resultats_sirus_final.qmd section 1).
  df$vrai_n4 <- if ("code_lvl4" %in% names(df)) df$code_lvl4 else tronquer_niveau(df$code, niveau)
  df
}

cat("Chargement des runs labellisés...\n")
runs <- lapply(runs_labellises, charger_run_labellise)
for (i in seq_along(runs)) cat(sprintf("  %s : %d lignes\n", runs_labellises[i], nrow(runs[[i]])))

# Les runs doivent porter des observations disjointes -- sinon les accuracy
# ci-dessous seraient biaisees (meme produit compte deux fois, ou fuite
# train/test si le meme produit tombe des deux cotes du split). `id` est un
# UUID stable par produit, independant de `ligne` (indice relatif au run
# seulement) : verifie explicitement plutot que suppose.
if (all(vapply(runs, function(df) "id" %in% names(df), logical(1)))) {
  tous_ids <- unlist(lapply(runs, function(df) df$id))
  n_dup <- sum(duplicated(tous_ids))
  if (n_dup > 0) {
    stop(sprintf(
      "%d id(s) dupliqué(s) entre les runs labellisés -- au moins un produit
       apparaît dans plusieurs runs, à investiguer avant de poursuivre
       (les runs ne sont plus disjoints).", n_dup
    ))
  }
}

# --- Table long, un run a la fois puis empilees ---------------------------------
# `ligne` est un identifiant 0-indexe RELATIF a chaque run (pas un ID
# global) : prefixe par l'indice du run avant d'empiler, pour ne jamais
# confondre un produit d'un run avec un produit d'un autre run.
tables_long <- lapply(seq_along(runs), function(i) {
  tl <- construire_table_long(runs[[i]], niveau)
  tl$ligne <- paste0("run", i, "_", tl$ligne)
  tl
})
table_long <- bind_rows(tables_long) %>% mutate(code_candidat_n1 = as.factor(code_candidat_n1))

cat(sprintf(
  "\nTable long totale : %d lignes candidates (%d produits, %d run(s))\n",
  nrow(table_long), n_distinct(table_long$ligne), length(runs)
))

n_sans_candidat <- sum(vapply(seq_along(runs), function(i) {
  df_prefixe <- runs[[i]] %>% mutate(ligne = paste0("run", i, "_", ligne))
  length(produits_sans_candidat(df_prefixe, tables_long[[i]]))
}, integer(1)))
cat(sprintf("Produits sans aucun candidat proposé (tous runs, borne haute structurelle) : %d\n", n_sans_candidat))

# --- Split train/test (80/20 par produit) pour ESTIMER l'accuracy ---------------
set.seed(seed_split)
lignes_uniques <- unique(table_long$ligne)
lignes_train <- sample(lignes_uniques, size = round(0.8 * length(lignes_uniques)))
train <- table_long %>% filter(ligne %in% lignes_train)
test  <- table_long %>% filter(!ligne %in% lignes_train)

modele_eval <- sirus.fit(
  as.data.frame(train[, FEATURES_SIRUS]), train$correcte,
  type = "classif", num.rule = num_regles, max.depth = profondeur_max, verbose = FALSE,
)
cat("\nRègles retenues (modèle d'évaluation, entraîné sur 80%) :\n")
sirus.print(modele_eval)

test$proba <- sirus.predict(modele_eval, as.data.frame(test[, FEATURES_SIRUS]))
choix_par_produit <- test %>% group_by(ligne) %>% slice_max(proba, n = 1, with_ties = FALSE) %>% ungroup()
borne_haute <- test %>% group_by(ligne) %>% summarise(possible = max(correcte), .groups = "drop") %>% pull(possible) %>% mean()

cat("\n=== Accuracy estimée (test 20%, tous runs labellisés confondus) ===\n")
print(data.frame(
  Indicateur = c("Accuracy candidat-level", "Accuracy produit-level", "Borne haute"),
  Valeur = sprintf("%.1f%%", 100 * c(
    mean(as.integer(test$proba > 0.5) == test$correcte),
    mean(choix_par_produit$correcte),
    borne_haute
  ))
))

# Calibration : sert à choisir/vérifier `seuil_decision` dans
# production_sirus.R -- à revérifier après chaque réentraînement, la
# calibration peut dériver (cf. régression RAG-ANN, resultats_sirus_final.qmd
# section 4).
table_produit <- test %>%
  group_by(ligne) %>%
  arrange(desc(proba), .by_group = TRUE) %>%
  summarise(proba_gagnant = first(proba), sirus_correct = first(correcte), .groups = "drop")

cat("\n=== Calibration (test 20%) : % correct par tranche de confiance ===\n")
print(table_produit %>%
  mutate(tranche = cut(proba_gagnant, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)) %>%
  group_by(tranche) %>%
  summarise(n = n(), pct_correct = mean(sirus_correct), .groups = "drop") %>%
  mutate(pct_correct = sprintf("%.1f%%", pct_correct * 100)))

# --- Modele final de production : reentraine sur 100% des donnees labellisees --
# L'evaluation ci-dessus (split 80/20) sert a ESTIMER l'accuracy. Une fois
# cette estimation obtenue, il n'y a plus de raison de reserver 20% des
# donnees : le modele livre en production utilise tout ce qui est disponible.
cat("\nRéentraînement du modèle final sur 100% des données labellisées...\n")
modele_final <- sirus.fit(
  as.data.frame(table_long[, FEATURES_SIRUS]), table_long$correcte,
  type = "classif", num.rule = num_regles, max.depth = profondeur_max, verbose = FALSE,
)

bundle_production <- list(
  modele = modele_final,
  features = FEATURES_SIRUS,
  niveau = niveau,
  num_regles = num_regles,
  profondeur_max = profondeur_max,
  niveaux_code_candidat_n1 = levels(table_long$code_candidat_n1),
  accuracy_estimee = list(
    candidat = mean(as.integer(test$proba > 0.5) == test$correcte),
    produit = mean(choix_par_produit$correcte),
    borne_haute = borne_haute
  ),
  # Proba et correction du candidat retenu par produit (test 20%) : permet a
  # production_sirus.R d'estimer, pour N'IMPORTE QUEL seuil_decision (pas
  # seulement les tranches de la table de calibration ci-dessus), le volume
  # et la fiabilite attendus -- sans jamais voir le vrai code du nouveau run
  # a coder.
  calibration_test = list(
    proba = choix_par_produit$proba,
    correct = as.integer(choix_par_produit$correcte)
  ),
  runs_entrainement = runs_labellises,
  date_entrainement = as.character(Sys.Date())
)
saveRDS(bundle_production, chemin_modele_sortie)

cat(sprintf("\nModèle de production sauvegardé : %s\n", chemin_modele_sortie))
cat("(features, niveaux de code_candidat_n1 et accuracy estimée inclus dans le bundle -- utilisé par production_sirus.R)\n")
