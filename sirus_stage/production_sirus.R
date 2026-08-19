#!/usr/bin/env Rscript
# ============================================================================
# PRODUCTION -- applique le modèle SIRUS sauvegardé (cf. evaluation_sirus.R)
# à de NOUVELLES lignes dont le vrai code COICOP est INCONNU, pour produire
# une décision par produit : code automatique (avec une fiabilité estimée
# par la calibration du modèle), ou reprise manuelle / LLM-judge.
#
# Ne calcule et n'utilise AUCUN vrai code, même si une colonne `code` existe
# par ailleurs dans le run (cf. fonctions_sirus.R::construire_table_long) --
# pour mesurer une accuracy, voir evaluation_sirus.R.
#
# Usage : Rscript production_sirus.R <chemin_s3_du_run_a_coder>
#   ex.  : Rscript production_sirus.R data/workflow_runs/2026-08-15/codif-xxxxx/decide-coicop/predictions.parquet
# ============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(sirus)
})
source("fonctions_sirus.R")

# --- Parametres -----------------------------------------------------------------
bucket_s3     <- "projet-budget-famille"
endpoint_s3   <- "minio.lab.sspcloud.fr"
chemin_modele <- "modele_sirus_production.rds"

# Seuil de décision sur `proba_gagnant` : au-dessus, on accepte le choix de
# SIRUS (code automatique) ; en dessous, reprise manuelle / LLM-judge plutôt
# que de forcer un choix peu fiable. 0.6 est le point de départ établi sur
# le run 8m8jn (test.qmd : 63% du volume auto-codé à ~80% de fiabilité) --
# À RE-VÉRIFIER (section calibration de evaluation_sirus.R) à chaque
# réentraînement du modèle : la calibration peut dériver.
seuil_decision <- 0.3

# Estime, à partir du test 20% mis de côté par evaluation_sirus.R (pas du run
# à coder -- son vrai code est inconnu), le volume et la fiabilité attendus
# si on règle seuil_decision à `seuil`. Fonctionne pour n'importe quel seuil,
# pas seulement les tranches de la table de calibration affichée par
# evaluation_sirus.R.
estimer_fiabilite_au_seuil <- function(bundle, seuil) {
  if (is.null(bundle$calibration_test)) {
    return(NULL)  # bundle genere par une version anterieure du script
  }
  proba <- bundle$calibration_test$proba
  correct <- bundle$calibration_test$correct
  confiant <- proba >= seuil
  if (!any(confiant)) {
    return(list(volume = 0, fiabilite = NaN, n = 0L))
  }
  list(
    volume = mean(confiant),
    fiabilite = mean(correct[confiant]),
    n = sum(confiant)
  )
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage : Rscript production_sirus.R <chemin_s3_du_run_a_coder>")
}
objet_s3_a_coder <- args[[1]]

# --- Chargement du modèle --------------------------------------------------------
if (!file.exists(chemin_modele)) {
  stop(sprintf("Modèle introuvable (%s) -- exécuter d'abord evaluation_sirus.R.", chemin_modele))
}
bundle <- readRDS(chemin_modele)
cat(sprintf(
  "Modèle chargé (entraîné le %s sur %d run(s), accuracy produit ESTIMÉE %.1f%% -- cf. evaluation_sirus.R)\n",
  bundle$date_entrainement, length(bundle$runs_entrainement), 100 * bundle$accuracy_estimee$produit
))

estimation <- estimer_fiabilite_au_seuil(bundle, seuil_decision)
if (is.null(estimation)) {
  cat(
    "(Estimation du volume/fiabilité au seuil actuel indisponible -- modèle ",
    "généré par une version antérieure d'evaluation_sirus.R, relancer ",
    "evaluation_sirus.R pour la régénérer.)\n", sep = ""
  )
} else {
  cat(sprintf(
    paste0(
      "À seuil_decision = %s : sur le test de evaluation_sirus.R, ça ",
      "correspondrait à %.1f%% du volume codé automatiquement (%d cas), ",
      "avec une fiabilité estimée de %.1f%% (donc ~%.1f%% de faux positifs ",
      "attendus parmi ce volume) -- estimation basée sur le run ",
      "d'entraînement, pas sur le run ci-dessous.\n"
    ),
    seuil_decision, 100 * estimation$volume, estimation$n,
    100 * estimation$fiabilite, 100 * (1 - estimation$fiabilite)
  ))
}

# --- Chargement des nouvelles données (vrai code INCONNU, jamais utilisé) -------
df <- aws.s3::s3read_using(
  FUN = arrow::read_parquet, object = objet_s3_a_coder, bucket = bucket_s3,
  opts = list(region = "", base_url = endpoint_s3),
)
if (!"ligne" %in% names(df)) df$ligne <- seq_len(nrow(df)) - 1L
stopifnot("des identifiants `ligne` en double rendraient le résultat ambigu" = !any(duplicated(df$ligne)))

cat(sprintf("Run à coder : %d lignes\n", nrow(df)))

# Pas de colonne `vrai_n4` fournie ici -> construire_table_long() calcule
# `correcte = NA` partout (mode production, cf. fonctions_sirus.R).
table_long <- construire_table_long(df, niveau = bundle$niveau) %>%
  mutate(code_candidat_n1 = factor(code_candidat_n1, levels = bundle$niveaux_code_candidat_n1))

n_categorie_inconnue <- sum(is.na(table_long$code_candidat_n1))
if (n_categorie_inconnue > 0) {
  warning(sprintf(
    "%d ligne(s) candidates ont une catégorie code_candidat_n1 absente du modèle de production -- proba non calculable, exclues (produits concernés basculés en reprise manuelle plus bas).",
    n_categorie_inconnue
  ))
  table_long <- table_long %>% filter(!is.na(code_candidat_n1))
}

# --- Scoring et choix du meilleur candidat par produit ---------------------------
table_long$proba <- sirus.predict(bundle$modele, as.data.frame(table_long[, bundle$features]))

resultats <- table_long %>%
  group_by(ligne) %>%
  slice_max(proba, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    ligne,
    code_propose = code_candidat,
    proba,
    decision = if_else(proba >= seuil_decision, "Code automatique", "Reprise manuelle / LLM"),
  )

# --- Produits sans aucun candidat exploitable : ne PEUVENT PAS être codés
# automatiquement, et disparaîtraient silencieusement de `resultats` sans ce
# contrôle explicite (aucune ligne candidate -> absents de `table_long`).
lignes_manquantes <- setdiff(unique(df$ligne), resultats$ligne)
if (length(lignes_manquantes) > 0) {
  resultats <- bind_rows(
    resultats,
    data.frame(
      ligne = lignes_manquantes, code_propose = NA_character_, proba = NA_real_,
      decision = "Reprise manuelle / LLM (aucun candidat exploitable)"
    )
  )
}

stopifnot(
  "chaque produit d'entrée doit apparaître exactement une fois en sortie" =
    setequal(resultats$ligne, unique(df$ligne)) && !any(duplicated(resultats$ligne))
)

# --- Bilan et sauvegarde ----------------------------------------------------------
cat("\n=== Bilan de la décision ===\n")
print(resultats %>%
  group_by(decision) %>%
  summarise(n = n(), pct_volume = n() / nrow(resultats), .groups = "drop") %>%
  mutate(pct_volume = sprintf("%.1f%%", pct_volume * 100)))

# Chaque run s'appelle "predictions.parquet" (seul le dossier parent change,
# cf. data/workflow_runs/<date>/codif-xxxxx/decide-coicop/predictions.parquet)
# -- baser le nom de sortie sur basename() collisionnerait silencieusement
# entre deux runs distincts et écraserait le CSV precedent. On identifie le
# run par <date>_<codif-xxxxx> ; si le chemin ne suit pas ce format connu, on
# retombe sur le chemin complet aplati (jamais de collision possible).
identifiant_run <- local({
  segments <- strsplit(objet_s3_a_coder, "/")[[1]]
  i_codif <- grep("^codif-", segments)
  if (length(i_codif) == 1 && i_codif >= 2) {
    paste(segments[i_codif - 1], segments[i_codif], sep = "_")
  } else {
    gsub("[/.]", "_", sub("\\.parquet$", "", objet_s3_a_coder))
  }
})
chemin_sortie <- paste0(identifiant_run, "_codes_sirus.csv")
write.csv(resultats, chemin_sortie, row.names = FALSE)
cat(sprintf("\nRésultats écrits dans %s (%d lignes, une par produit d'entrée)\n", chemin_sortie, nrow(resultats)))
