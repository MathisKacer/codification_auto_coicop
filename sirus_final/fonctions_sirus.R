# ============================================================================
# Fonctions partagees entre evaluation_sirus.R et production_sirus.R.
#
# Toute correction du pipeline (troncature, construction de la table long)
# doit etre faite ICI UNIQUEMENT -- plusieurs scripts du projet ont
# longtemps coexiste avec des versions differentes du fix de troncature
# (cf. resultats_sirus_final.qmd section 1 et 6), source de confusion sur
# les resultats. Centraliser evite que ca se reproduise.
# ============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

SENTINELLES_CLASSIFIEUR <- c("AUCUNE_SUGGESTION", "NON_CODABLE")

# Troncature niveau 4 (5 chiffres significatifs) AVEC cascade de retrait des
# "0" terminaux. Validee : aucune position de chiffre atteinte par cette
# troncature n'a jamais "0" en concurrence avec un chiffre non nul dans les
# donnees observees (cf. resultats_sirus_final.qmd section 1) -- la cascade
# ne peut donc pas fusionner deux vraies sous-categories distinctes.
tronquer_niveau_un <- function(code, niveau = 4) {
  if (is.na(code)) return(NA_character_)
  s <- as.character(code)
  if (s %in% SENTINELLES_CLASSIFIEUR) return(s)
  n_chiffres_cible <- niveau + 1
  chiffres <- 0L
  out <- character(0)
  for (ch in strsplit(s, "")[[1]]) {
    if (grepl("[0-9]", ch)) {
      if (chiffres == n_chiffres_cible) break
      chiffres <- chiffres + 1L
      out <- c(out, ch)
    } else if (chiffres < n_chiffres_cible) {
      out <- c(out, ch)
    }
  }
  resultat <- paste(out, collapse = "")
  while (grepl("\\.0$", resultat)) resultat <- sub("\\.0$", "", resultat)
  resultat
}
tronquer_niveau <- function(codes, niveau = 4) {
  vapply(codes, tronquer_niveau_un, character(1), niveau = niveau, USE.NAMES = FALSE)
}

# Construit la table "long" (une ligne par candidat propose par au moins un
# des 4 classifieurs de base) a partir d'un run brut.
#
# - Mode EVALUATION : si `df` contient une colonne `vrai_n4` (vrai code deja
#   tronque -- utiliser `code_lvl4` si present dans le run, sinon
#   `tronquer_niveau(df$code, niveau)`), `correcte` est calcule normalement.
# - Mode PRODUCTION : si `df` n'a pas de colonne `vrai_n4`, `correcte` vaut
#   NA partout -- le vrai code est par definition inconnu pour de nouvelles
#   lignes. Ne jamais essayer de le "deviner" a partir d'une colonne `code`
#   presente par ailleurs dans le run : construire_table_long() l'ignore
#   volontairement si `vrai_n4` n'a pas ete fourni explicitement.
construire_table_long <- function(df, niveau = 4) {
  a_verite <- "vrai_n4" %in% names(df)
  if (!a_verite) df$vrai_n4 <- NA_character_

  df <- df %>%
    mutate(
      lcs_n4    = tronquer_niveau(lcs_code, niveau),
      rag_n4    = tronquer_niveau(rag_code, niveau),
      ragann_n4 = tronquer_niveau(ragann_code, niveau),
      ttc_n4    = tronquer_niveau(ttc_code_1, niveau),
    )

  candidats <- df %>%
    select(ligne, lcs_n4, rag_n4, ragann_n4, ttc_n4, vrai_n4) %>%
    pivot_longer(
      cols = c(lcs_n4, rag_n4, ragann_n4, ttc_n4),
      names_to = "classifieur", values_to = "code_candidat",
    ) %>%
    filter(
      !is.na(code_candidat),
      # une sentinelle "AUCUNE_SUGGESTION"/"NON_CODABLE" veut dire que le
      # classifieur n'a PAS propose de code -- ce n'est pas un candidat
      # votable (correctif valide dans synthese_bis.qmd).
      !code_candidat %in% SENTINELLES_CLASSIFIEUR,
    ) %>%
    distinct(ligne, code_candidat, vrai_n4)

  candidats %>%
    left_join(
      df %>% select(
        ligne, lcs_n4, rag_n4, ragann_n4, ttc_n4,
        lcs_distance, rag_confidence, ragann_confidence, ttc_conf_1,
      ),
      by = "ligne",
    ) %>%
    mutate(
      vote_lcs    = as.integer(!is.na(lcs_n4)    & lcs_n4    == code_candidat),
      vote_rag    = as.integer(!is.na(rag_n4)    & rag_n4    == code_candidat),
      vote_ragann = as.integer(!is.na(ragann_n4) & ragann_n4 == code_candidat),
      vote_ttc    = as.integer(!is.na(ttc_n4)    & ttc_n4    == code_candidat),
      nb_votants  = vote_lcs + vote_rag + vote_ragann + vote_ttc,
      correcte    = if (a_verite) as.integer(!is.na(vrai_n4) & code_candidat == vrai_n4) else NA_integer_,
      code_candidat_n1 = tronquer_niveau(code_candidat, niveau = 1),
      # confiances/distance : sentinelle hors plage reelle quand le
      # classifieur n'a pas vote pour CE candidat precis -- verifie sans
      # risque de collision (rag/ragann/ttc confidence in [0,1], lcs_distance
      # in [0, ~0.8] sur les donnees observees ; cf. discussion du projet).
      conf_rag    = if_else(vote_rag == 1L,    coalesce(rag_confidence, -1),    -1),
      conf_ragann = if_else(vote_ragann == 1L, coalesce(ragann_confidence, -1), -1),
      conf_ttc    = if_else(vote_ttc == 1L,    coalesce(ttc_conf_1, -1),        -1),
      dist_lcs    = if_else(vote_lcs == 1L,    coalesce(lcs_distance, 1.5),     1.5),
    ) %>%
    select(
      ligne, code_candidat, code_candidat_n1,
      vote_lcs, vote_rag, vote_ragann, vote_ttc, nb_votants,
      conf_rag, conf_ragann, conf_ttc, dist_lcs, correcte,
    )
}

# Features utilisees par le modele SIRUS. `budget`/`shop_type_name`/`source`
# testes et ecartes (aucun gain mesurable, ni sur l'ancien run ni sur x98xl
# -- cf. resultats_sirus_final.qmd section 3) : volontairement absents.
FEATURES_SIRUS <- c(
  "vote_lcs", "vote_rag", "vote_ragann", "vote_ttc", "nb_votants",
  "conf_rag", "conf_ragann", "conf_ttc", "dist_lcs", "code_candidat_n1"
)

# Produits presents dans `df` (identifiant `ligne`) mais absents de
# `table_long` -- aucun candidat propose par aucun classifieur (ou tous en
# sentinelle). Ne peuvent structurellement pas etre codes automatiquement :
# a traiter explicitement (cf. production_sirus.R), sinon ils disparaissent
# silencieusement des resultats plutot que d'etre signales.
produits_sans_candidat <- function(df, table_long) {
  setdiff(unique(df$ligne), unique(table_long$ligne))
}
