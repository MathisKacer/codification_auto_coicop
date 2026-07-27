system("mkdir -p ~/.R")
system("echo 'CXX11STD = -std=gnu++11' >> ~/.R/Makevars")
system("echo 'CXXFLAGS = -std=gnu++11' >> ~/.R/Makevars")

install.packages(c("ROCR", "glmnet", "RcppEigen"))

install.packages(
  "https://cran.r-project.org/src/contrib/Archive/sirus/sirus_0.3.3.tar.gz",
  repos = NULL, type = "source"
)

library(arrow)
library(dplyr)
library(tidyr)
library(sirus)

niveau <- 4  # convention du projet : niveau N = N+1 chiffres significatifs

bucket_s3  <- "projet-budget-famille"
objet_s3   <- "data/workflow_runs/2026-06-29/codif-vvkv9/decide-coicop/predictions.parquet"
endpoint_s3 <- "minio.lab.sspcloud.fr"

df <- aws.s3::s3read_using(
  FUN = arrow::read_parquet,
  object = objet_s3,
  bucket = bucket_s3,
  opts = list(region = "", base_url = endpoint_s3),
)

# %% Verification des donnees ----

cols_classifieurs <- c("lcs_code", "rag_code", "ragann_code", "ttc_code_1")
stopifnot(
  "colonnes manquantes dans df" = all(c(cols_classifieurs, "code") %in% names(df))
)

if (!"ligne" %in% names(df)) {
  df$ligne <- seq_len(nrow(df)) - 1L  # 0-indexe, aligne sur la convention Python (df.index)
}

tronquer_niveau_un <- function(code, niveau = 4) {
  if (is.na(code)) return(NA_character_)
  s <- as.character(code)
  if (s %in% c("AUCUNE_SUGGESTION", "NON_CODABLE")) return(s)
  
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
  paste(out, collapse = "")
}

tronquer_niveau <- function(codes, niveau = 4) {
  vapply(codes, tronquer_niveau_un, character(1), niveau = niveau, USE.NAMES = FALSE)
}

# %% Construction du format long ----

df <- df %>%
  mutate(
    lcs_n4    = tronquer_niveau(lcs_code, niveau),
    rag_n4    = tronquer_niveau(rag_code, niveau),
    ragann_n4 = tronquer_niveau(ragann_code, niveau),
    ttc_n4    = tronquer_niveau(ttc_code_1, niveau),
    ttc2_n4   = tronquer_niveau(ttc_code_2, niveau),  # rang 2 de TTC
    ttc3_n4   = tronquer_niveau(ttc_code_3, niveau),  # rang 3 de TTC
    vrai_n4   = tronquer_niveau(code, niveau),
  )


candidats <- df %>%
  select(ligne, lcs_n4, rag_n4, ragann_n4, ttc_n4, vrai_n4) %>%
  pivot_longer(
    cols = c(lcs_n4, rag_n4, ragann_n4, ttc_n4),
    names_to = "classifieur", values_to = "code_candidat",
  ) %>%
  filter(!is.na(code_candidat)) %>%
  distinct(ligne, code_candidat, vrai_n4)

table_long <- candidats %>%
  left_join(
    df %>% select(
      ligne, lcs_n4, rag_n4, ragann_n4, ttc_n4, ttc2_n4, ttc3_n4,
      lcs_distance, rag_confidence, ragann_confidence,
      ttc_conf_1, ttc_conf_2, ttc_conf_3,
    ),
    by = "ligne",
  ) %>%
  mutate(
    vote_lcs    = as.integer(!is.na(lcs_n4)    & lcs_n4    == code_candidat),
    vote_rag    = as.integer(!is.na(rag_n4)    & rag_n4    == code_candidat),
    vote_ragann = as.integer(!is.na(ragann_n4) & ragann_n4 == code_candidat),
    vote_ttc    = as.integer(!is.na(ttc_n4)    & ttc_n4    == code_candidat),
    nb_votants  = vote_lcs + vote_rag + vote_ragann + vote_ttc,
    correcte    = as.integer(!is.na(vrai_n4) & code_candidat == vrai_n4),
    # Feature "simple" demandee, derivee du candidat (pas le code lui-meme) :
    code_candidat_n1 = tronquer_niveau(code_candidat, niveau = 1),
    conf_rag    = if_else(vote_rag == 1L,    coalesce(rag_confidence, -1),    -1),
    conf_ragann = if_else(vote_ragann == 1L, coalesce(ragann_confidence, -1), -1),
    dist_lcs    = if_else(vote_lcs == 1L,    coalesce(lcs_distance, 1.5),     1.5),
    ttc_rang = case_when(
      code_candidat == ttc_n4  ~ 1L,
      code_candidat == ttc2_n4 ~ 2L,
      code_candidat == ttc3_n4 ~ 3L,
      TRUE                     ~ 4L,
    ),
    ttc_conf_au_rang = case_when(
      code_candidat == ttc_n4  ~ coalesce(ttc_conf_1, -1),
      code_candidat == ttc2_n4 ~ coalesce(ttc_conf_2, -1),
      code_candidat == ttc3_n4 ~ coalesce(ttc_conf_3, -1),
      TRUE                     ~ -1,
    ),
  ) %>%
  select(
    ligne,
    code_candidat,      # reference/tracabilite uniquement -- NE PAS utiliser comme feature
    code_candidat_n1, vote_lcs, vote_rag, vote_ragann, vote_ttc, nb_votants,
    conf_rag, conf_ragann, dist_lcs,
    ttc_rang, ttc_conf_au_rang, correcte,
  )

cat(sprintf(
  "%d produits -> %d lignes candidates (%.2f lignes/produit en moyenne)\n",
  n_distinct(table_long$ligne), nrow(table_long),
  nrow(table_long) / n_distinct(table_long$ligne)
))

set.seed(42)  # cf. random_state=42 cote Python, pour rester coherent
lignes_uniques <- unique(table_long$ligne)
lignes_train <- sample(lignes_uniques, size = round(0.8 * length(lignes_uniques)))

table_long <- table_long %>% mutate(code_candidat_n1 = as.factor(code_candidat_n1))

features <- c(
  "vote_lcs", "vote_rag", "vote_ragann", "vote_ttc", "nb_votants",
  "conf_rag", "conf_ragann", "dist_lcs",
  "ttc_rang", "ttc_conf_au_rang", "code_candidat_n1"
)

train <- table_long %>% filter(ligne %in% lignes_train)
test  <- table_long %>% filter(!ligne %in% lignes_train)

modele <- sirus.fit(
  data = as.data.frame(train[, features]), y = train$correcte, type = "classif", num.rule = 10
)
sirus.print(modele)

# %% Evaluation candidat-level ----
# "Ce code candidat est-il le bon ?" (une ligne = un candidat). Non comparable
# directement a la RF Python, qui raisonne au niveau du produit (cf. ci-dessous).
test$proba <- sirus.predict(modele, as.data.frame(test[, features]))
acc_candidat <- mean(as.integer(test$proba > 0.5) == test$correcte)
cat(sprintf("Accuracy candidat-level (test) : %.3f\n", acc_candidat))

choix_par_produit <- test %>%
  group_by(ligne) %>%
  slice_max(proba, n = 1, with_ties = FALSE) %>%
  ungroup()

acc_produit <- mean(choix_par_produit$correcte)
cat(sprintf(
  "Accuracy produit-level (test) : %.3f  (%d produits)\n",
  acc_produit, nrow(choix_par_produit)
))

# Borne haute atteignable : part des produits du test dont le vrai code figure
# parmi les candidats (si absent, aucun choix ne peut etre correct).
borne_haute <- test %>% group_by(ligne) %>% summarise(possible = max(correcte)) %>%
  pull(possible) %>% mean()
cat(sprintf(
  "Borne haute (vrai code present parmi les candidats) : %.3f\n", borne_haute
))

# %% Test multi-graines : le gain du top-3 est-il reel ou dans le bruit ? ----
# Les comparaisons ci-dessus reposent sur une seule graine. Or la selection de
# regles de SIRUS (foret aleatoire interne) ET le split train/test introduisent
# de la variance -- du meme ordre (~2 pts) que les ecarts qu'on a compares. On
# relance donc chaque config sur plusieurs graines et on compare les
# DISTRIBUTIONS de produit-level, pas un point unique.
#
# num.trees fixe (5000) pour rendre les runs comparables et tenir le temps de
# calcul (sinon SIRUS augmente les arbres jusqu'a stabilisation, cout variable).

evaluer_config <- function(feats, graines = 1:8, n_arbres = 5000) {
  bind_rows(lapply(graines, function(g) {
    set.seed(g)
    lignes_tr <- sample(lignes_uniques, size = round(0.8 * length(lignes_uniques)))
    tr <- table_long %>% filter(ligne %in% lignes_tr)
    te <- table_long %>% filter(!ligne %in% lignes_tr)
    m  <- sirus.fit(
      as.data.frame(tr[, feats]), tr$correcte,
      type = "classif", num.trees = n_arbres, seed = g, verbose = FALSE,
    )
    te$proba <- sirus.predict(m, as.data.frame(te[, feats]))
    choix <- te %>% group_by(ligne) %>%
      slice_max(proba, n = 1, with_ties = FALSE) %>% ungroup()
    data.frame(
      graine   = g,
      acc_cand = mean(as.integer(te$proba > 0.5) == te$correcte),
      acc_prod = mean(choix$correcte)
    )
  }))
}

# Deux configs, toutes deux sous-ensembles des colonnes deja dans table_long :
feats_sans_top3 <- setdiff(features, c("ttc_rang", "ttc_conf_au_rang"))
feats_avec_top3 <- features  # config courante

recap_graines <- bind_rows(
  data.frame(config = "sans top-3", evaluer_config(feats_sans_top3)),
  data.frame(config = "avec top-3", evaluer_config(feats_avec_top3)),
) %>%
  group_by(config) %>%
  summarise(
    prod_moy = mean(acc_prod), prod_sd = sd(acc_prod),
    prod_min = min(acc_prod), prod_max = max(acc_prod),
    cand_moy = mean(acc_cand), cand_sd = sd(acc_cand),
    .groups = "drop",
  )
print(recap_graines)