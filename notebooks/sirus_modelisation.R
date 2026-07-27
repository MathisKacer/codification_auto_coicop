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
endpoint_s3 <- "minio.lab.sspcloud.fr"

objet_s3         <- "data/workflow_runs/2026-06-29/codif-vvkv9/decide-coicop/predictions.parquet"
objet_s3_nouveau <- "data/workflow_runs/2026-07-23/codif-8m8jn/decide-coicop/predictions.parquet"

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

construire_table_long <- function(df, niveau = 4) {
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

  candidats %>%
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
}

table_long <- construire_table_long(df, niveau)

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

# %% Table produit-level : la confiance de SIRUS est-elle calibree ? ----
# Objectif : savoir si on peut faire confiance au choix de SIRUS sans passer
# par le LLM -- meme logique que le RF Python sur la baseline, mais ici on
# teste d'abord un simple seuil sur la confiance que SIRUS donne DEJA a son
# propre choix (pas besoin d'un modele supplementaire si c'est deja calibre).
#
# On redescend de table_long (un candidat par ligne) au niveau produit :
# proba du candidat retenu, proba du 2e (marge de victoire), nombre de
# candidats en lice, et si le choix retenu est correct.

table_produit <- test %>%
  group_by(ligne) %>%
  arrange(desc(proba), .by_group = TRUE) %>%
  summarise(
    n_candidats = n(),
    proba_gagnant = first(proba),
    proba_second = if (n() >= 2) nth(proba, 2) else NA_real_,
    sirus_correct = first(correcte),
    .groups = "drop",
  ) %>%
  mutate(
    # sans 2e candidat, pas de "marge" a proprement parler : on retient la
    # proba du gagnant elle-meme (rien pour la contester localement).
    ecart_proba = if_else(is.na(proba_second), proba_gagnant, proba_gagnant - proba_second),
  )

# Calibration de la proba du candidat retenu (meme methode que
# calibration-confiance.qmd cote site, appliquee ici a la sortie de SIRUS).
bornes_proba <- c(0, 0.2, 0.4, 0.6, 0.8, 1.0)
table_produit$tranche_proba_gagnant <- cut(table_produit$proba_gagnant, breaks = bornes_proba, include.lowest = TRUE)

recap_calibration_proba <- table_produit %>%
  group_by(tranche_proba_gagnant) %>%
  summarise(n = n(), pct_correct = mean(sirus_correct), .groups = "drop")
cat("Calibration de la proba du candidat retenu :\n")
print(recap_calibration_proba)

# Calibration de la marge (ecart entre le 1er et le 2e candidat) : parfois
# plus discriminant qu'une proba brute (un gagnant a 0.6 contre un 2e a 0.58
# est bien plus incertain qu'un gagnant a 0.6 contre un 2e a 0.1).
bornes_ecart <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)
table_produit$tranche_ecart <- cut(table_produit$ecart_proba, breaks = bornes_ecart, include.lowest = TRUE)

recap_calibration_ecart <- table_produit %>%
  group_by(tranche_ecart) %>%
  summarise(n = n(), pct_correct = mean(sirus_correct), .groups = "drop")
cat("\nCalibration de la marge (proba gagnant - proba 2e) :\n")
print(recap_calibration_ecart)

# Repere : accuracy selon le nombre de candidats en lice (avec un seul
# candidat, SIRUS ne "choisit" rien -- la aussi bon ou mauvais que le
# candidat lui-meme).
recap_par_n_candidats <- table_produit %>%
  group_by(n_candidats) %>%
  summarise(n = n(), pct_correct = mean(sirus_correct), proba_gagnant_moy = mean(proba_gagnant), .groups = "drop")
cat("\nAccuracy selon le nombre de candidats en lice :\n")
print(recap_par_n_candidats)

# %% Test multi-graines : le gain du top-3 est-il reel ou dans le bruit ? ----
# Les comparaisons ci-dessus reposent sur une seule graine. Or la selection de
# regles de SIRUS (foret aleatoire interne) ET le split train/test introduisent
# de la variance -- du meme ordre (~2 pts) que les ecarts qu'on a compares. On
# relance donc chaque config sur plusieurs graines et on compare les
# DISTRIBUTIONS de produit-level, pas un point unique.
#
# num.trees n'est PAS fixe ici : force a une valeur commune (5000, puis
# 10000 testes), la foret interne de SIRUS n'atteignait pas forcement sa
# stabilisation propre a chaque graine. On laisse plutot SIRUS s'auto-
# stabiliser (comme le modele principal plus haut), et on fixe a la place
# num.rule = 5 : c'est ce nombre de regles qui rend les configs comparables
# entre elles, pas le nombre d'arbres internes.

evaluer_config <- function(feats, graines = 1:8, regles = 5) {
  bind_rows(lapply(graines, function(g) {
    set.seed(g)
    lignes_tr <- sample(lignes_uniques, size = round(0.8 * length(lignes_uniques)))
    tr <- table_long %>% filter(ligne %in% lignes_tr)
    te <- table_long %>% filter(!ligne %in% lignes_tr)
    m  <- sirus.fit(
      as.data.frame(tr[, feats]), tr$correcte,
      type = "classif", num.rule = regles, seed = g, verbose = FALSE,
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

# %% Generalisation au nouveau run ----
# Meme logique que notebooks/generalisation_nouveau_run.py cote Python :
# 1) le modele entraine sur l'ancien run (deja fit plus haut) est evalue sur
#    le nouveau run integral (jamais vu a l'entrainement) ;
# 2) en reference, un modele reentraine directement sur le nouveau run
#    (meme split 80/20, seed 42) donne l'accuracy "in-distribution".

df_nouveau <- aws.s3::s3read_using(
  FUN = arrow::read_parquet,
  object = objet_s3_nouveau,
  bucket = bucket_s3,
  opts = list(region = "", base_url = endpoint_s3),
)
if (!"ligne" %in% names(df_nouveau)) {
  df_nouveau$ligne <- seq_len(nrow(df_nouveau)) - 1L
}

table_long_nouveau <- construire_table_long(df_nouveau, niveau) %>%
  mutate(code_candidat_n1 = factor(code_candidat_n1, levels = levels(table_long$code_candidat_n1)))

# 1) Modele entraine sur l'ancien run (`modele`, fit plus haut), teste sur le nouveau run integral
table_long_nouveau$proba <- sirus.predict(modele, as.data.frame(table_long_nouveau[, features]))

acc_candidat_ancien_sur_nouveau <- mean(as.integer(table_long_nouveau$proba > 0.5) == table_long_nouveau$correcte)

choix_par_produit_nouveau <- table_long_nouveau %>%
  group_by(ligne) %>%
  slice_max(proba, n = 1, with_ties = FALSE) %>%
  ungroup()
acc_produit_ancien_sur_nouveau <- mean(choix_par_produit_nouveau$correcte)

borne_haute_nouveau <- table_long_nouveau %>%
  group_by(ligne) %>%
  summarise(possible = max(correcte)) %>%
  pull(possible) %>%
  mean()

# 2) Reference in-distribution : reentraine et evalue directement sur le nouveau run
set.seed(42)
lignes_uniques_nouveau <- unique(table_long_nouveau$ligne)
lignes_train_nouveau <- sample(lignes_uniques_nouveau, size = round(0.8 * length(lignes_uniques_nouveau)))

train_nouveau <- table_long_nouveau %>% filter(ligne %in% lignes_train_nouveau)
test_nouveau  <- table_long_nouveau %>% filter(!ligne %in% lignes_train_nouveau)

modele_nouveau <- sirus.fit(
  data = as.data.frame(train_nouveau[, features]), y = train_nouveau$correcte, type = "classif", num.rule = 10
)
test_nouveau$proba <- sirus.predict(modele_nouveau, as.data.frame(test_nouveau[, features]))

acc_candidat_nouveau_reentraine <- mean(as.integer(test_nouveau$proba > 0.5) == test_nouveau$correcte)

choix_par_produit_nouveau_reentraine <- test_nouveau %>%
  group_by(ligne) %>%
  slice_max(proba, n = 1, with_ties = FALSE) %>%
  ungroup()
acc_produit_nouveau_reentraine <- mean(choix_par_produit_nouveau_reentraine$correcte)

recap_generalisation <- data.frame(
  modele = c(
    "Entraine ancien, teste ancien (test 20%)",
    "Entraine ancien, teste nouveau (integral)",
    "Entraine nouveau, teste nouveau (test 20%)"
  ),
  acc_candidat = c(acc_candidat, acc_candidat_ancien_sur_nouveau, acc_candidat_nouveau_reentraine),
  acc_produit  = c(acc_produit,  acc_produit_ancien_sur_nouveau,  acc_produit_nouveau_reentraine)
)
print(recap_generalisation)
cat(sprintf("Borne haute (nouveau run, produit-level) : %.3f\n", borne_haute_nouveau))

# %% Generalisation au nouveau run, multi-graines ----
# Le resultat ci-dessus est mono-graine (seed 42) : meme logique que le test
# multi-graines plus haut (variance ~2 pts liee au split train/test ET a la
# selection de regles de SIRUS). On repete ici la comparaison generalisation
# (modele ancien teste sur le nouveau run vs modele reentraine directement
# sur le nouveau run) sur plusieurs graines, pour verifier que l'ecart
# (quasi nul) constate en mono-graine n'est pas juste du bruit. Comme pour
# evaluer_config, on fixe num.rule (5) plutot que num.trees, pour laisser
# SIRUS s'auto-stabiliser a chaque graine.

evaluer_generalisation <- function(graines = 1:8, regles = 5) {
  bind_rows(lapply(graines, function(g) {
    set.seed(g)
    lignes_tr <- sample(lignes_uniques, size = round(0.8 * length(lignes_uniques)))
    tr <- table_long %>% filter(ligne %in% lignes_tr)
    m_ancien <- sirus.fit(
      as.data.frame(tr[, features]), tr$correcte,
      type = "classif", num.rule = regles, seed = g, verbose = FALSE,
    )

    tn <- table_long_nouveau
    tn$proba <- sirus.predict(m_ancien, as.data.frame(tn[, features]))
    choix_ancien_sur_nouveau <- tn %>% group_by(ligne) %>%
      slice_max(proba, n = 1, with_ties = FALSE) %>% ungroup()

    lignes_tr_n <- sample(lignes_uniques_nouveau, size = round(0.8 * length(lignes_uniques_nouveau)))
    tr_n <- table_long_nouveau %>% filter(ligne %in% lignes_tr_n)
    te_n <- table_long_nouveau %>% filter(!ligne %in% lignes_tr_n)
    m_nouveau <- sirus.fit(
      as.data.frame(tr_n[, features]), tr_n$correcte,
      type = "classif", num.rule = regles, seed = g, verbose = FALSE,
    )
    te_n$proba <- sirus.predict(m_nouveau, as.data.frame(te_n[, features]))
    choix_nouveau_reentraine <- te_n %>% group_by(ligne) %>%
      slice_max(proba, n = 1, with_ties = FALSE) %>% ungroup()

    data.frame(
      graine = g,
      acc_produit_ancien_sur_nouveau = mean(choix_ancien_sur_nouveau$correcte),
      acc_produit_nouveau_reentraine = mean(choix_nouveau_reentraine$correcte)
    )
  }))
}

recap_generalisation_multi <- evaluer_generalisation()
print(recap_generalisation_multi)

recap_generalisation_multi %>%
  summarise(
    moy_ancien_sur_nouveau = mean(acc_produit_ancien_sur_nouveau),
    sd_ancien_sur_nouveau = sd(acc_produit_ancien_sur_nouveau),
    moy_nouveau_reentraine = mean(acc_produit_nouveau_reentraine),
    sd_nouveau_reentraine = sd(acc_produit_nouveau_reentraine),
  ) %>%
  print()