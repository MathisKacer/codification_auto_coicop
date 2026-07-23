# Notebook de recherche : modele SIRUS (regles interpretables, package R
# uniquement) sur un format long -- une ligne par code candidat propose par au
# moins un des 4 classifieurs (lcs, rag, ragann, ttc), donc 1 a 4 lignes par
# produit selon leur accord. Premiere base volontairement minimale, avant
# feature engineering : pas de code candidat brut en feature (trop de
# modalites), seulement des derives simples (troncature niveau 1).
#
# NON TESTE dans l'environnement de redaction (R n'y est pas installe) : a
# valider dans un environnement R (ex. service RStudio SSPCloud). Deux points
# a verifier en priorite, signales plus bas :
#   - le chargement S3 (aws.s3::s3read_using) : parametres base_url/region
#     adaptes au pattern SSPCloud habituel pour MinIO, a confirmer
#   - le chemin d'installation de `sirus` (archive CRAN vs GitLab)

# %% Setup ----

# Packages requis :
#   install.packages("aws.s3")        # lecture depuis S3/MinIO
#   install.packages("nanoparquet")   # lecture du parquet telecharge
#   install.packages("dplyr"); install.packages("tidyr")  # manipulation des donnees
#   install.packages(
#     "https://cran.r-project.org/src/contrib/Archive/sirus/sirus_0.3.3.tar.gz",
#     repos = NULL, type = "source"
#   )
#   # ou, alternative :
#   # install.packages("remotes"); remotes::install_gitlab("drti/sirus")

library(dplyr)
library(tidyr)
library(sirus)

niveau <- 4  # convention du projet : niveau N = N+1 chiffres significatifs

# %% Chargement des donnees (S3/MinIO) ----

# Meme run que CHEMIN_S3_MODELISATION cote Python (data/load_data.py).
# Les identifiants (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN)
# sont normalement deja presents comme variables d'environnement ambiantes
# dans un service SSPCloud -- aws.s3 les lit automatiquement, rien a
# renseigner a la main.
bucket_s3  <- "projet-budget-famille"
objet_s3   <- "data/workflow_runs/2026-06-29/codif-vvkv9/decide-coicop/predictions.parquet"
endpoint_s3 <- "minio.lab.sspcloud.fr"

df <- aws.s3::s3read_using(
  FUN = nanoparquet::read_parquet,
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

# %% Port R de tronquer_niveau (cf. src/coicop.py:13-39) ----
# Traduction directe : meme convention niveau N = N+1 chiffres significatifs,
# meme gestion des NA et des sentinels "AUCUNE_SUGGESTION"/"NON_CODABLE"
# (preserves tels quels). Toute evolution de la version Python doit etre
# repercutee ici.
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

# Un candidat = un (ligne, code propose) distinct parmi les 4 classifieurs
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
    # Confiance de chaque classifieur POUR CE candidat : sa confiance propre s'il
    # a vote pour ce candidat, sinon une sentinelle hors [0,1] (meme convention
    # que le preprocessing Python : -1 pour les confidences ; lcs_distance est une
    # distance, plus petite = plus proche, sentinelle 1.5 = "tres loin"). SIRUS ne
    # gere pas les NA, d'ou le coalesce vers la sentinelle.
    # Alternative non retenue ici : attacher la confiance brute meme quand le
    # classifieur a vote pour un AUTRE code (evidence "contre" ce candidat).
    conf_rag    = if_else(vote_rag == 1L,    coalesce(rag_confidence, -1),    -1),
    conf_ragann = if_else(vote_ragann == 1L, coalesce(ragann_confidence, -1), -1),
    # Pas de conf_ttc ici : ttc_conf_au_rang (plus bas) la subsume -- identique au
    # rang 1 et couvre en plus les rangs 2/3. L'ancienne conf_ttc produisait des
    # regles doublons avec ttc_conf_au_rang (cf. regles 6 et 7 identiques).
    dist_lcs    = if_else(vote_lcs == 1L,    coalesce(lcs_distance, 1.5),     1.5),
    # Top-3 de TTC (option B) : rang auquel CE candidat apparait dans le classement
    # de TTC, et confiance TTC a ce rang. Enrichit le signal TTC au-dela du seul
    # rang 1 (vote_ttc / conf_ttc) : un candidat classe 2e/3e par TTC n'est plus
    # confondu avec un candidat que TTC n'a jamais mentionne. Le case_when teste les
    # rangs dans l'ordre -> si un code apparait a plusieurs rangs (apres troncature),
    # on garde le meilleur (rang le plus haut, donc la confiance la plus forte).
    # Encodage monotone "plus petit = meilleur" : rang 1 (meilleur) ... 3, puis 4 =
    # absent du top-3 (pire) ; confiance -1 = absent (meme convention que conf_ttc).
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
    # budget a ete teste (contexte produit) puis retire : absent de toutes les
    # regles selectionnees, et son ajout perturbait la selection de regles au point
    # de FAIRE BAISSER le produit-level (0.864 -> 0.843). Etant identique pour tous
    # les candidats d'un produit, il ne pouvait de toute facon aider le re-ranking
    # que via des interactions, que SIRUS n'a pas retenues.
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

# %% Ajustement SIRUS ----

# Split AU NIVEAU PRODUIT (pas au niveau candidat) : tous les candidats d'un
# meme produit vont ensemble dans train ou test. Indispensable pour
# l'agregation produit-level plus bas -- sinon un produit aurait certains de ses
# candidats en train et d'autres en test, et l'argmax par produit sur le test
# n'aurait plus de sens.
set.seed(42)  # cf. random_state=42 cote Python, pour rester coherent
lignes_uniques <- unique(table_long$ligne)
lignes_train <- sample(lignes_uniques, size = round(0.8 * length(lignes_uniques)))

table_long <- table_long %>% mutate(code_candidat_n1 = as.factor(code_candidat_n1))
# Facteur construit sur l'ensemble avant le split, pour eviter tout niveau
# inedit au moment du predict().

# On GARDE vote_ttc bien qu'il soit logiquement equivalent a (ttc_rang == 1) :
# le retirer a fait CHUTER le produit-level (0.864 -> 0.839). SIRUS exploite le
# binaire vote_ttc et l'ordinal ttc_rang de facon COMPLEMENTAIRE (encodages
# differents, splits differents) -- ce n'etait donc pas une vraie redondance.
# (A l'inverse, conf_ttc etait bien redondant, DOMINE par ttc_conf_au_rang qui
# porte les memes valeurs continues + les rangs 2/3 : le retirer avait aide.)
features <- c(
  "vote_lcs", "vote_rag", "vote_ragann", "vote_ttc", "nb_votants",
  "conf_rag", "conf_ragann", "dist_lcs",
  "ttc_rang", "ttc_conf_au_rang", "code_candidat_n1"
)

train <- table_long %>% filter(ligne %in% lignes_train)
test  <- table_long %>% filter(!ligne %in% lignes_train)

modele <- sirus.fit(
  data = as.data.frame(train[, features]), y = train$correcte, type = "classif",
)
sirus.print(modele)

# %% Evaluation candidat-level ----
# "Ce code candidat est-il le bon ?" (une ligne = un candidat). Non comparable
# directement a la RF Python, qui raisonne au niveau du produit (cf. ci-dessous).
test$proba <- sirus.predict(modele, as.data.frame(test[, features]))
acc_candidat <- mean(as.integer(test$proba > 0.5) == test$correcte)
cat(sprintf("Accuracy candidat-level (test) : %.3f\n", acc_candidat))

# %% Evaluation produit-level ----
# Pour chaque produit du test, on retient le candidat de plus forte proba
# predite (SIRUS sert alors de re-ranker des codes candidats), puis on regarde
# si ce candidat retenu est le bon. Comparable a l'accuracy produit-level de la
# RF Python (page modelisation).
#
# Note : sur ce run, AUCUN produit du test n'a d'ex aequo sur la proba max
# (verifie : 0 / 1165). Le with_ties = FALSE ne tranche donc en pratique jamais
# rien ; inutile de chercher un departage plus fin.
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

# Lecture : si (prod_moy avec top-3) - (prod_moy sans top-3) est nettement plus
# grand que les prod_sd, le gain du top-3 est reel ; s'il est du meme ordre que
# l'ecart-type, c'etait (au moins en partie) du bruit de run.
