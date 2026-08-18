#!/usr/bin/env Rscript
# ============================================================================
# Installe les packages R necessaires au pipeline SIRUS (fonctions_sirus.R,
# evaluation_sirus.R, production_sirus.R). A relancer a chaque nouvel
# environnement / service SSP Cloud -- ces packages ne sont pas installes par
# defaut et ne persistent generalement pas entre deux relances du service.
#
# Usage : Rscript install_dependencies.R   (depuis le dossier stage/)
# ============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

# randomForest et stringr ne sont utilises que par resultats_sirus_final.qmd
# (pas par evaluation_sirus.R / production_sirus.R) -- necessaires seulement
# si vous voulez re-rendre ce document.
deps_cran <- c("arrow", "dplyr", "tidyr", "aws.s3", "ROCR", "glmnet", "RcppEigen", "randomForest", "stringr")
manquants <- setdiff(deps_cran, rownames(installed.packages()))
if (length(manquants) > 0) {
  cat("Installation des dependances CRAN manquantes :", paste(manquants, collapse = ", "), "\n")
  install.packages(manquants)
} else {
  cat("Dependances CRAN deja presentes.\n")
}

# --- Package sirus -----------------------------------------------------------
# IMPORTANT : installer sirus_0.3.3_patched.tar.gz, PAS sirus_0.3.3.tar.gz.
#
# Le tarball original (sirus_0.3.3.tar.gz, conserve tel quel a titre de
# reference/provenance) ne compile plus avec les toolchains R/gcc recents :
#   1. Le package definit son propre polyfill de `make_unique` (pour
#      pre-C++14). R ignore desormais la directive `CXX_STD = CXX11` de son
#      Makevars et compile avec un standard C++ recent (ex. gnu++20) ou
#      std::make_unique existe deja -- l'appel devient ambigu.
#   2. Rinternals.h definit des macros brutes (`error`, `length`, ...) qui
#      polluent des headers standard C++ inclus plus tard dans la meme unite
#      de compilation (ex. <bits/locale_conv.h>), cassant la compilation.
#
# sirus_0.3.3_patched.tar.gz corrige ces deux points (macro R_NO_REMAP +
# polyfill make_unique restreint a `#if __cplusplus < 201402L`) sans toucher
# au comportement du modele. Si ce tarball venait a manquer, regenerer le
# patch a partir des memes principes sur les fichiers src/utility.h et
# src/Makevars du package sirus.
if (!"sirus" %in% rownames(installed.packages())) {
  chemin_tarball <- "sirus_0.3.3_patched.tar.gz"
  if (!file.exists(chemin_tarball)) {
    stop(sprintf(
      "%s introuvable dans le dossier courant -- executer ce script depuis stage/.",
      chemin_tarball
    ))
  }
  cat("Installation de sirus (source patchee)...\n")
  install.packages(chemin_tarball, repos = NULL, type = "source")
} else {
  cat("sirus deja installe.\n")
}

stopifnot(
  "sirus ne s'est pas installe correctement" = "sirus" %in% rownames(installed.packages())
)
cat("\nToutes les dependances sont pretes.\n")
