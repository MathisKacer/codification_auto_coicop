# %%
import sys
sys.path.append("/home/onyxia/work/codification_auto_coicop")  # racine du projet
from data.load_data import charger_base
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# %%
df = charger_base()

# %%
print("Dimensions :", df.shape)
print("\nColonnes :", df.columns.tolist())
print("\nTypes :\n", df.dtypes)
print("\nValeurs manquantes par colonne :\n", df.isna().sum())
print("\nDistribution de la categorie vraie :\n", df["code"].value_counts())
print("\nApercu :\n", df.head())

# %%
# enlève les lignes dont il manque la coicop ou la réponse du llm (11 lignes)
df_valide = df.dropna(subset=["code", "llm_code"]).copy()

print(f"Lignes valides : {df_valide.shape[0]} / {df.shape[0]} ({df.shape[0] - df_valide.shape[0]} lignes exclues)")

accuracy_llm = accuracy_score(df_valide["code"], df_valide["llm_code"])
print(f"Accuracy actuelle du LLM-as-judge : {accuracy_llm:.3f}")  #accuracy de 0.725

# %%
# preparation features

colonnes_categorielles = [
    "lcs_code", "rag_code", "ragann_code", "ttc_code_1", "ttc_code_2", "ttc_code_3",
    "shop_type_name", "codable"
]
colonnes_numeriques = [
    "lcs_distance", "rag_confidence", "ragann_confidence", "ttc_conf_1", "ttc_conf_2", "ttc_conf_3",
    "budget"
]

#Recodage des NA

df_valide["lcs_manquant"] = df_valide["lcs_code"].isna().astype(int)
df_valide["lcs_code"] = df_valide["lcs_code"].fillna("AUCUNE_SUGGESTION")

df_valide["lcs_distance"] = df_valide["lcs_distance"].fillna(df_valide["lcs_distance"].max() * 1.5)

df_valide["rag_code"] = df_valide["rag_code"].fillna("NON_CODABLE")

df_valide["ragann_manquant"] = df_valide["ragann_code"].isna().astype(int)
df_valide["ragann_code"] = df_valide["ragann_code"].fillna("NON_CODABLE")

df_valide["shop_type_name"] = df_valide["shop_type_name"].fillna("INCONNU")

df_valide["budget"] = df_valide["budget"].fillna(df_valide["budget"].median())


X = df_valide[colonnes_categorielles + colonnes_numeriques]
y = df_valide["code"]

print(X.shape, y.shape)
print(X.isna().sum())

# %%
