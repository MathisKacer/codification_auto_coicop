import sys
sys.path.append("/home/onyxia/work/codification_auto_coicop")  # racine du projet
from data.load_data import charger_base
from sklearn.metrics import accuracy_score, classification_report

df = charger_base()

###

print("Dimensions :", df.shape)
print("\nColonnes :", df.columns.tolist())
print("\nTypes :\n", df.dtypes)
print("\nValeurs manquantes par colonne :\n", df.isna().sum())
print("\nDistribution de la categorie vraie :\n", df["code"].value_counts())
print("\nApercu :\n", df.head())

###

#enlève les lignes dont il manque la coicop ou la réponse du llm (11 lignes)
df_valide = df.dropna(subset=["code", "llm_code"])

print(f"Lignes valides : {df_valide.shape[0]} / {df.shape[0]} ({df.shape[0] - df_valide.shape[0]} lignes exclues)")

accuracy_llm = accuracy_score(df_valide["code"], df_valide["llm_code"])
print(f"Accuracy actuelle du LLM-as-judge : {accuracy_llm:.3f}")  #accuracy de 0.725

###

#preparation features 

colonnes_categorielles = [
    "lcs_code", "rag_code", "ragann_code", "ttc_code_1", "ttc_code_2", "ttc_code_3",
    "shop_type_name"
]
colonnes_numeriques = [
    "lcs_distance", "rag_confidence", "ragann_confidence", "ttc_conf_1", "ttc_conf_2", "ttc_conf_3",
    "budget"
]

X = df_valide[colonnes_categorielles + colonnes_numeriques]
y = df_valide["code"]

print(X.shape, y.shape)
print(X.isna().sum())