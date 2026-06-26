"""
Evaluation du meta-modele arbre (formulation B) :
- accuracy sur la prediction du gagnant (5-6 classes)
- accuracy finale au niveau hierarchique 4 (apres reconversion en code)
- comparaison avec le LLM-as-judge sur le meme jeu de test
"""
from sklearn.metrics import accuracy_score, classification_report

from src.correspondance_hierarchique import a_raison_jusqu_a_niveau
from src.cible_gagnant import gagnant_vers_code


def evaluer_prediction_gagnant(pipeline, X_test, y_test):
    """
    Evalue la qualite de la prediction du 'gagnant' (avant reconversion en code).
    Affiche l'accuracy globale et le rapport detaille par classe.
    """
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy (prediction du gagnant) : {accuracy:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return y_pred


def evaluer_code_final(y_pred_gagnant, X_test, df_valide, niveau_max: int = 4):
    """
    Reconvertit la prediction du gagnant en code, puis evalue la correspondance
    avec la verite terrain au niveau hierarchique cible.
    Compare aussi avec l'accuracy du LLM-as-judge sur exactement le meme jeu de test.

    Parameters
    ----------
    y_pred_gagnant : predictions du pipeline sur X_test (sortie de predict)
    X_test : DataFrame de test (utilise uniquement pour l'index)
    df_valide : DataFrame complet (pour acceder a "code", "llm_code", et aux codes des classifieurs)
    niveau_max : niveau hierarchique cible pour l'evaluation

    Returns
    -------
    (accuracy_arbre, accuracy_llm)
    """
    df_test = df_valide.loc[X_test.index]

    # Reconstruire les codes predits a partir du gagnant
    y_pred_code = [
        gagnant_vers_code(g, row)
        for g, (_, row) in zip(y_pred_gagnant, df_test.iterrows())
    ]

    y_vrai = df_test["code"].values

    # Accuracy de l'arbre au niveau cible
    correct_arbre = [
        a_raison_jusqu_a_niveau(pred, vrai, niveau_max)
        for pred, vrai in zip(y_pred_code, y_vrai)
    ]
    accuracy_arbre = sum(correct_arbre) / len(correct_arbre)

    # Accuracy du LLM sur le meme jeu de test, meme niveau
    y_llm = df_test["llm_code"].values
    correct_llm = [
        a_raison_jusqu_a_niveau(pred, vrai, niveau_max)
        for pred, vrai in zip(y_llm, y_vrai)
    ]
    accuracy_llm = sum(correct_llm) / len(correct_llm)

    print(f"Accuracy arbre (niveau {niveau_max}) : {accuracy_arbre:.3f}")
    print(f"Accuracy LLM   (niveau {niveau_max}) : {accuracy_llm:.3f}")
    return accuracy_arbre, accuracy_llm
