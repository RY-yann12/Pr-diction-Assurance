# shap_utils.py
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st  # Utilisé ici juste pour les décorateurs de cache


@st.cache_resource
def load_model_and_data_for_shap(model_path='best_xgb_model.pkl', data_path='X_y_test.pkl'):
# xgboost_model.pkl et X_test_processed.pkl



    """
    Charge le modèle entraîné et les données prétraitées nécessaires pour SHAP.
    """

"""
    try:
        model = joblib.load(model_path)
        X_data_for_shap = joblib.load(data_path)
        if not isinstance(X_data_for_shap, pd.DataFrame):
            st.warning("Les données chargées pour SHAP ne sont pas un DataFrame Pandas. Conversion tentée.")
            X_data_for_shap = pd.DataFrame(X_data_for_shap) # Assurez-vous que c'est un DataFrame

        return model, X_data_for_shap
    except FileNotFoundError as e:
        st.error(f"Erreur de chargement pour SHAP : {e}. Assurez-vous que '{model_path}' et '{data_path}' existent et sont correctement nommés.")
        return None, None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du chargement des ressources SHAP : {e}")
        return None, None
"""   

@st.cache_resource
def load_model_and_data_for_shap(model_path='best_xgb_model.pkl', data_path='X_y_test.pkl'):
    """
    Charge le modèle entraîné et les données prétraitées nécessaires pour SHAP.
    """
    try:
        model = joblib.load(model_path)
        data = joblib.load(data_path)

        # Ici, pas besoin d'extraire quoi que ce soit, c'est déjà un DataFrame
        X_data_for_shap = data

        # Si ce n’est pas un DataFrame, on tente une conversion
        if not isinstance(X_data_for_shap, pd.DataFrame):
            st.warning("Les données chargées ne sont pas un DataFrame. Tentative de conversion.")
            X_data_for_shap = pd.DataFrame(X_data_for_shap)

        # Vérifie qu'il n'y a pas de colonnes contenant des listes ou tableaux
        for col in X_data_for_shap.columns:
            if X_data_for_shap[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                st.error(f"⚠️ La colonne '{col}' contient des séquences. SHAP nécessite des scalaires.")
                return None, None

        return model, X_data_for_shap

    except FileNotFoundError as e:
        st.error(f"Erreur de chargement : {e}")
        return None, None
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
        return None, None


























@st.cache_data
def calculate_shap_values(model, data):
    """
    #Calcule les SHAP values pour un modèle et un jeu de données donnés.
    """
    try:
        explainer = shap.TreeExplainer(model)
        # Pour les modèles XGBoost de classification binaire, shap_values retourne souvent
        # un tableau de deux arrays (un pour chaque classe). Nous prenons celui de la classe positive.
        shap_values = explainer.shap_values(data)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1] # Prend les SHAP values pour la classe positive
        return explainer, shap_values
    except Exception as e:
        st.error(f"Erreur lors du calcul des SHAP values : {e}")
        return None, None

#----------------------------------------------------



#----------------------------------------------------



def plot_shap_force(explainer, shap_values_client, features_client):
    """
    Crée un SHAP force plot pour une observation client.
    Retourne une figure matplotlib.
    """
    fig = plt.figure(figsize=(10, 6))
    shap.force_plot(explainer.expected_value, shap_values_client, features_client, show=False, matplotlib=True)
    plt.tight_layout()
    return fig

def plot_shap_summary_bar(shap_values, features):
    """
    Crée un SHAP summary plot (barre) pour l'importance globale.
    Retourne une figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    plt.tight_layout()
    return fig

def plot_shap_summary_dot(shap_values, features):
    """
    Crée un SHAP summary plot (points) pour l'importance globale et la direction.
    Retourne une figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, features, show=False)
    plt.tight_layout()
    return fig

def plot_shap_dependence(feature_name, shap_values, features, interaction_feature=None):
    """
    Crée un SHAP dependence plot pour une caractéristique donnée.
    Retourne une figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(feature_name, shap_values, features, interaction_index=interaction_feature, show=False)
    plt.tight_layout()
    return fig

