# streamlit_app/pages/03_Sensibilite.py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from shap_utils import (  # Importez les fonctions de votre nouveau fichier streamlit_app.
    calculate_shap_values, load_model_and_data_for_shap, plot_shap_dependence,
    plot_shap_force, plot_shap_summary_bar, plot_shap_summary_dot)

#print(type(loa"""d_model_and_data_for_shap))
#print(load_model_and_data_for_shap.shape if hasattr(load_model_and_data_for_shap, 'shape') else "Pas de shape")#
#print(load_model_and_data_for_shap.head() if isinstance(load_model_and_data_for_shap, pd.DataFrame) else load_model_and_data_for_shap[:2])



# Vérifie l'état de connexion avant d'afficher la page
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Veuillez vous connecter pour accéder à cette page.")
    st.stop()

st.set_page_config(layout="wide")
st.title("🔬 Analyse de Sensibilité & Importance des Caractéristiques")
st.markdown("Comprenez les facteurs les plus influents qui poussent les prédictions du modèle.")

# --- Chargement des données d'importance ---
@st.cache_resource
def load_importance_data():
    try:
        performance_data = joblib.load('model_performance_data.pkl')
        features_df = performance_data['features_importance_df']
        return features_df
    except FileNotFoundError:
        st.error("Les données d'importance des caractéristiques ('model_performance_data.pkl') n'ont pas été trouvées. Veuillez exécuter la phase 5 dans le notebook pour les générer.")
        st.stop()

features_df = load_importance_data()

st.subheader("Top 15 des Caractéristiques les Plus Importantes")
st.write("Ces caractéristiques ont eu le plus grand impact sur la capacité du modèle à prédire les sinistres.")

fig_feat_imp, ax_feat_imp = plt.subplots(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=features_df.head(15), palette='viridis', ax=ax_feat_imp)
ax_feat_imp.set_title('Top 15 des Caractéristiques les Plus Importantes pour le Modèle XGBoost')
ax_feat_imp.set_xlabel('Importance (Gain de Split)')
ax_feat_imp.set_ylabel('Caractéristique')
plt.tight_layout()
st.pyplot(fig_feat_imp)

st.markdown("---")
st.subheader("Interprétabilité Locale (SHAP Values - Pour une Amélioration Future)")



# Optionnel : Afficher le DataFrame des importances complètes
if st.checkbox("Afficher toutes les importances des caractéristiques"):
    st.dataframe(features_df)