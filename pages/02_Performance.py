# streamlit_app/pages/02_Performance.py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (auc, classification_report,
                             precision_recall_curve, roc_auc_score, roc_curve)

# Vérification de  l'état de connexion avant d'afficher la page
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Veuillez vous connecter pour accéder à cette page.")
    st.stop()

st.set_page_config(layout="wide")
st.title("Analyse de Performance Globale du Modèle")
st.markdown("Cette section présente les métriques et les courbes de performance du modèle XGBoost optimisé sur l'ensemble de test.")

# ----------------Chargement des données de performance----------------
@st.cache_resource
def load_performance_data():
    try:
        performance_data = joblib.load('model_performance_data.pkl')
    
        return performance_data
    except FileNotFoundError:
        st.error("Les données de performance du modèle ('model_performance_data.pkl') n'ont pas été trouvées. Veuillez exécuter la phase 5 dans le notebook pour les générer.")
        st.stop()

performance_data = load_performance_data()

# Récupération des données spécifiques du modèle optimisé
roc_auc_xgb_tuned = performance_data['roc_auc_tuned']
auc_pr_xgb_tuned = performance_data['auc_pr_tuned']
fpr_xgb_tuned = performance_data['fpr_tuned']
tpr_xgb_tuned = performance_data['tpr_tuned']
precision_xgb_tuned = performance_data['precision_tuned']
recall_xgb_tuned = performance_data['recall_tuned']

st.subheader("Métriques Clés du Modèle XGBoost Optimisé")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="ROC AUC", value=f"{roc_auc_xgb_tuned:.4f}", delta="> 0.5 (bien)")
with col2:
    st.metric(label="AUC Precision-Recall", value=f"{auc_pr_xgb_tuned:.4f}", delta="Plus élevé = mieux")
with col3:
    st.metric(label="Précision Moyenne", value=f"{np.mean(precision_xgb_tuned):.4f}", delta="À interpréter avec rappel")

st.markdown("---")

st.subheader("Courbes de Performance")

col_roc, col_pr = st.columns(2)

with col_roc:
    st.write("#### Courbe ROC (Receiver Operating Characteristic)")
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    ax_roc.plot(fpr_xgb_tuned, tpr_xgb_tuned, label=f'XGBoost Optimisé (AUC = {roc_auc_xgb_tuned:.2f})', color='blue')
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Classifieur Aléatoire (AUC = 0.50)')
    ax_roc.set_xlabel('Taux de Faux Positifs (1 - Spécificité)')
    ax_roc.set_ylabel('Taux de Vrais Positifs (Rappel)')
    ax_roc.set_title('Courbe ROC')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True)
    st.pyplot(fig_roc)
    st.info("La courbe ROC montre la capacité du modèle à distinguer les classes. Un AUC proche de 1 indique une excellente discrimination.")

with col_pr:
    st.write("#### Courbe Precision-Recall")
    fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
    ax_pr.plot(recall_xgb_tuned, precision_xgb_tuned, label=f'XGBoost Optimisé (AUC PR = {auc_pr_xgb_tuned:.2f})', color='green')
    ax_pr.set_xlabel('Rappel')
    ax_pr.set_ylabel('Précision')
    ax_pr.set_title('Courbe Precision-Recall')
    ax_pr.legend(loc='lower left')
    ax_pr.grid(True)
    st.pyplot(fig_pr)
    st.info("La courbe Precision-Recall est particulièrement utile pour les datasets déséquilibrés. Un AUC PR élevé est un bon signe.")
