Identifiant de connection à l'application: test, 
Code de connection: data
------------------------------------------------------------------------------------------------
# Prédiction de Sinistres pour une Entreprise d'Assurance

Ce projet propose une application de machine learning permettant de prédire la probabilité qu'un client réalise un sinistre dans l'année à venir. Il s'appuie sur des modèles avancés (XGBoost, Random Forest, Régression Logistique) et propose une interface utilisateur via Streamlit.

## Structure du projet

- `app.py` : Application principale Streamlit (prédiction, visualisation globale).
- `streamlit_app/` : Dossier contenant l'application modulaire et les ressources :
  - `streamlit_app.py` : Tableau de bord principal et gestion de la connexion.
  - `pages/` : Pages Streamlit (Prédiction, Performance, Sensibilité).
  - `shap_utils.py` : Fonctions d'interprétabilité (SHAP). "à développer" 
  - Fichiers modèles et encodage (`best_xgb_model.pkl`, `onehot_encoder.pkl`, etc.).
  - `requirements.txt` : Dépendances Python.

## Installation

1. **Cloner le dépôt**  
   ```sh
   git clone <lien_du_repo>
   cd Data_2025_prédiction-sinistre

Installer les dépendances
Utilise un environnement virtuel :
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r streamlit_app/requirements.txt
------------------------------------------------------------------------------------------------
Lancement de l'application
Pour démarrer l'application Streamlit :
Streamlit run 
------------------------------------------------------------------------------------------------
Navigue ensuite dans le menu à gauche pour accéder aux différentes pages :

Prédiction : Saisis les données d'un client pour obtenir sa probabilité de sinistre.
Performance : Visualise les métriques et courbes ROC/PR du modèle.
Sensibilité : Analyse l'importance des caractéristiques et l'interprétabilité locale (SHAP).
Fonctionnalités
Prédiction personnalisée : Interface pour entrer les données d'un client et obtenir une prédiction.
Analyse de performance : Affichage des scores AUC, courbes ROC et Precision-Recall.
Interprétabilité : Visualisation des facteurs influents (importance globale et locale via SHAP).
Sécurité : Connexion requise pour accéder aux pages (identifiant de démo : test / mot de passe : data).
-----------------------------------------------------------------------------------------------

Fichiers importants
Modèle XGBoost optimisé : streamlit_app/best_xgb_model.pkl
Encodage OneHot : streamlit_app/onehot_encoder.pkl
Standardisation : streamlit_app/standard_scaler.pkl
Données de test : streamlit_app/X_y_test.pkl
Données de performance : streamlit_app/model_performance_data.pkl
