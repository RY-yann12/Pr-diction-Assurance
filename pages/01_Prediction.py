# streamlit_app/pages/01_Prediction.py
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Vérifie l'état de connexion avant d'afficher la page
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Veuillez vous connecter pour accéder à cette page.")
    st.stop() # Arrête l'exécution de la page

st.set_page_config(layout="wide")
st.title("Prédiction de la Probabilité de Sinistre")
st.markdown("Entrez les données d'un client pour obtenir une prédiction de sa probabilité de sinistre.")

# --- Chargement des composants (fonctions) ---
@st.cache_resource
def load_prediction_components():
    try:
        model = joblib.load('best_xgb_model.pkl')
        encoder = joblib.load('onehot_encoder.pkl')
        scaler = joblib.load('standard_scaler.pkl')
        model_features = joblib.load('model_features.pkl')
        return model, encoder, scaler, model_features
    except FileNotFoundError as e:
        st.error(f"Erreur de chargement des fichiers nécessaires. Assurez-vous que '{e.filename}' est dans le dossier 'streamlit_app/'.")
        st.stop() # Arrête l'exécution si les fichiers ne sont pas trouvés

model, encoder, scaler, model_features = load_prediction_components()

# --- Fonctions de Prétraitement pour une Nouvelle Entrée ---
def preprocess_new_data(input_data_df, encoder, scaler, model_features):
    # Appliquer le Feature Engineering (comme dans la Phase 3)
    input_data_df['prime_sur_revenu'] = input_data_df['prime_annuelle'] / input_data_df['revenu_mensuel']
    input_data_df['prime_sur_revenu'].replace([np.inf, -np.inf], np.nan, inplace=True)
    input_data_df['prime_sur_revenu'].fillna(0, inplace=True)
    input_data_df['age_groupe'] = pd.cut(input_data_df['age'], bins=[18, 25, 40, 60, 85],
                                        labels=['Jeune', 'Adulte_Jeune', 'Adulte_Moyen', 'Senior'], right=False)
    
    # Gérer les colonnes catégorielles
    categorical_features_to_encode = [
        'sexe', 'situation_familiale', 'localisation_cat', 'type_contrat', 'age_groupe'
    ]
    
    # Appliquer l'encodage One-Hot
    encoded_features = encoder.transform(input_data_df[categorical_features_to_encode])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features_to_encode))
    
    # Supprimer les colonnes catégorielles originales et concaténer les encodées
    processed_df = pd.concat([input_data_df.drop(columns=categorical_features_to_encode), encoded_df], axis=1)
    
    # Réorganiser et ajouter les colonnes manquantes
    final_input_df = pd.DataFrame(columns=model_features)
    for col in model_features:
        if col in processed_df.columns:
            final_input_df[col] = processed_df[col]
        else:
            final_input_df[col] = 0
    
    # Appliquer le Scaling sur les colonnes numériques (sauf les binaires/ordinales qui ne sont pas scalées)
    numerical_cols_to_scale = [col for col in model_features if col in scaler.feature_names_in_]
    final_input_df[numerical_cols_to_scale] = scaler.transform(final_input_df[numerical_cols_to_scale])
    
    return final_input_df

# --- Interface utilisateur ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Informations Démographiques & Contrat")
    age = st.slider("Âge", 18, 85, 35)
    sexe = st.selectbox("Sexe", ['M', 'F'])
    situation_familiale = st.selectbox("Situation Familiale", ['Célibataire', 'Marié(e)', 'Divorcé(e)', 'Veuf(ve)'])
    localisation_cat = st.selectbox("Catégorie de Localisation", ['Urbaine', 'Semi-Urbaine', 'Rurale'])
    taux_urbanisation_zone = st.slider("Taux d'Urbanisation de la Zone (0.0=Rurale, 1.0=Urbaine)", 0.0, 1.0, 0.5)
    type_contrat = st.selectbox("Type de Contrat", ['Auto', 'Habitation', 'Sante'])
    anciennete_contrat = st.slider("Ancienneté Contrat (années)", 0, 25, 5)
    nb_produits = st.slider("Nombre de Produits d'Assurance", 1, 6, 2)

with col2:
    st.subheader("Informations Financières & Historique")
    prime_annuelle = st.number_input("Prime Annuelle (€)", 100.0, 5000.0, 1000.0)
    historique_sinistres_precedents = st.slider("Sinistres sur 3 dernières années", 0, 5, 0)
    loyal_client = st.selectbox("Client Fidèle (Oui/Non)", [0, 1], format_func=lambda x: 'Oui' if x == 1 else 'Non')
    score_credit = st.slider("Score de Crédit", 300, 850, 650)
    revenu_mensuel = st.number_input("Revenu Mensuel (€)", 500.0, 10000.0, 2500.0)
    nombre_contraventions_passees = st.slider("Nombre de Contraventions Passées", 0, 10, 0)
    interaction_digitale_freq = st.selectbox("Fréquence Interaction Digitale", [0, 1, 2], format_func=lambda x: ['Faible', 'Moyenne', 'Élevée'][x])
    produits_croises_assurance = st.selectbox("Produits Croisés Assurance (Oui/Non)", [0, 1], format_func=lambda x: 'Oui' if x == 1 else 'Non')

st.markdown("---")
if st.button("Prédire la Probabilité de Sinistre", help="Cliquez pour obtenir la prédiction basée sur les données saisies."):
    input_df = pd.DataFrame([[
        age, sexe, situation_familiale, localisation_cat, taux_urbanisation_zone,
        type_contrat, anciennete_contrat, nb_produits, prime_annuelle,
        historique_sinistres_precedents, loyal_client, score_credit,
        revenu_mensuel, nombre_contraventions_passees, interaction_digitale_freq,
        produits_croises_assurance
    ]], columns=[
        'age', 'sexe', 'situation_familiale', 'localisation_cat', 'taux_urbanisation_zone',
        'type_contrat', 'anciennete_contrat', 'nb_produits', 'prime_annuelle',
        'historique_sinistres_precedents', 'loyal_client', 'score_credit',
        'revenu_mensuel', 'nombre_contraventions_passees', 'interaction_digitale_freq',
        'produits_croises_assurance'
    ])

    processed_input = preprocess_new_data(input_df.copy(), encoder, scaler, model_features)
    
    prediction_proba = model.predict_proba(processed_input)[0, 1]

    st.subheader(f"Probabilité de Sinistre pour ce Client : **{prediction_proba:.2%}**")

    if prediction_proba > 0.5:
        st.error(" Ce client présente une probabilité de sinistre élevée. Recommandation : Analyse approfondie ou ajustement tarifaire.")
    else:
        st.success("Ce client présente une probabilité de sinistre faible à modérée. Recommandation : Suivi standard.")

    st.markdown("---")
    st.subheader("Facteurs clés influençant cette prédiction (Interprétabilité locale)")
    st.info(" Utilisation d'une intégartion de SHAT pout une explication plus détaillée (méthode puissante pour expliquer" \
            "les prédictions des modèles de Machine Learning. Basée sur la théorie des jeux coopératifs et les valeurs de Shapley).")
    st.write("Pour l'instant, veuillez vous référer à la page 'Analyse de Sensibilité' pour les facteurs globaux.")