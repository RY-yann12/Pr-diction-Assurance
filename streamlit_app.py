# streamlit_app/streamlit_app.py
import streamlit as st

# Stockage simple de l'état de connexion (à des fins de démonstration)
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login_page():
    st.set_page_config(layout="centered")
    st.title("Bienvenue dans l'Application de Prédiction de Sinistres Assurantiels")
    st.markdown("Veuillez vous connecter pour accéder aux outils d'analyse et de prédiction.")

    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submit_button = st.form_submit_button("Se Connecter")

        if submit_button:
            # Vérification simple (pour la démo, utilisateur/mot de passe fixes)
            if username == "test" and password == "data":
                st.session_state['logged_in'] = True
                st.success("Connexion réussie ! Redirection...")
                st.rerun()
# Rafraîchit la page pour afficher les pages
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")

if not st.session_state['logged_in']:
    login_page()
else:
    st.sidebar.success("Connecté en tant que Generali")
    st.sidebar.button("Déconnexion", on_click=lambda: st.session_state.update({'logged_in': False}))
    st.markdown("# Tableau de Bord Principal")
    st.markdown("Utilisez le menu à gauche pour naviguer entre les différentes sections de l'application.")
    st.image("https://www.generali.fr/wp-content/uploads/2021/04/generali-logo.png", width=200) # Exemple de logo
    st.write("""
    Analyses des données clées.
    """)
#
#st.write("""
    #Bienvenue sur le tableau de bord de l'application de prédiction de la probabilité de sinistre.
    #Cette plateforme met à votre disposition des outils puissants basés sur l'apprentissage machine
    #pour évaluer le risque client, analyser la performance du modèle et comprendre les facteurs clés.
    #""")    




    st.subheader("Mes Objectifs de l'Application :")
    st.markdown("""
    *   **Prédiction du risque :** Évaluer la probabilité qu'un client fasse un sinistre.
    *   **Optimisation de la tarification :** Ajuster les primes en fonction du profil de risque.
    *   **Gestion proactive des sinistres :** Identifier les clients à risque pour des actions préventives.
    *   **Amélioration de la connaissance client :** Comprendre les facteurs qui influencent le risque.
    """)
    st.markdown("---")
    st.info("Menu de gauche pour la Navigation.")