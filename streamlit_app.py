import os
import requests
import streamlit as st

# ------------------------------------------------------------------
# Configuration : URL et clé de la Function Azure
# ------------------------------------------------------------------
FUNCTION_URL = (
    st.secrets.get("FUNCTION_URL") or os.getenv("FUNCTION_URL") or ""
)

# Liste de user_id d'exemple (à adapter selon votre dataset)
USER_IDS = [
    1001,
    1002,
    1003,
    1010,
    1025,
    1050,
    1100,
    1200,
    1250,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
    2100,
    2200,
]

st.set_page_config(page_title="Recommandations SVD", page_icon="🧠")
st.title("🎯 Générateur de recommandations via Azure Functions")

if not FUNCTION_URL:
    st.warning(
        "Configurez d'abord l'URL de la Function Azure (variable FUNCTION_BASE_URL). "
        "Utilisez st.secrets, un fichier .streamlit/secrets.toml ou les variables d'environnement."
    )

col1, col2 = st.columns(2)
with col1:
    user_id = st.selectbox("Sélectionner un user_id", USER_IDS)

with col2:
    top_k = st.number_input("Nombre d'articles à recommander", min_value=1, max_value=20, value=5)

st.divider()

if st.button("Générer les recommandations", type="primary"):
    if not FUNCTION_URL:
        st.error("URL de la Function Azure manquante.")
    else:
        params = {"user_id": user_id, "top_k": top_k}

        try:
            response = requests.get(FUNCTION_URL, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                recs = data.get("recommendations", [])
                st.success(f"Recommandations pour l'utilisateur {user_id} :")
                if recs:
                    for idx, article_id in enumerate(recs, start=1):
                        st.write(f"{idx}. Article {article_id}")
                else:
                    st.info("Aucune recommandation retournée.")
            else:
                st.error(f"Erreur HTTP {response.status_code} : {response.text}")
        except requests.RequestException as exc:
            st.error(f"Erreur lors de l'appel à la Function : {exc}")

