# ==============================
# 1D DC Forward Modelling (SimPEG)
# Streamlit app — Schlumberger + Wenner
# ==============================

# --- Core scientific libraries ---
import numpy as np                  # Calcul numérique
import pandas as pd                 # Manipulation de tableaux
import matplotlib.pyplot as plt     # Librairie de PLOT (remplace Plotly)
import streamlit as st              # Interface Web

# --- SimPEG modules for DC resistivity ---
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps

# --- Outils Matplotlib pour les axes log-log ---
from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter

# ---------------------------
# 1) CONFIGURATION DE LA PAGE
# ---------------------------

st.set_page_config(page_title="Modélisation 1D DC (SimPEG)", page_icon="⚡", layout="wide")

st.title("⚡ 1D DC Resistivity — Modélisation Robuste (Schlumberger vs Wenner)")
st.markdown(
    "Configurez un modèle de sous-sol stratifié. L'application calcule la **Résistivité Apparente** "
    "et les **Facteurs Géométriques (K)** pour les dispositifs **Schlumberger** et **Wenner**."
)

# ==============================================================
# 2) BARRE LATÉRALE — INPUTS UTILISATEUR
# ==============================================================

with st.sidebar:
    st.header("1. Géométrie (AB/2)")

    colA1, colA2 = st.columns(2)
    with colA1:
        ab2_min = st.number_input(
            "AB/2 min (m)", min_value=0.1, value=1.0, step=0.1, format="%.1f"
        )
    with colA2:
        ab2_max = st.number_input(
            "AB/2 max (m)", min_value=ab2_min + 1.0, value=500.0, step=10.0, format="%.1f"
        )

    n_stations = st.slider("Nombre de mesures", 10, 100, 35)

    st.info(
        "**Schlumberger :** MN/2 $\\approx$ 10% de AB/2.\n\n"
        "**Wenner :** Espacement égal $a$, où AB/2 = 1.5$a$."
    )

    st.divider()
    st.header("2. Modèle de Terre")

    n_layers = st.slider("Nombre de couches", 2, 5, 3)

    # Valeurs par défaut
    default_rho = [100.0, 20.0, 500.0, 100.0, 1000.0]
    default_thk = [10.0, 20.0, 50.0, 50.0]

    layer_rhos = []
    thicknesses = []
    
    for i in range(n_layers):
        st.markdown(f"**Couche {i+1}**")
        c1, c2 = st.columns(2)
        with c1:
            r = st.number_input(
                f"ρ{i+1} (Ω·m)", min_value=0.1, value=float(default_rho[i]), key=f"rho_{i}"
            )
            layer_rhos.append(r)
        with c2:
            if i < n_layers - 1:
                t = st.number_input(
                    f"Épaisseur {i+1} (m)", min_value=0.1, value=float(default_thk[i]), key=f"thk_{i}"
                )
                thicknesses.append(t)
            else:
                st.write("(Demi-espace)")

thicknesses = np.array(thicknesses)
rho_layers = np.array(layer_rhos)

# ==============================================================
# 3) CALCUL DE LA GÉOMÉTRIE & FACTEURS K
# ==============================================================

AB2 = np.geomspace(ab2_min, ab2_max, n_stations)

# Schlumberger
MN2_s = np.minimum(0.1 * AB2, 0.45 * AB2) 
K_schlumberger = np.pi * (AB2**2 - MN2_s**2) / (2 * MN2_s)

# Wenner
a_wenner = (2.0/3.0) * AB2
K_wenner = 2 * np.pi * a_wenner

# ==============================================================
# 4) SIMULATION SIMPEG
# ==============================================================

src_list_s = []
src_list_w = []
eps = 1e-4

for i in range(n_stations):
    # Sources Schlumberger
    A_s = np.r_[-AB2[i], 0., 0.]
    B_s = np.r_[+AB2[i], 0., 0.]
    M_s = np.r_[-MN2_s[i] + eps, 0., 0.]
    N_s = np.r_[+MN2_s[i] - eps, 0., 0.]
    rx_s = dc.receivers.Dipole(M_s, N_s, data_type="apparent_resistivity")
    src_list_s.append(dc.sources.Dipole([rx_s], A_s, B_s))

    # Sources Wenner
    a = a_wenner[i]
    A_w = np.r_[-1.5*a, 0., 0.]
    M_w = np.r_[-0.5*a, 0., 0.]
    N_w = np.r_[+0.5*a, 0., 0.]
    B_w = np.r_[+1.5*a, 0., 0.]
    rx_w = dc.receivers.Dipole(M_w, N_w, data_type="apparent_resistivity")
    src_list_w.append(dc.sources.Dipole([rx_w], A_w, B_w))

survey_s = dc.Survey(src_list_s)
survey_w = dc.Survey(src_list_w)

rho_map = maps.IdentityMap(nP=len(rho_layers))

sim_s = dc.simulation_1d.Simulation1DLayers(
    survey=survey_s, rhoMap=rho_map, thicknesses=thicknesses
)
sim_w = dc.simulation_1d.Simulation1DLayers(
    survey=survey_w, rhoMap=rho_map, thicknesses=thicknesses
)

try:
    d_s = sim_s.dpred(rho_layers)
    d_w = sim_w.dpred(rho_layers)
    success = True
except Exception as e:
    st.error(f"Erreur de simulation : {e}")
    success = False

# ==============================================================
# 5) AFFICHAGE DES RÉSULTATS (Matplotlib)
# ==============================================================

if success:
    
    # Préparation du DataFrame pour l'export
    df = pd.DataFrame({
        "AB/2 (m)": AB2,
        "Schlumberger ρa": d_s,
        "Wenner ρa": d_w,
        "K Schlumberger": K_schlumberger,
        "K Wenner": K_wenner,
        "MN/2 Schlumberger": MN2_s,
        "a Wenner": a_wenner
    })

    # --- Onglets ---
    tab1, tab2, tab3 = st.tabs(["📈 Courbes de Sondage", "📐 Facteurs Géométriques (K)", "🔢 Données & Modèle"])

    # --- ONGLET 1 : Courbes de Sondage (Matplotlib) ---
    with tab1:
        st.subheader("Courbes de Sondage Électrique (Log-Log)")
        
        fig, ax = plt.subplots(figsize=(8, 6))

        # Tracé des deux courbes en échelle log-log
        ax.loglog(AB2, d_s, "o-", label="Schlumberger ρₐ")
        ax.loglog(AB2, d_w, "s--", label="Wenner ρₐ")

        # Configuration des limites Y (étendu aux décades entières)
        ymin = np.minimum(d_s.min(), d_w.min())
        ymax = np.maximum(d_s.max(), d_w.max())
        ymin = 10 ** np.floor(np.log10(ymin))
        ymax = 10 ** np.ceil(np.log10(ymax))
        ax.set_ylim(ymin, ymax)

        # Configuration des axes log-log (pour une meilleure lisibilité)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=True))
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=True))
        ax.xaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, which="both", ls=":", alpha=0.7)

        ax.set_xlabel("AB/2 (m)")
        ax.set_ylabel("Résistivité Apparente (Ω·m)")
        ax.legend()
        st.pyplot(fig, clear_figure=True)


    # --- ONGLET 2 : Facteurs Géométriques (K) ---
    with tab2:
        st.subheader("Facteurs Géométriques (K)")
        st.markdown("Visualisation Log-Log des facteurs $K$ en fonction de $AB/2$.")
        
        fig_k, ax_k = plt.subplots(figsize=(8, 6))
        ax_k.loglog(AB2, K_schlumberger, "o-", label="K Schlumberger")
        ax_k.loglog(AB2, K_wenner, "s--", label="K Wenner")
        
        ax_k.set_xlabel("AB/2 (m)")
        ax_k.set_ylabel("Facteur K (m)")
        ax_k.grid(True, which="both", ls=":", alpha=0.7)
        ax_k.legend()
        st.pyplot(fig_k, clear_figure=True)


    # --- ONGLET 3 : Tableau & Modèle ---
    with tab3:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Modèle de Couches (Viz)")
            fig2, ax2 = plt.subplots(figsize=(4, 5))
            
            # Représentation du modèle de terre
            if len(thicknesses):
                interfaces = np.r_[0.0, np.cumsum(thicknesses)]
            else:
                interfaces = np.r_[0.0]

            z_bottom = interfaces[-1] + max(interfaces[-1] * 0.3, 10.0)

            tops = np.r_[interfaces, interfaces[-1]]
            bottoms = np.r_[interfaces[1:], z_bottom]

            for i in range(n_layers):
                ax2.fill_betweenx([tops[i], bottoms[i]], 0, rho_layers[i], alpha=0.35)
                ax2.text(
                    rho_layers[i] * 1.05,
                    (tops[i] + bottoms[i]) / 2,
                    f"{rho_layers[i]:.1f} Ω·m",
                    va="center",
                    fontsize=9,
                )

            ax2.invert_yaxis()
            ax2.set_xlabel("Résistivité (Ω·m)")
            ax2.set_ylabel("Profondeur (m)")
            ax2.grid(True, ls=":")
            ax2.set_title("Modèle Vrai")
            st.pyplot(fig2, clear_figure=True)

        with col2:
            st.subheader("Export des données")
            
            # Affichage du DataFrame
            st.dataframe(df, use_container_width=True)
            
            # Bouton de téléchargement
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Télécharger CSV (Résultats + Facteurs K)",
                csv,
                "resultats_sondage_electrique.csv",
                "text/csv",
                type="primary"
            )
