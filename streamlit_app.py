# ==============================
# 1D DC Forward Modelling (SimPEG)
# App Streamlit — Schlumberger + Wenner
# ==============================

# --- Core scientific libraries ---
import numpy as np                  # Calcul numérique
import pandas as pd                 # Manipulation de tableaux
import streamlit as st              # Interface Web
import plotly.graph_objects as go   # GRAPHIQUES INTERACTIFS

# --- SimPEG modules for DC resistivity ---
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps

# ---------------------------
# 1) CONFIGURATION DE LA PAGE
# ---------------------------

st.set_page_config(page_title="Modélisation 1D DC (SimPEG)", page_icon="⚡", layout="wide")

st.title("⚡ 1D DC Resistivity — Modélisation Interactive (Schlumberger vs Wenner)")
st.markdown(
    "Configurez un modèle de sous-sol stratifié. Cette application calcule la **Résistivité Apparente** "
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

    # Entrées dynamiques pour les couches
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

# Conversion en numpy arrays
thicknesses = np.array(thicknesses)
rho_layers = np.array(layer_rhos)

# ==============================================================
# 3) CALCUL DE LA GÉOMÉTRIE & FACTEURS K
# ==============================================================

# Distribution logarithmique de AB/2
AB2 = np.geomspace(ab2_min, ab2_max, n_stations)

# --- Géométrie Schlumberger ---
MN2_s = np.minimum(0.1 * AB2, 0.45 * AB2) 
# K = pi * ( (AB/2)^2 - (MN/2)^2 ) / MN ; MN = 2 * MN/2
K_schlumberger = np.pi * (AB2**2 - MN2_s**2) / (2 * MN2_s)

# --- Géométrie Wenner ---
a_wenner = (2.0/3.0) * AB2
# K = 2 * pi * a
K_wenner = 2 * np.pi * a_wenner

# ==============================================================
# 4) SIMULATION SIMPEG
# ==============================================================

src_list_s = []
src_list_w = []

for i in range(n_stations):
    # Sources Schlumberger
    A_s = np.r_[-AB2[i], 0., 0.]
    B_s = np.r_[+AB2[i], 0., 0.]
    M_s = np.r_[-MN2_s[i], 0., 0.]
    N_s = np.r_[+MN2_s[i], 0., 0.]
    
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

# Exécution du modèle direct
try:
    d_s = sim_s.dpred(rho_layers)
    d_w = sim_w.dpred(rho_layers)
    success = True
except Exception as e:
    st.error(f"Erreur de simulation : {e}")
    success = False

# ==============================================================
# 5) AFFICHAGE INTERACTIF & RÉSULTATS
# ==============================================================

if success:
    # Création du DataFrame
    df = pd.DataFrame({
        "AB/2 (m)": AB2,
        "Schlumberger ρa": d_s,
        "Wenner ρa": d_w,
        "K Schlumberger": K_schlumberger,
        "K Wenner": K_wenner,
        "MN/2 Schlumberger": MN2_s,
        "a Wenner": a_wenner
    })

    # --- DISPOSITION EN ONGLETS ---
    tab1, tab2, tab3 = st.tabs(["📈 Courbes de Sondage", "📐 Facteurs Géométriques (K)", "🔢 Données & Modèle"])

    # --- ONGLET 1 : Courbes de Sondage (Plotly) ---
    with tab1:
        fig_rho = go.Figure()

        # Trace Schlumberger
        fig_rho.add_trace(go.Scatter(
            x=AB2, y=d_s, mode='lines+markers', name='Schlumberger',
            marker=dict(symbol='circle', size=7),
            hovertemplate="<b>Schlumberger</b><br>AB/2: %{x:.2f} m<br>ρa: %{y:.2f} Ω·m<extra></extra>"
        ))

        # Trace Wenner
        fig_rho.add_trace(go.Scatter(
            x=AB2, y=d_w, mode='lines+markers', name='Wenner',
            line=dict(dash='dash'),
            marker=dict(symbol='square', size=7),
            hovertemplate="<b>Wenner</b><br>AB/2: %{x:.2f} m<br>ρa: %{y:.2f} Ω·m<extra></extra>"
        ))

        fig_rho.update_layout(
            title="Courbes de Sondage Électrique (Log-Log)",
            xaxis_title="AB/2 (m)",
            yaxis_title="Résistivité Apparente (Ω·m)",
            xaxis_type="log",
            yaxis_type="log",
            hovermode="x unified",
            height=600,
            template="plotly_white"
        )
        fig_rho.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', minor=dict(showgrid=True))
        fig_rho.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', minor=dict(showgrid=True))
        
        st.plotly_chart(fig_rho, use_container_width=True)

    # --- ONGLET 2 : Facteurs Géométriques (Plotly) ---
    with tab2:
        st.markdown("Le facteur géométrique $K$ relie la mesure à la résistivité : $V = \\rho_a \cdot I / K$")
        fig_k = go.Figure()

        fig_k.add_trace(go.Scatter(
            x=AB2, y=K_schlumberger, mode='lines', name='K (Schlumberger)',
            line=dict(color='orange', width=3)
        ))
        fig_k.add_trace(go.Scatter(
            x=AB2, y=K_wenner, mode='lines', name='K (Wenner)',
            line=dict(color='green', width=3, dash='dot')
        ))

        fig_k.update_layout(
            title="Facteur Géométrique K vs Espacement",
            xaxis_title="AB/2 (m)",
            yaxis_title="Facteur K (m)",
            xaxis_type="log",
            yaxis_type="log",
            hovermode="x unified",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig_k, use_container_width=True)

    # --- ONGLET 3 : Tableau & Visualisation du Modèle ---
    with tab3:
        c_left, c_right = st.columns([1, 2])

        # Visualisation du modèle (Step Plot)
        with c_left:
            st.subheader("Modèle de Couches")
            
            # Construction des tableaux de profondeur pour l'affichage en escalier
            plot_depths = [0]
            plot_rhos = [rho_layers[0]]
            
            current_z = 0
            for i in range(len(thicknesses)):
                current_z += thicknesses[i]
                # Coin du "step"
                plot_depths.append(current_z)
                plot_rhos.append(rho_layers[i])
                # Marche du "step"
                plot_depths.append(current_z)
                plot_rhos.append(rho_layers[i+1])
            
            # Extension de la dernière couche vers "l'infini"
            plot_depths.append(current_z * 1.5 + 10)
            plot_rhos.append(rho_layers[-1])

            fig_model = go.Figure()
            fig_model.add_trace(go.Scatter(
                x=plot_rhos, y=plot_depths, 
                mode='lines', line_shape='hv', 
                fill='tozerox', fillcolor='rgba(0,100,80,0.2)'
            ))
            fig_model.update_yaxes(autorange="reversed", title="Profondeur (m)")
            fig_model.update_xaxes(title="Résistivité (Ω·m)", type="log")
            fig_model.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=400,
                showlegend=False,
                title="Résistivité Vraie vs Profondeur"
            )
            st.plotly_chart(fig_model, use_container_width=True)

        with c_right:
            st.subheader("Exporter les données")
            st.dataframe(df, height=350)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Télécharger CSV (Résultats + Facteurs K)",
                csv,
                "resultats_sondage_electrique.csv",
                "text/csv",
                type="primary"
            )
