# ==============================
# 1D DC Forward Modelling (SimPEG)
# Streamlit app — Schlumberger + Wenner
# Version pédagogique pour étudiants
# ==============================

# --- Core scientific libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter

# --- SimPEG modules for DC resistivity ---
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps

# ---------------------------
# 1) PAGE SETUP & HEADER
# ---------------------------

st.set_page_config(page_title="1D DC Resistivity (Pédagogique)", page_icon="🎓", layout="wide")

st.title("🎓 Modélisation DC 1D - Pédagogique pour étudiants")
st.markdown("""
**Objectif pédagogique** : Comprendre les principes des sondages électriques et 
comparer les configurations Schlumberger et Wenner.

### Concepts clés :
- **Résistivité apparente (ρₐ)** : Résistivité "mesurée" pour un milieu hétérogène
- **Facteur de géométrie (K)** : Dépend uniquement de la disposition des électrodes
- **AB/2** : Demi-distance entre électrodes de courant
- **MN/2** : Demi-distance entre électrodes de potentiel
""")

# ==============================================================
# 2) SIDEBAR — PARAMÈTRES D'ENTRÉE
# ==============================================================

with st.sidebar:
    st.header("⚙️ Configuration géométrique")
    
    # Section pédagogique sur les configurations
    with st.expander("📚 Aide sur les configurations"):
        st.markdown("""
        **Schlumberger** :
        - Électrodes MN proches du centre
        - MN/2 ≪ AB/2 (typiquement 10%)
        - Avantage : meilleur rapport signal/bruit pour les grands AB/2
        
        **Wenner** :
        - Électrodes équidistantes : AM = MN = NB = a
        - AB = 3a, MN = a
        - Avantage : symétrie parfaite, calcul simple du facteur K
        """)
    
    # AB/2 range
    colA1, colA2 = st.columns(2)
    with colA1:
        ab2_min = st.number_input(
            "AB/2 min (m)", 
            min_value=0.1, 
            value=5.0, 
            step=0.1, 
            format="%.2f",
            help="Distance minimale entre les électrodes de courant"
        )
    with colA2:
        ab2_max = st.number_input(
            "AB/2 max (m)", 
            min_value=ab2_min + 0.1, 
            value=300.0, 
            step=1.0, 
            format="%.2f",
            help="Distance maximale entre les électrodes de courant"
        )
    
    n_stations = st.slider(
        "Nombre de stations", 
        min_value=8, 
        max_value=60, 
        value=25, 
        step=1,
        help="Nombre de points de mesure le long du profil"
    )
    
    st.divider()
    st.header("🏗️ Modèle de sous-sol")
    
    # Section pédagogique sur les modèles de sous-sol
    with st.expander("📚 Aide sur les modèles de sous-sol"):
        st.markdown("""
        **Modèle à couches** :
        - Chaque couche a une résistivité constante
        - La dernière couche est un demi-espace (épaisseur infinie)
        - La profondeur d'investigation augmente avec AB/2
        """)
    
    # Nombre de couches
    n_layers = st.slider(
        "Nombre de couches", 
        2, 5, 4,
        help="Total des couches (la dernière est un demi-espace)"
    )
    
    # Valeurs par défaut
    default_rho = [10.0, 30.0, 15.0, 50.0, 100.0][:n_layers]
    default_thk = [2.0, 8.0, 60.0, 120.0][:max(0, n_layers - 1)]
    
    # Résistivités
    st.subheader("Résistivités des couches")
    layer_rhos = []
    for i in range(n_layers):
        layer_rhos.append(
            st.number_input(
                f"ρ{i+1} (Ω·m)",
                min_value=0.1,
                value=float(default_rho[i]),
                step=0.1,
                help=f"Résistivité de la couche {i+1}"
            )
        )
    
    # Épaisseurs
    if n_layers > 1:
        st.subheader("Épaisseurs des couches")
        thicknesses = []
        for i in range(n_layers - 1):
            thicknesses.append(
                st.number_input(
                    f"h{i+1} (m)",
                    min_value=0.1,
                    value=float(default_thk[i]),
                    step=0.1,
                    help=f"Épaisseur de la couche {i+1}"
                )
            )
    else:
        thicknesses = []

thicknesses = np.r_[thicknesses] if len(thicknesses) else np.array([])

# ==============================================================
# 3) FONCTION POUR CALCULER LE FACTEUR K
# ==============================================================

def calculate_k_factor(config_type, L, a_s=None):
    """
    Calcule le facteur de géométrie K pour différentes configurations
    
    Formules :
    - Schlumberger : K = π * (L²/a - a) (approximation pour a << L)
    - Wenner : K = 2πa
    """
    if config_type == "schlumberger":
        if a_s is None:
            a_s = 0.1 * L  # Valeur par défaut
        # Formule exacte pour Schlumberger
        AB = 2 * L
        MN = 2 * a_s
        K = np.pi * ((AB/2)**2 - (MN/2)**2) / MN
        return K, a_s
    elif config_type == "wenner":
        a_w = (2.0 / 3.0) * L  # AB = 3a => a = (2/3)*L
        K = 2 * np.pi * a_w
        return K, a_w
    return None, None

# ==============================================================
# 4) CONSTRUCTION DE LA GÉOMÉTRIE ET CALCUL DES FACTEURS K
# ==============================================================

# Stations AB/2
AB2 = np.geomspace(ab2_min, ab2_max, n_stations)

# Initialisation des tableaux pour les facteurs K
K_schlumberger = np.zeros_like(AB2)
K_wenner = np.zeros_like(AB2)
MN2_schlumberger = np.zeros_like(AB2)

# Calcul des facteurs K pour chaque station
for i, L in enumerate(AB2):
    # Schlumberger
    a_s = np.minimum(0.10 * L, 0.49 * L)  # MN/2
    K_s, _ = calculate_k_factor("schlumberger", L, a_s)
    K_schlumberger[i] = K_s
    MN2_schlumberger[i] = a_s
    
    # Wenner
    K_w, _ = calculate_k_factor("wenner", L)
    K_wenner[i] = K_w

# ==============================================================
# 5) CONSTRUCTION DES SURVEYS
# ==============================================================

# Schlumberger
src_list_s = []
eps = 1e-6
for L, a_s in zip(AB2, MN2_schlumberger):
    A_s = np.r_[-L, 0.0, 0.0]
    B_s = np.r_[+L, 0.0, 0.0]
    M_s = np.r_[-(a_s - eps), 0.0, 0.0]
    N_s = np.r_[+(a_s - eps), 0.0, 0.0]
    
    rx_s = dc.receivers.Dipole(M_s, N_s, data_type="apparent_resistivity")
    src_s = dc.sources.Dipole([rx_s], A_s, B_s)
    src_list_s.append(src_s)

survey_s = dc.Survey(src_list_s)

# Wenner
src_list_w = []
for L in AB2:
    a_w = (2.0 / 3.0) * L
    A_w = np.r_[-1.5 * a_w, 0.0, 0.0]
    M_w = np.r_[-0.5 * a_w, 0.0, 0.0]
    N_w = np.r_[+0.5 * a_w, 0.0, 0.0]
    B_w = np.r_[+1.5 * a_w, 0.0, 0.0]
    
    rx_w = dc.receivers.Dipole(M_w, N_w, data_type="apparent_resistivity")
    src_w = dc.sources.Dipole([rx_w], A_w, B_w)
    src_list_w.append(src_w)

survey_w = dc.Survey(src_list_w)

# ==============================================================
# 6) MODÉLISATION DIRECTE
# ==============================================================

rho = np.r_[layer_rhos]
rho_map = maps.IdentityMap(nP=len(rho))

sim_s = dc.simulation_1d.Simulation1DLayers(
    survey=survey_s,
    rhoMap=rho_map,
    thicknesses=thicknesses,
)

sim_w = dc.simulation_1d.Simulation1DLayers(
    survey=survey_w,
    rhoMap=rho_map,
    thicknesses=thicknesses,
)

try:
    rho_app_s = sim_s.dpred(rho)
    rho_app_w = sim_w.dpred(rho)
    ok = True
except Exception as e:
    ok = False
    st.error(f"Erreur lors de la modélisation : {e}")

# ==============================================================
# 7) AFFICHAGE DES RÉSULTATS - INTERFACE PRINCIPALE
# ==============================================================

if ok:
    # Section pédagogique sur les calculs
    with st.expander("🎯 Concepts fondamentaux - Équations clés"):
        col_eq1, col_eq2 = st.columns(2)
        
        with col_eq1:
            st.markdown("""
            **Résistivité apparente** :
            ```
            ρₐ = K × (ΔV / I)
            ```
            où :
            - K : facteur de géométrie
            - ΔV : différence de potentiel mesurée
            - I : courant injecté
            """)
        
        with col_eq2:
            st.markdown("""
            **Facteurs de géométrie** :
            
            Schlumberger :
            ```
            K = π × [(AB/2)² - (MN/2)²] / (MN/2)
            ```
            
            Wenner (AM = MN = NB = a) :
            ```
            K = 2πa
            ```
            """)
    
    # ==============================================================
    # 8) VISUALISATION DES FACTEURS K
    # ==============================================================
    
    st.divider()
    st.subheader("📊 Facteurs de géométrie K")
    
    col_k1, col_k2 = st.columns(2)
    
    with col_k1:
        fig_k, ax_k = plt.subplots(figsize=(6, 4))
        ax_k.loglog(AB2, np.abs(K_schlumberger), 'o-', label='Schlumberger', color='blue')
        ax_k.loglog(AB2, np.abs(K_wenner), 's--', label='Wenner', color='red')
        
        ax_k.set_xlabel("AB/2 (m)")
        ax_k.set_ylabel("Facteur K (m)")
        ax_k.set_title("Facteur de géométrie vs AB/2")
        ax_k.grid(True, which="both", ls=":", alpha=0.7)
        ax_k.legend()
        st.pyplot(fig_k)
    
    with col_k2:
        # Tableau des valeurs K pour quelques points
        st.markdown("**Valeurs typiques du facteur K :**")
        indices = [0, n_stations//4, n_stations//2, 3*n_stations//4, -1]
        k_data = []
        for idx in indices:
            k_data.append({
                "AB/2 (m)": f"{AB2[idx]:.1f}",
                "K Schlumberger": f"{K_schlumberger[idx]:.1f}",
                "K Wenner": f"{K_wenner[idx]:.1f}",
                "MN/2 Schlumb.": f"{MN2_schlumberger[idx]:.2f}"
            })
        
        st.dataframe(pd.DataFrame(k_data), use_container_width=True)
        
        st.info("""
        **Observation** : 
        Le facteur K Wenner est constant pour une distance a donnée, 
        tandis que le facteur K Schlumberger varie avec AB/2.
        """)
    
    # ==============================================================
    # 9) COURBES DE SONDAGE
    # ==============================================================
    
    st.divider()
    st.subheader("📈 Courbes de sondage électrique")
    
    col_curves1, col_curves2 = st.columns([2, 1])
    
    with col_curves1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Courbes de résistivité apparente
        ax.loglog(AB2, rho_app_s, "o-", label="Schlumberger ρₐ", linewidth=2, markersize=6)
        ax.loglog(AB2, rho_app_w, "s--", label="Wenner ρₐ", linewidth=2, markersize=6)
        
        # Ajouter les résistivités vraies comme lignes de référence
        colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
        for i, rho_i in enumerate(rho):
            if i < n_layers - 1:
                depth_label = f"ρ{i+1} = {rho_i:.1f} Ω·m (jusqu'à {np.sum(thicknesses[:i]):.1f} m)"
            else:
                depth_label = f"ρ{i+1} = {rho_i:.1f} Ω·m (demi-espace)"
            ax.axhline(y=rho_i, color=colors[i], linestyle=':', alpha=0.5, label=depth_label)
        
        # Configuration du graphique
        ymin = np.minimum(rho_app_s.min(), rho_app_w.min())
        ymax = np.maximum(rho_app_s.max(), rho_app_w.max())
        ymin = 10 ** np.floor(np.log10(ymin))
        ymax = 10 ** np.ceil(np.log10(ymax))
        ax.set_ylim(ymin, ymax)
        
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=True))
        ax.yaxis.set_minor_formatter(NullFormatter())
        
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=True))
        ax.xaxis.set_minor_formatter(NullFormatter())
        
        ax.grid(True, which="both", ls=":", alpha=0.7)
        ax.set_xlabel("AB/2 (m) - Distance demi-électrodes courant")
        ax.set_ylabel("Résistivité apparente ρₐ (Ω·m)")
        ax.set_title("Comparaison Schlumberger vs Wenner - Courbes VES 1D")
        
        # Légende améliorée
        ax.legend(loc='best', fontsize=9)
        
        st.pyplot(fig)
    
    with col_curves2:
        # Informations pédagogiques
        st.markdown("### 🎯 Interprétation des courbes")
        
        st.info("""
        **Points clés à observer** :
        1. **Petits AB/2** : Sondent les couches superficielles
        2. **Grands AB/2** : Sondent les couches profondes
        3. **Différences entre configurations** :
           - Wenner : plus sensible aux couches intermédiaires
           - Schlumberger : meilleure résolution verticale
        """)
        
        # Calcul de la profondeur d'investigation approximative
        st.markdown("### 📏 Profondeur d'investigation")
        depth_approx = AB2 * 0.4  # Règle empirique
        df_depth = pd.DataFrame({
            "AB/2 (m)": AB2[[0, n_stations//4, n_stations//2, -1]],
            "Prof. approx. (m)": depth_approx[[0, n_stations//4, n_stations//2, -1]]
        })
        st.dataframe(df_depth, use_container_width=True)
        
        st.caption("*Profondeur approximative ≈ 0.4 × AB/2 (règle empirique)*")
    
    # ==============================================================
    # 10) VISUALISATION DU MODÈLE DE COUCHES
    # ==============================================================
    
    st.divider()
    st.subheader("🏗️ Modèle de sous-sol à couches")
    
    col_model1, col_model2 = st.columns([1, 1])
    
    with col_model1:
        # Diagramme des couches
        fig2, ax2 = plt.subplots(figsize=(5, 6))
        
        # Calcul des interfaces
        if len(thicknesses):
            interfaces = np.r_[0.0, np.cumsum(thicknesses)]
        else:
            interfaces = np.r_[0.0]
        
        z_bottom = interfaces[-1] * 1.5 if interfaces[-1] > 0 else 50.0
        
        # Dessin des couches
        for i in range(n_layers):
            top = interfaces[i] if i < len(interfaces) else interfaces[-1]
            bottom = interfaces[i+1] if i < len(interfaces)-1 else z_bottom
            
            # Rectangle coloré pour chaque couche
            ax2.fill_betweenx([top, bottom], 0, rho[i], 
                             alpha=0.4, color=plt.cm.tab10(i/n_layers))
            
            # Texte avec résistivité
            ax2.text(rho[i] * 1.1, (top + bottom) / 2,
                    f"ρ{i+1} = {rho[i]:.1f} Ω·m",
                    va='center', fontsize=9)
            
            # Ligne de séparation
            if i < len(interfaces) - 1:
                ax2.axhline(y=interfaces[i+1], color='black', linestyle='-', alpha=0.5)
        
        ax2.set_xlim(0, max(rho) * 1.3)
        ax2.invert_yaxis()
        ax2.set_xlabel("Résistivité (Ω·m)")
        ax2.set_ylabel("Profondeur (m)")
        ax2.set_title("Modèle de sous-sol")
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
    
    with col_model2:
        # Tableau détaillé du modèle
        st.markdown("### 📋 Paramètres du modèle")
        
        model_data = []
        cumulative_depth = 0.0
        
        for i in range(n_layers):
            if i < len(thicknesses):
                thickness = thicknesses[i]
                note = f"Couche {i+1}"
            else:
                thickness = "∞"
                note = "Demi-espace"
            
            model_data.append({
                "Couche": i+1,
                "Résistivité (Ω·m)": f"{rho[i]:.1f}",
                "Épaisseur (m)": f"{thickness}" if isinstance(thickness, str) else f"{thickness:.1f}",
                "Prof. sommet (m)": f"{cumulative_depth:.1f}",
                "Note": note
            })
            
            if i < len(thicknesses):
                cumulative_depth += thicknesses[i]
        
        st.dataframe(pd.DataFrame(model_data), use_container_width=True)
        
        # Informations sur les épaisseurs
        st.markdown("### 📊 Sommaire")
        st.metric("Nombre de couches", n_layers)
        st.metric("Profondeur totale des couches finies", 
                 f"{np.sum(thicknesses):.1f} m" if len(thicknesses) > 0 else "0 m")
        st.metric("Plage de résistivité", f"{min(rho):.1f} - {max(rho):.1f} Ω·m")
    
    # ==============================================================
    # 11) EXPORT DES DONNÉES ET CONCLUSION
    # ==============================================================
    
    st.divider()
    st.subheader("💾 Export des données")
    
    # Préparation des données pour export
    df_export = pd.DataFrame({
        'AB2_m': AB2,
        'MN2_Schlumberger_m': MN2_schlumberger,
        'K_Schlumberger_m': K_schlumberger,
        'RhoApp_Schlumberger_Ohmm': rho_app_s,
        'K_Wenner_m': K_wenner,
        'RhoApp_Wenner_Ohmm': rho_app_w,
        'Depth_Approx_m': depth_approx
    })
    
    # Boutons d'export
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.download_button(
            label="📥 Télécharger données (CSV)",
            data=df_export.to_csv(index=False, sep=';').encode('utf-8'),
            file_name="donnees_sondage_electrique.csv",
            mime="text/csv",
            help="Exporte toutes les données de simulation"
        )
    
    with col_exp2:
        # Export du modèle
        model_export = pd.DataFrame({
            'Couche': range(1, n_layers + 1),
            'Resistivite_Ohmm': rho,
            'Epaisseur_m': [*thicknesses, np.nan],
            'Type': ['Couche'] * (n_layers - 1) + ['Demi-espace']
        })
        
        st.download_button(
            label="📥 Télécharger modèle (CSV)",
            data=model_export.to_csv(index=False, sep=';').encode('utf-8'),
            file_name="modele_sous_sol.csv",
            mime="text/csv",
            help="Exporte les paramètres du modèle de sous-sol"
        )
    
    # ==============================================================
    # 12) SECTION PÉDAGOGIQUE FINALE
    # ==============================================================
    
    st.divider()
    
    with st.expander("🎓 Exercices/Question"):
        st.markdown("""
        ### Questions pour les étudiants :
        
        1. **Question à poser pour les étudiant** :
           - Que se passe-t-il si on augmente MN/2 pour Schlumberger ?
           - Pourquoi utilise-t-on MN/2 petit pour Schlumberger ?
        
        2. **Analyse des courbes** :
           - Quelle configuration est la plus sensible aux couches intermédiaires ?
           - À quel AB/2 détecte-t-on la dernière interface ?
        
        4. **Exercice de calcul manuel** :
           - Pour AB/2 = 100 m, calculez manuellement K pour Wenner
          
        
        5. **Interprétation** :
           - Si ρₐ > ρ₁ pour petits AB/2, que peut-on dire du sous-sol ?
           - Comment évolue la profondeur d'investigation avec AB/2 ?
        """)
else:
    st.warning("Ajustez les paramètres pour obtenir une simulation valide.")
