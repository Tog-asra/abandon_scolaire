import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
from ml_models import StudentAnalysisML
from data_generator import generate_student_data
from fpdf import FPDF
import tempfile
import os


# Configuration de la page
st.set_page_config(
    page_title="Prévention Abandon Scolaire",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3A7CA5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Charge les données avec mise en cache"""
    if not os.path.exists('student_data.csv'):
        df = generate_student_data(1000)
        df.to_csv('student_data.csv', index=False)
    return pd.read_csv('student_data.csv')

@st.cache_resource
def load_ml_models():
    """Charge les modèles ML avec mise en cache"""
    ml_analyzer = StudentAnalysisML()
    
    if os.path.exists('models/models.pkl'):
        try:
            # Charger les modèles existants
            ml_analyzer.models = joblib.load('models/models.pkl')
            ml_analyzer.scaler = joblib.load('models/scaler.pkl')
            ml_analyzer.label_encoders = joblib.load('models/label_encoders.pkl')
            ml_analyzer.feature_selector = joblib.load('models/feature_selector.pkl')
            ml_analyzer.selected_features = joblib.load('models/selected_features.pkl')
            
            # Charger et préprocesser les données
            ml_analyzer.load_and_preprocess_data('student_data.csv')
            
            print("✅ Modèles chargés avec succès")
            
        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles: {e}")
            # Fallback: entraîner les modèles
            ml_analyzer.load_and_preprocess_data('student_data.csv')
            ml_analyzer.train_classification_models()
            perform_clustering_safe(ml_analyzer)
            ml_analyzer.save_models()
    else:
        # Entraîner les modèles
        try:
            ml_analyzer.load_and_preprocess_data('student_data.csv')
            ml_analyzer.train_classification_models()
            perform_clustering_safe(ml_analyzer)
            ml_analyzer.save_models()
            print("✅ Modèles entraînés et sauvegardés")
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement des modèles: {e}")
    
    return ml_analyzer

def perform_clustering_safe(ml_analyzer):
    """Effectue le clustering de manière sécurisée"""
    try:
        if ml_analyzer.X_scaled is not None and hasattr(ml_analyzer, 'df') and ml_analyzer.df is not None:
            ml_analyzer.perform_clustering()
        else:
            st.warning("Clustering non disponible - données non préparées correctement")
    except Exception as e:
        st.warning(f"Clustering non effectué: {e}")

def create_pdf_report(student_data, prediction, recommendations):
    """Crée un rapport PDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    
    # Titre
    pdf.cell(0, 10, 'Rapport de Risque d\'Abandon Scolaire', 0, 1, 'C')
    pdf.ln(10)
    
    # Informations de l'étudiant
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Informations de l\'etudiant:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for key, value in student_data.items():
        pdf.cell(0, 8, f'{key}: {value}', 0, 1)
    
    pdf.ln(5)
    
    # Prédiction
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Evaluation du risque:', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 8, f'Probabilite d\'abandon: {prediction["risk_probability"]:.1%}', 0, 1)
    pdf.cell(0, 8, f'Niveau de risque: {prediction["risk_level"]}', 0, 1)
    
    pdf.ln(5)
    
    # Recommandations
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Recommandations:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for rec in recommendations:
        pdf.cell(0, 8, f'- {rec}', 0, 1)
    
    return pdf

def generate_recommendations(student_data, prediction):
    """Génère des recommandations personnalisées"""
    recommendations = []
    
    if prediction['risk_probability'] > 0.7:
        recommendations.append("Suivi personnalisé urgent recommandé")
        recommendations.append("Rencontre avec le conseiller pédagogique")
    
    if student_data['note_moyenne'] < 10:
        recommendations.append("Mise en place d'un tutorat académique")
        recommendations.append("Révision des méthodes d'apprentissage")
    
    if student_data['taux_absenteisme'] > 25:
        recommendations.append("Suivi de l'assiduité renforcé")
        recommendations.append("Identification des causes d'absentéisme")
    
    if student_data['satisfaction_etudiant'] < 6:
        recommendations.append("Enquête de satisfaction approfondie")
        recommendations.append("Amélioration de l'expérience étudiante")
    
    if student_data['temps_moodle_heures'] < 30:
        recommendations.append("Encourager l'utilisation des ressources numériques")
        recommendations.append("Formation aux outils pédagogiques en ligne")
    
    if student_data['participation_forums'] < 3:
        recommendations.append("Encourager la participation aux forums")
        recommendations.append("Activités collaboratives en ligne")
    
    if not recommendations:
        recommendations.append("Continuer le suivi régulier")
        recommendations.append("Maintenir l'engagement actuel")
    
    return recommendations

# Interface principale
def main():
    st.markdown('<h1 class="main-header">🎓 Prévention de l\'Abandon Scolaire</h1>', unsafe_allow_html=True)
    
    # Chargement des données et modèles
    try:
        df = load_data()
        ml_analyzer = load_ml_models()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données/modèles: {e}")
        st.stop()
    
    # Sidebar pour navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page:",
        ["Dashboard Exploratoire", "Analyse des Clusters", "Simulation Individuelle", "Règles d'Association"]
    )
    
    if page == "Dashboard Exploratoire":
        st.header("📊 Analyse Exploratoire des Données")
        
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nombre d'étudiants", len(df))
        
        with col2:
            abandon_rate = df['abandon'].mean()
            st.metric("Taux d'abandon", f"{abandon_rate:.1%}")
        
        with col3:
            avg_grade = df['note_moyenne'].mean()
            st.metric("Note moyenne", f"{avg_grade:.1f}/20")
        
        with col4:
            avg_satisfaction = df['satisfaction_etudiant'].mean()
            st.metric("Satisfaction moyenne", f"{avg_satisfaction:.1f}/10")
        
        # Visualisations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution des abandons par région")
            fig_region = px.histogram(df, x='region', color='abandon', 
                                    title='Abandons par région',
                                    color_discrete_map={0: 'lightblue', 1: 'red'})
            st.plotly_chart(fig_region, use_container_width=True)
        
        with col2:
            st.subheader("Corrélation entre variables")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(corr_matrix, 
                               title='Matrice de corrélation',
                               color_continuous_scale='RdBu')
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Analyse des notes vs abandon
        st.subheader("Analyse des facteurs de risque")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_notes = px.box(df, x='abandon', y='note_moyenne',
                             title='Distribution des notes par statut d\'abandon')
            st.plotly_chart(fig_notes, use_container_width=True)
        
        with col2:
            fig_absent = px.box(df, x='abandon', y='taux_absenteisme',
                              title='Absentéisme par statut d\'abandon')
            st.plotly_chart(fig_absent, use_container_width=True)
        
        # Importance des features
        if hasattr(ml_analyzer, 'models') and 'random_forest' in ml_analyzer.models:
            st.subheader("Importance des caractéristiques")
            try:
                feature_importance = ml_analyzer.get_feature_importance()
                if feature_importance is not None:
                    fig_importance = px.bar(feature_importance, 
                                          x='importance', 
                                          y='feature',
                                          orientation='h',
                                          title='Importance des variables pour prédire l\'abandon')
                    st.plotly_chart(fig_importance, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible d'afficher l'importance des features: {e}")
    
    elif page == "Analyse des Clusters":
        st.header("🎯 Segmentation des Étudiants")
        
        # Vérifier si le clustering a été effectué
        if 'cluster_kmeans' not in df.columns and hasattr(ml_analyzer, 'df') and ml_analyzer.df is not None:
            with st.spinner("Effectuation du clustering..."):
                perform_clustering_safe(ml_analyzer)
                if hasattr(ml_analyzer, 'df') and 'cluster_kmeans' in ml_analyzer.df.columns:
                    df = ml_analyzer.df
        
        if 'cluster_kmeans' in df.columns:
            # Visualisation des clusters
            st.subheader("Profils d'étudiants identifiés")
            
            # Réduction de dimensionnalité pour visualisation
            try:
                from sklearn.decomposition import PCA
                
                # Préparation des données pour PCA
                numeric_cols = ['age', 'note_moyenne', 'taux_absenteisme', 'taux_remise_devoirs', 
                               'temps_moodle_heures', 'participation_forums', 'satisfaction_etudiant']
                X_pca = df[numeric_cols].fillna(df[numeric_cols].mean())
                
                pca = PCA(n_components=2)
                X_pca_transformed = pca.fit_transform(X_pca)
                
                df_pca = pd.DataFrame({
                    'PC1': X_pca_transformed[:, 0],
                    'PC2': X_pca_transformed[:, 1],
                    'Cluster': df['cluster_kmeans'].astype(str),
                    'Abandon': df['abandon'].astype(str)
                })
                
                fig_clusters = px.scatter(df_pca, x='PC1', y='PC2', 
                                        color='Cluster',
                                        symbol='Abandon',
                                        title='Visualisation des clusters d\'étudiants',
                                        labels={'PC1': 'Première composante principale',
                                               'PC2': 'Deuxième composante principale'})
                st.plotly_chart(fig_clusters, use_container_width=True)
                
                # Analyse des profils par cluster
                st.subheader("Caractéristiques par cluster")
                
                cluster_stats = df.groupby('cluster_kmeans').agg({
                    'abandon': 'mean',
                    'note_moyenne': 'mean',
                    'taux_absenteisme': 'mean',
                    'satisfaction_etudiant': 'mean',
                    'temps_moodle_heures': 'mean'
                }).round(2)
                
                cluster_stats.columns = ['Taux abandon', 'Note moyenne', 'Absentéisme (%)', 
                                       'Satisfaction', 'Temps Moodle (h)']
                
                st.dataframe(cluster_stats, use_container_width=True)
                
                # Description des profils
                st.subheader("Interprétation des profils")
                
                for cluster_id in sorted(df['cluster_kmeans'].unique()):
                    cluster_data = df[df['cluster_kmeans'] == cluster_id]
                    abandon_rate = cluster_data['abandon'].mean()
                    
                    if abandon_rate > 0.5:
                        risk_level = "🔴 Très haut risque"
                        color = "risk-high"
                    elif abandon_rate > 0.3:
                        risk_level = "🟡 Risque modéré"
                        color = "risk-medium"
                    else:
                        risk_level = "🟢 Faible risque"
                        color = "risk-low"
                    
                    st.markdown(f"""
                    <div class="metric-card {color}">
                        <h4>Cluster {cluster_id} - {risk_level}</h4>
                        <p><strong>Taille:</strong> {len(cluster_data)} étudiants ({len(cluster_data)/len(df)*100:.1f}%)</p>
                        <p><strong>Taux d'abandon:</strong> {abandon_rate:.1%}</p>
                        <p><strong>Note moyenne:</strong> {cluster_data['note_moyenne'].mean():.1f}/20</p>
                        <p><strong>Absentéisme:</strong> {cluster_data['taux_absenteisme'].mean():.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Erreur lors de l'analyse des clusters: {e}")
        else:
            st.warning("Clustering non disponible. Veuillez relancer l'application ou réentraîner les modèles.")
    
    elif page == "Simulation Individuelle":
        st.header("👤 Évaluation Individuelle du Risque")
        
        st.markdown("Entrez les informations de l'étudiant pour évaluer son risque d'abandon:")
        
        # Interface de saisie
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informations personnelles")
            age = st.slider("Âge", 17, 30, 20)
            sexe = st.selectbox("Sexe", ['M', 'F'])
            region = st.selectbox("Région", ['Urbaine', 'Rurale', 'Semi-urbaine'])
            niveau_parents = st.selectbox("Niveau d'éducation des parents", 
                                        ['Primaire', 'Secondaire', 'Université', 'Post-grad'])
            situation_financiere = st.selectbox("Situation financière", 
                                              ['Précaire', 'Moyenne', 'Bonne'])
        
        with col2:
            st.subheader("Performances académiques")
            note_moyenne = st.slider("Note moyenne (/20)", 0.0, 20.0, 12.0, 0.1)
            taux_absenteisme = st.slider("Taux d'absentéisme (%)", 0, 100, 15)
            taux_remise_devoirs = st.slider("Taux de remise des devoirs (%)", 0, 100, 85)
            temps_moodle = st.slider("Temps sur Moodle (heures/mois)", 0, 200, 50)
            participation_forums = st.slider("Participation aux forums", 0, 20, 5)
            satisfaction = st.slider("Satisfaction (/10)", 1, 10, 7)
        
        # Bouton de prédiction
        if st.button("Évaluer le risque", type="primary"):
            student_data = {
                'age': age,
                'sexe': sexe,
                'region': region,
                'niveau_education_parents': niveau_parents,
                'situation_financiere': situation_financiere,
                'note_moyenne': note_moyenne,
                'taux_absenteisme': taux_absenteisme,
                'taux_remise_devoirs': taux_remise_devoirs,
                'temps_moodle_heures': temps_moodle,
                'participation_forums': participation_forums,
                'satisfaction_etudiant': satisfaction
            }
            
            # Prédiction
            try:
                prediction = ml_analyzer.predict_dropout_risk(student_data)
                
                # Affichage du résultat
                st.subheader("Résultat de l'évaluation")
                
                risk_prob = prediction['risk_probability']
                risk_level = prediction['risk_level']
                
                if risk_level == 'Élevé':
                    color = "risk-high"
                    emoji = "🔴"
                elif risk_level == 'Moyen':
                    color = "risk-medium"
                    emoji = "🟡"
                else:
                    color = "risk-low"
                    emoji = "🟢"
                
                st.markdown(f"""
                <div class="metric-card {color}">
                    <h3>{emoji} Niveau de risque: {risk_level}</h3>
                    <h4>Probabilité d'abandon: {risk_prob:.1%}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Détail des prédictions des modèles
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Random Forest", f"{prediction['rf_probability']:.1%}")
                with col2:
                    st.metric("XGBoost", f"{prediction['xgb_probability']:.1%}")
                
                # Génération des recommandations
                recommendations = generate_recommendations(student_data, prediction)
                
                st.subheader("Recommandations personnalisées")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                # Crée une variable de session pour stocker le PDF temporairement
                if "pdf_ready" not in st.session_state:
                    st.session_state.pdf_ready = False
                    st.session_state.pdf_path = ""

                # Génération du PDF
                if st.button("Générer rapport PDF"):
                    try:
                        pdf = create_pdf_report(student_data, prediction, recommendations)

                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            pdf.output(tmp_file.name)
                            st.session_state.pdf_path = tmp_file.name
                            st.session_state.pdf_ready = True
                            st.success("✅ Rapport généré avec succès ! Cliquez ci-dessous pour le télécharger.")
                    except Exception as e:
                        st.error(f"Erreur lors de la génération du PDF : {e}")
                    st.session_state.pdf_ready = False

                # Bouton de téléchargement s’affiche après génération
                if st.session_state.pdf_ready:
                    with open(st.session_state.pdf_path, "rb") as f:
                        st.download_button(
                        label="📄 Télécharger le rapport PDF",
                        data=f.read(),
                        file_name="rapport_risque_abandon.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {str(e)}")
                st.info("Veuillez vérifier que les modèles sont correctement entraînés.")
    
    elif page == "Règles d'Association":
        st.header("🔗 Règles d'Association")
        
        st.markdown("Découverte de schémas comportementaux menant à l'abandon scolaire")
        
        # Génération des règles d'association
        with st.spinner("Génération des règles d'association..."):
            try:
                rules = ml_analyzer.generate_association_rules(min_support=0.01, min_confidence=0.4)
                
                if not rules.empty:
                    st.subheader("Règles découvertes")
                    
                    # Filtrage interactif
                    min_confidence = st.slider("Confiance minimale", 0.0, 1.0, 0.6, 0.05)
                    min_lift = st.slider("Lift minimal", 1.0, 10.0, 1.2, 0.1)
                    
                    filtered_rules = rules[(rules['confidence'] >= min_confidence) & 
                                         (rules['lift'] >= min_lift)]
                    
                    if not filtered_rules.empty:
                        # Affichage des règles
                        st.write(f"**{len(filtered_rules)} règles trouvées**")
                        
                        for idx, rule in filtered_rules.head(10).iterrows():
                            antecedents = ', '.join(list(rule['antecedents']))
                            consequents = ', '.join(list(rule['consequents']))
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Si {antecedents}</h4>
                                <h4>Alors {consequents}</h4>
                                <p><strong>Confiance:</strong> {rule['confidence']:.2%} | 
                                   <strong>Support:</strong> {rule['support']:.2%} | 
                                   <strong>Lift:</strong> {rule['lift']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Visualisation des règles
                        if len(filtered_rules) > 0:
                            st.subheader("Visualisation des règles")
                            
                            fig_rules = px.scatter(filtered_rules, 
                                                 x='support', 
                                                 y='confidence',
                                                 size='lift',
                                                 hover_data=['lift'],
                                                 title='Support vs Confiance des règles')
                            st.plotly_chart(fig_rules, use_container_width=True)
                    else:
                        st.warning("Aucune règle ne correspond aux critères sélectionnés.")
                else:
                    st.warning("Aucune règle d'association trouvée avec les paramètres actuels.")
                    
            except Exception as e:
                st.error(f"Erreur lors de la génération des règles: {str(e)}")
                st.info("Essayez de réduire les seuils de support et de confiance.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🎓 **Système de Prévention de l'Abandon Scolaire** | "
        "Développé avec Streamlit et Machine Learning"
    )

if __name__ == "__main__":
    main()