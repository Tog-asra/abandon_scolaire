🎓 Lutte contre l’Abandon Universitaire grâce au Data Mining
Ce projet met en œuvre des approches de fouilles de données (classification, regroupement, règles d’association) afin de repérer les éléments prédictifs de l’abandon scolaire au sein d’une université, tout en offrant une interface interactive permettant d’estimer les risques d’abandon pour chaque étudiant.

🚀 Fonctions Principales
1. Tableau de Bord d’Analyse Exploratoire
Graphiques dynamiques des données étudiantes

Statistiques clés : taux d’abandon, moyennes générales, satisfaction

Analyse de corrélation et distributions

Importance des variables dans les prédictions

2. Segmentation par Clustering
Catégorisation des étudiants selon leur comportement

Identification des profils à haut risque

Représentation visuelle des groupes

Détails statistiques par catégorie

3. Évaluation Personnalisée
Interface pour tester un étudiant en particulier

Prédiction immédiate de probabilité d’abandon

Conseils individualisés

Exportation en format PDF

4. Extraction de Règles d’Association
Détection de comportements récurrents

Identification des combinaisons critiques de facteurs

Visualisation intuitive des règles trouvées

📁 Organisation du Projet
bash
Copier
Modifier
abandon-scolaire/
├── app.py                 # Interface Streamlit
├── main.py               # Point d'entrée principal
├── data_generator.py     # Génération de données simulées
├── ml_models.py          # Implémentation des algorithmes ML
├── requirements.txt      # Liste des dépendances
├── README.md             # Guide du projet
├── models/               # Modèles sauvegardés
├── data/                 # Fichiers de données
└── reports/              # Rapports générés automatiquement
🛠️ Mise en Place
Prérequis
Python ≥ 3.8

Gestionnaire de paquets pip ou conda

Étapes d’installation
Créer le dossier du projet

bash
Copier
Modifier
mkdir abandon-scolaire
cd abandon-scolaire
Ajouter les fichiers suivants dans le dossier :

requirements.txt

data_generator.py

ml_models.py

app.py

main.py

Installer les librairies requises

bash
Copier
Modifier
pip install -r requirements.txt
Démarrer l’application

bash
Copier
Modifier
python app.py
🎯 Comment Utiliser
Option 1 : Lancer le script principal
bash
Copier
Modifier
python app.py
Le programme vous guidera à travers :

Création des données

Entraînement des algorithmes

Lancement de l’interface

Option 2 : Lancer directement Streamlit
bash
Copier
Modifier
streamlit run app.py
Option 3 : Exécuter les modules à part
Génération de données
python
Copier
Modifier
from data_generator import generate_student_data, save_dataset

df = generate_student_data(2000)
save_dataset()
Entraînement des modèles
python
Copier
Modifier
from ml_models import StudentAnalysisML

ml_analyzer = StudentAnalysisML()
ml_analyzer.load_and_preprocess_data('student_data.csv')
ml_analyzer.train_classification_models()
ml_analyzer.perform_clustering()
Simulation d’un étudiant
python
Copier
Modifier
student_data = {
    'age': 20,
    'sexe': 'M',
    'region': 'Urbaine',
    'niveau_education_parents': 'Université',
    'situation_financiere': 'Moyenne',
    'note_moyenne': 12.5,
    'taux_absenteisme': 15,
    'taux_remise_devoirs': 85,
    'temps_moodle_heures': 45,
    'participation_forums': 5,
    'satisfaction_etudiant': 7
}

prediction = ml_analyzer.predict_dropout_risk(student_data)
print(f"Risque d'abandon : {prediction['risk_probability']:.1%}")
🔬 Méthodes de Data Mining Appliquées
1. Classification
Random Forest : Algorithme robuste basé sur des arbres

XGBoost : Méthode de boosting performante

Évaluations : Accuracy, Précision, Rappel, F1-score

2. Clustering
K-Means : Partage en k groupes similaires

DBSCAN : Détection de groupes denses et anomalies

Réduction de dimension : PCA

3. Association
Algorithme Apriori : Extraction de règles fréquentes

Mesures : Support, Confiance, Lift

Utilité : Détection de profils typiques

4. Sélection de Caractéristiques
SelectKBest : Test univarié

Feature Importance : Pondération via Random Forest

📊 Données Simulées
Contenu du jeu de données :

Informations Socio-démographiques
Âge (17 à 30 ans)

Sexe (M/F)

Région d’origine

Niveau éducatif des parents

Situation économique

Informations Académiques
Moyenne générale (/20)

Taux d’absence

Taux de devoirs rendus

Engagement et Motivation
Temps passé sur Moodle

Activité sur les forums

Satisfaction vis-à-vis des cours

Variable à prédire
Abandon : 1 (abandon) / 0 (poursuite)

📈 Évaluation des Résultats
Pour les modèles prédictifs
Accuracy : Taux de réussite global

Precision : Fiabilité des alertes

Recall : Détection des abandons réels

F1-Score : Équilibre entre rappel et précision

Pour les clusters
Score de silhouette : Cohérence interne

Inertie : Compacité des regroupements

Pour les règles d’association
Support : Fréquence d’occurrence

Confiance : Fiabilité des règles

Lift : Gain d’information

🎨 Interface Utilisateur Streamlit
L’application web se compose de 4 sections :

Exploration des Données

Indicateurs globaux

Graphiques dynamiques

Analyse des corrélations

Clustering Étudiant

Visualisation des profils

Résumé des caractéristiques

Estimation du risque

Simulation d’Étudiant

Formulaire d’entrée

Prédiction instantanée

Recommandations adaptées

Export PDF

Analyse des Règles

Affichage des règles

Filtres interactifs

Graphiques associés

🔧 Options de Personnalisation
Modifier les paramètres des modèles
python
Copier
Modifier
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
Ajuster les seuils de risque
python
Copier
Modifier
if prediction['risk_probability'] > 0.8:
    recommendations.append("Intervention urgente")
Personnaliser les visualisations
python
Copier
Modifier
fig = px.scatter(df, color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
🚨 Résolution de Problèmes
Erreurs courantes
Imports manquants :

bash
Copier
Modifier
pip install --upgrade -r requirements.txt
Modèles introuvables :

bash
Copier
Modifier
python main.py
Fichier de données vide :

python
Copier
Modifier
from data_generator import save_dataset
save_dataset()
Performance
Optimisé pour ~2000 étudiants

Pour de plus grands volumes, ajustez les paramètres

Utilisez la mise en cache de Streamlit

📚 Améliorations Futures
Utilisation de Données Réelles

Connexion avec des bases de données universitaires

Adaptation aux contextes locaux

Modèles Plus Puissants

Intégration de réseaux neuronaux

Analyse temporelle

Méthodes hybrides

Interface Améliorée

Connexion utilisateur

Base de données persistante

API RESTful

Suivi Continu

Surveillance des performances

Alertes et notifications

Tableaux de bord en temps réel

👥 Participer au Projet
C’est un projet à but éducatif. Pour contribuer :

Forkez le dépôt

Créez une branche dédiée

Effectuez vos modifications

Ouvrez une pull request