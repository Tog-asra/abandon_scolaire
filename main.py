#!/usr/bin/env python3
"""
Script principal pour le projet de prévention de l'abandon scolaire
Auteur: Assistant IA
Date: 2025
"""

import os
import sys
import subprocess
import pandas as pd
from data_generator import generate_student_data, save_dataset
from ml_models import StudentAnalysisML

def setup_project():
    """Configuration initiale du projet"""
    print("🎓 Configuration du projet de prévention de l'abandon scolaire")
    print("=" * 60)
    
    # Création des dossiers nécessaires
    folders = ['models', 'data', 'reports']
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            # Vérifier les permissions d'écriture
            test_file = os.path.join(folder, 'test_write.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"✅ Dossier '{folder}' créé/vérifié avec permissions d'écriture")
        except PermissionError as e:
            print(f"❌ Erreur : Permissions insuffisantes pour le dossier '{folder}' : {e}")
            raise
        except Exception as e:
            print(f"❌ Erreur lors de la création du dossier '{folder}' : {e}")
            raise
    
    return True

def generate_data():
    """Génération du dataset d'étudiants"""
    print("\n📊 Génération des données d'étudiants...")
    
    if not os.path.exists('student_data.csv'):
        print("Génération d'un nouveau dataset...")
        try:
            df = save_dataset()
            print(f"✅ Dataset généré: {len(df)} étudiants")
            print(f"   - Taux d'abandon: {df['abandon'].mean():.2%}")
            return df
        except Exception as e:
            print(f"❌ Erreur lors de la génération des données : {e}")
            raise
    else:
        print("Dataset existant trouvé")
        try:
            df = pd.read_csv('student_data.csv')
            print(f"✅ Dataset chargé: {len(df)} étudiants")
            return df
        except Exception as e:
            print(f"❌ Erreur lors du chargement du dataset : {e}")
            raise

def train_models():
    """Entraînement des modèles ML"""
    print("\n🤖 Entraînement des modèles de Machine Learning...")
    
    try:
        ml_analyzer = StudentAnalysisML()
        
        # Chargement et préprocessing
        print("   - Chargement des données...")
        ml_analyzer.load_and_preprocess_data('student_data.csv')
        
        # Entraînement des modèles de classification
        print("   - Entraînement des modèles de classification...")
        ml_analyzer.train_classification_models()
        
        # Clustering
        print("   - Analyse des clusters...")
        ml_analyzer.perform_clustering()
        
        # Génération des règles d'association
        print("   - Génération des règles d'association...")
        rules = ml_analyzer.generate_association_rules()
        if not rules.empty:
            print(f"     {len(rules)} règles d'association trouvées")
        else:
            print("     Aucune règle d'association trouvée")
        
        # Sauvegarde des modèles
        print("   - Sauvegarde des modèles...")
        ml_analyzer.save_models()
        
        print("✅ Modèles entraînés et sauvegardés avec succès")
        
        # Test de prédiction
        print("\n🧪 Test de prédiction...")
        test_student = {
            'age': 19,
            'sexe': 'M',
            'region': 'Urbaine',
            'niveau_education_parents': 'Université',
            'situation_financiere': 'Moyenne',
            'note_moyenne': 8.5,
            'taux_absenteisme': 35,
            'taux_remise_devoirs': 70,
            'temps_moodle_heures': 25,
            'participation_forums': 2,
            'satisfaction_etudiant': 6
        }
        
        prediction = ml_analyzer.predict_dropout_risk(test_student)
        print(f"   Risque d'abandon: {prediction['risk_probability']:.1%}")
        print(f"   Niveau de risque: {prediction['risk_level']}")
        
        return ml_analyzer
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement des modèles: {e}")
        return None

def run_streamlit_app():
    """Lance l'application Streamlit"""
    print("\n🚀 Lancement de l'application Streamlit...")
    print("   L'application va s'ouvrir dans votre navigateur à http://localhost:8501...")
    print("   Utilisez Ctrl+C pour arrêter l'application")
    print("=" * 60)
    
    try:
        # Vérifiez si le port 8501 est disponible
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            port_available = s.connect_ex(('localhost', 8501)) != 0
        if not port_available:
            print("⚠️ Le port 8501 est déjà utilisé. Essayez le port 8502...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8502"])
        else:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application arrêtée par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur lors du lancement de Streamlit: {e}")

def main():
    """Fonction principale"""
    print("🎓 PROJET: PRÉVENTION DE L'ABANDON SCOLAIRE")
    print("=" * 60)
    
    # Configuration
    setup_project()
    
    # Génération des données
    df = generate_data()
    
    # Entraînement des modèles
    ml_analyzer = train_models()
    
    if ml_analyzer is None:
        print("❌ Impossible de continuer sans modèles entraînés")
        return
    
    # Affichage des statistiques
    print("\n📈 STATISTIQUES DU PROJET")
    print("=" * 40)
    print(f"Nombre d'étudiants: {len(df)}")
    print(f"Taux d'abandon: {df['abandon'].mean():.2%}")
    print(f"Note moyenne: {df['note_moyenne'].mean():.1f}/20")
    print(f"Satisfaction moyenne: {df['satisfaction_etudiant'].mean():.1f}/10")
    
    # Affichage de l'importance des features
    if hasattr(ml_analyzer, 'selected_features'):
        print(f"\nFeatures sélectionnées: {len(ml_analyzer.selected_features)}")
        feature_importance = ml_analyzer.get_feature_importance()
        print("\nTop 5 des facteurs prédictifs:")
        for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
            print(f"  {i}. {row['feature']}: {row['importance']:.3f}")
    
    # Menu interactif
    print("\n" + "=" * 60)
    print("MENU PRINCIPAL")
    print("1. Lancer l'application Streamlit")
    print("2. Régénérer les données")
    print("3. Réentraîner les modèles")
    print("4. Quitter")
    
    while True:
        try:
            choice = input("\nVotre choix (1-4): ").strip()
            
            if choice == '1':
                run_streamlit_app()
                break
            elif choice == '2':
                if os.path.exists('student_data.csv'):
                    os.remove('student_data.csv')
                generate_data()
                print("✅ Données régénérées")
            elif choice == '3':
                train_models()
                print("✅ Modèles réentraînés")
            elif choice == '4':
                print("👋 Au revoir!")
                break
            else:
                print("❌ Choix invalide. Veuillez entrer 1, 2, 3 ou 4.")
                
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()