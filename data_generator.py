import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random

def generate_student_data(n_students=1000):
    """
    Génère un dataset simulé d'étudiants avec des facteurs réalistes d'abandon scolaire
    """
    print(f"Génération de {n_students} données d'étudiants...")  # Journal console
    np.random.seed(42)
    random.seed(42)
    
    # Informations socio-démographiques
    ages = np.random.normal(20, 2, n_students).astype(int)
    ages = np.clip(ages, 17, 30)
    
    sexes = np.random.choice(['M', 'F'], n_students, p=[0.45, 0.55])
    
    regions = np.random.choice(['Urbaine', 'Rurale', 'Semi-urbaine'], n_students, p=[0.6, 0.25, 0.15])
    
    niveau_parents = np.random.choice(['Primaire', 'Secondaire', 'Université', 'Post-grad'], 
                                     n_students, p=[0.2, 0.35, 0.35, 0.1])
    
    situation_financiere = np.random.choice(['Précaire', 'Moyenne', 'Bonne'], 
                                          n_students, p=[0.25, 0.55, 0.2])
    
    # Données académiques
    notes_moyennes = np.random.normal(12, 3, n_students)
    notes_moyennes = np.clip(notes_moyennes, 0, 20)
    
    taux_absenteisme = np.random.beta(2, 5, n_students) * 100  # Entre 0 et 100%
    
    remise_devoirs = np.random.normal(85, 15, n_students)
    remise_devoirs = np.clip(remise_devoirs, 0, 100)
    
    # Engagement plateforme
    temps_moodle = np.random.exponential(50, n_students)  # heures par mois
    participation_forums = np.random.poisson(5, n_students)
    
    # Satisfaction
    satisfaction = np.random.normal(7, 2, n_students)
    satisfaction = np.clip(satisfaction, 1, 10)
    
    # Création du DataFrame
    df = pd.DataFrame({
        'age': ages,
        'sexe': sexes,
        'region': regions,
        'niveau_education_parents': niveau_parents,
        'situation_financiere': situation_financiere,
        'note_moyenne': notes_moyennes,
        'taux_absenteisme': taux_absenteisme,
        'taux_remise_devoirs': remise_devoirs,
        'temps_moodle_heures': temps_moodle,
        'participation_forums': participation_forums,
        'satisfaction_etudiant': satisfaction
    })
    
    # Création de la variable cible "abandon" basée sur des règles logiques
    abandon_prob = np.zeros(n_students)
    
    # Facteurs augmentant le risque d'abandon
    abandon_prob += (df['note_moyenne'] < 10) * 0.4
    abandon_prob += (df['taux_absenteisme'] > 30) * 0.3
    abandon_prob += (df['taux_remise_devoirs'] < 60) * 0.25
    abandon_prob += (df['satisfaction_etudiant'] < 5) * 0.2
    abandon_prob += (df['temps_moodle_heures'] < 20) * 0.15
    abandon_prob += (df['situation_financiere'] == 'Précaire') * 0.2
    abandon_prob += (df['participation_forums'] == 0) * 0.1
    
    # Facteurs protecteurs
    abandon_prob -= (df['niveau_education_parents'].isin(['Université', 'Post-grad'])) * 0.15
    abandon_prob -= (df['region'] == 'Urbaine') * 0.05
    abandon_prob -= (df['satisfaction_etudiant'] > 8) * 0.1
    
    # Normaliser les probabilités
    abandon_prob = np.clip(abandon_prob, 0, 0.8)
    
    # Générer la variable binaire abandon
    df['abandon'] = np.random.binomial(1, abandon_prob)
    
    print("Données générées avec succès.")
    return df

def save_dataset():
    """Génère et sauvegarde le dataset"""
    try:
        df = generate_student_data(1000)
        df.to_csv('student_data.csv', index=False)
        print(f"Dataset généré avec {len(df)} étudiants")
        print(f"Taux d'abandon: {df['abandon'].mean():.2%}")
        if os.path.exists('student_data.csv'):
            print(f"✅ Fichier student_data.csv sauvegardé avec succès.")
        else:
            print(f"❌ Erreur : Fichier student_data.csv non créé.")
        return df
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du dataset : {e}")
        raise

if __name__ == "__main__":
    df = save_dataset()
    print("\nAperçu des données:")
    print(df.head())
    print("\nStatistiques descriptives:")
    print(df.describe())