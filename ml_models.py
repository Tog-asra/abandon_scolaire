import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import joblib
import os

class StudentAnalysisML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.feature_selector = None
        self.selected_features = None
        
    def load_and_preprocess_data(self, file_path='student_data.csv'):
        """Charge et préprocesse les données"""
        print(f"Chargement des données depuis {file_path}...")  # Journal console
        try:
            self.df = pd.read_csv(file_path)
        except FileNotFoundError as e:
            print(f"❌ Erreur : Fichier {file_path} introuvable : {e}")
            raise
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données : {e}")
            raise
        
        # Séparation des features catégorielles et numériques
        categorical_cols = ['sexe', 'region', 'niveau_education_parents', 'situation_financiere']
        numerical_cols = ['age', 'note_moyenne', 'taux_absenteisme', 'taux_remise_devoirs', 
                         'temps_moodle_heures', 'participation_forums', 'satisfaction_etudiant']
        
        # Encodage des variables catégorielles
        df_encoded = self.df.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            try:
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
            except Exception as e:
                print(f"❌ Erreur lors de l'encodage de la colonne {col} : {e}")
                raise
        
        # Préparation des données pour ML
        self.X = df_encoded.drop('abandon', axis=1)
        self.y = df_encoded['abandon']
        
        # Normalisation
        try:
            self.X_scaled = pd.DataFrame(
                self.scaler.fit_transform(self.X),
                columns=self.X.columns
            )
            print("Données prétraitées avec succès.")  # Journal console
        except Exception as e:
            print(f"❌ Erreur lors de la normalisation des données : {e}")
            raise
        
        return self.X, self.y
    
    def feature_selection(self, k=8):
        """Sélection des k meilleures features"""
        print("Sélection des features...")  # Journal console
        try:
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(self.X, self.y)
            
            # Récupération des noms des features sélectionnées
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = self.X.columns[selected_indices].tolist()
            
            print(f"Features sélectionnées: {self.selected_features}")
            return X_selected
        except Exception as e:
            print(f"❌ Erreur lors de la sélection des features : {e}")
            raise
    
    def train_classification_models(self):
        """Entraîne les modèles de classification"""
        print("Entraînement des modèles de classification...")  # Journal console
        try:
            # Sélection des features
            X_selected = self.feature_selection()
            
            # Division train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            
            # Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            
            # XGBoost
            xgb_model = xgb.XGBClassifier(random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            
            # Stockage des modèles
            self.models['random_forest'] = rf_model
            self.models['xgboost'] = xgb_model
            
            # Évaluation
            print("=== RANDOM FOREST ===")
            print(f"Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
            print(classification_report(y_test, rf_pred))
            
            print("\n=== XGBOOST ===")
            print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.3f}")
            print(classification_report(y_test, xgb_pred))
            
            return self.models
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement des modèles : {e}")
            raise
    
    def perform_clustering(self, n_clusters=4):
        """Effectue le clustering des étudiants"""
        print("Exécution du clustering...")  # Journal console
        try:
            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters_kmeans = kmeans.fit_predict(self.X_scaled)
            
            # DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters_dbscan = dbscan.fit_predict(self.X_scaled)
            
            # Ajout des clusters au DataFrame
            self.df['cluster_kmeans'] = clusters_kmeans
            self.df['cluster_dbscan'] = clusters_dbscan
            
            self.models['kmeans'] = kmeans
            self.models['dbscan'] = dbscan
            
            print("Clustering terminé.")
            return clusters_kmeans, clusters_dbscan
        except Exception as e:
            print(f"❌ Erreur lors du clustering : {e}")
            raise
    
    def generate_association_rules(self, min_support=0.1, min_confidence=0.6):
        """Génère des règles d'association"""
        print("Génération des règles d'association...")  # Journal console
        try:
            # Discrétisation des variables continues
            df_discrete = self.df.copy()
            
            # Discrétisation basée sur les quartiles
            continuous_cols = ['note_moyenne', 'taux_absenteisme', 'taux_remise_devoirs', 
                              'temps_moodle_heures', 'satisfaction_etudiant']
            
            for col in continuous_cols:
                df_discrete[f'{col}_cat'] = pd.cut(df_discrete[col], 
                                                 bins=3, 
                                                 labels=['Faible', 'Moyen', 'Élevé'])
            
            # Création des transactions pour l'analyse des paniers
            transactions = []
            for _, row in df_discrete.iterrows():
                transaction = []
                
                # Ajout des caractéristiques catégorielles
                transaction.append(f"Sexe_{row['sexe']}")
                transaction.append(f"Région_{row['region']}")
                transaction.append(f"Éducation_parents_{row['niveau_education_parents']}")
                transaction.append(f"Situation_fin_{row['situation_financiere']}")
                
                # Ajout des caractéristiques discrétisées
                for col in continuous_cols:
                    transaction.append(f"{col}_{row[f'{col}_cat']}")
                
                # Ajout du statut d'abandon
                if row['abandon'] == 1:
                    transaction.append("ABANDON")
                else:
                    transaction.append("PERSISTE")
                    
                transactions.append(transaction)
            
            # Encodage des transactions
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_trans = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Génération des itemsets fréquents
            frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)
            
            # Génération des règles d'association
            if len(frequent_itemsets) > 0:
                rules = association_rules(frequent_itemsets, 
                                        metric="confidence", 
                                        min_threshold=min_confidence)
                
                # Filtrer les règles qui mènent à l'abandon
                abandon_rules = rules[rules['consequents'].astype(str).str.contains('ABANDON')]
                
                print(f"{len(abandon_rules)} règles d'association trouvées.")
                return abandon_rules.sort_values('lift', ascending=False)
            
            print("Aucune règle d'association trouvée.")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Erreur lors de la génération des règles d'association : {e}")
            raise
    
    def predict_dropout_risk(self, student_data):
        """Prédit le risque d'abandon pour un étudiant donné"""
        print("Prédiction du risque d'abandon...")  # Journal console
        try:
            if 'random_forest' not in self.models:
                raise ValueError("Les modèles n'ont pas été entraînés")
            
            # Préprocessing des données de l'étudiant
            student_df = pd.DataFrame([student_data])
            
            # Encodage des variables catégorielles
            for col, le in self.label_encoders.items():
                if col in student_df.columns:
                    student_df[col] = le.transform(student_df[col])
            
            # Sélection des features
            if self.selected_features:
                student_features = student_df[self.selected_features]
            else:
                student_features = student_df
            
            # Prédiction avec Random Forest
            rf_proba = self.models['random_forest'].predict_proba(student_features)[0][1]
            
            # Prédiction avec XGBoost
            xgb_proba = self.models['xgboost'].predict_proba(student_features)[0][1]
            
            # Moyenne des prédictions
            avg_proba = (rf_proba + xgb_proba) / 2
            
            print(f"Prédiction terminée : Probabilité moyenne = {avg_proba:.1%}")
            return {
                'risk_probability': avg_proba,
                'risk_level': 'Élevé' if avg_proba > 0.7 else 'Moyen' if avg_proba > 0.4 else 'Faible',
                'rf_probability': rf_proba,
                'xgb_probability': xgb_proba
            }
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction : {e}")
            raise
    
    def get_feature_importance(self):
        """Retourne l'importance des features"""
        print("Calcul de l'importance des features...")  # Journal console
        try:
            if 'random_forest' not in self.models:
                return None
            
            importances = self.models['random_forest'].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Importance des features calculée.")
            return feature_importance
        except Exception as e:
            print(f"❌ Erreur lors du calcul de l'importance des features : {e}")
            raise
    
    def save_models(self, path='models/'):
        """Sauvegarde les modèles entraînés"""
        print(f"Sauvegarde des modèles dans {path}...")  # Journal console
        try:
            os.makedirs(path, exist_ok=True)
            
            joblib.dump(self.models, f'{path}/models.pkl')
            joblib.dump(self.scaler, f'{path}/scaler.pkl')
            joblib.dump(self.label_encoders, f'{path}/label_encoders.pkl')
            joblib.dump(self.feature_selector, f'{path}/feature_selector.pkl')
            joblib.dump(self.selected_features, f'{path}/selected_features.pkl')
            
            print("Modèles sauvegardés avec succès.")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde des modèles : {e}")
            raise

if __name__ == "__main__":
    # Test du pipeline ML
    ml_analyzer = StudentAnalysisML()
    ml_analyzer.load_and_preprocess_data()
    ml_analyzer.train_classification_models()
    ml_analyzer.perform_clustering()
    
    # Test de prédiction
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
    print(f"\nPrédiction pour l'étudiant test: {prediction}")
    
    # Sauvegarde des modèles
    ml_analyzer.save_models()