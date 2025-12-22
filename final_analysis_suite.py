import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import glob
import json
import re
import warnings

# Machine Learning Imports
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, confusion_matrix, 
                             accuracy_score, classification_report, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

# --- CONFIGURATION ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set professional scientific style
sns.set_theme(style="white", context="talk") 
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

class UltimateIsoAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.data_dir = os.path.join(results_dir, "data")
        self.output_dir = os.path.join(results_dir, "ml_analysis_suite")
        self.matrix_dir = os.path.join(self.output_dir, "confusion_matrices")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.matrix_dir, exist_ok=True)
        
        # Load Data
        self.df = self.load_data()
        
        # Prepare Features and Targets
        self.X, self.y, self.feature_names = self.prepare_features()
        
        # Define Models
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
            "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
            "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=5000, random_state=42)
        }

    def load_data(self):
        """Reconstructs database from individual run files."""
        print("Loading data...")
        records = []
        energy_files = glob.glob(os.path.join(self.data_dir, "*_energies.npy"))
        
        if not energy_files:
            raise FileNotFoundError(f"No *_energies.npy files found in {self.data_dir}")

        for e_file in energy_files:
            base_name = os.path.basename(e_file).replace("_energies.npy", "")
            energies = np.load(e_file)
            
            hist_path = os.path.join(self.data_dir, f"{base_name}_histories.pkl")
            histories = []
            if os.path.exists(hist_path):
                with open(hist_path, 'rb') as f:
                    histories = pickle.load(f)

            match = re.search(r'_(\d+)nodes', base_name)
            n_nodes = int(match.group(1)) if match else 0
            ground_truth = "Non-Isomorphic" if "vs" in base_name else "Isomorphic"
            
            for i, energy in enumerate(energies):
                hist = histories[i] if i < len(histories) else []
                convergence_steps = len(hist)
                start_energy = hist[0] if len(hist) > 0 else energy
                
                records.append({
                    'pair_name': base_name,
                    'n_nodes': n_nodes,
                    'energy': energy,
                    'start_energy': start_energy,
                    'convergence_steps': convergence_steps,
                    'energy_drop': start_energy - energy,
                    'run_id': i,
                    'Ground_Truth': ground_truth,
                    'is_iso': 1 if ground_truth == "Isomorphic" else 0
                })
                
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} samples.")
        return df

    def prepare_features(self):
        """Standardizes features for ML."""
        self.df['log_energy'] = np.log1p(self.df['energy']) 
        
        feature_cols = ['n_nodes', 'energy', 'log_energy', 'convergence_steps', 'energy_drop']
        X = self.df[feature_cols].values
        y = self.df['is_iso'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols

    # ==========================================
    # VISUALIZATION METHODS
    # ==========================================

    def plot_energy_distributions(self):
        """Energy Histograms split by class."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x="energy", hue="Ground_Truth", 
                     kde=True, bins=40, log_scale=(False, True),
                     palette={"Isomorphic": "forestgreen", "Non-Isomorphic": "firebrick"})
        plt.title("Energy Distribution (Log Scale Y)")
        plt.savefig(os.path.join(self.output_dir, "viz_energy_dist.png"))
        plt.close()

    def plot_manifolds(self):
        """t-SNE and PCA visualization."""
        print("Generating Manifold Visualizations...")
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=self.df['Ground_Truth'], 
                        palette={"Isomorphic": "green", "Non-Isomorphic": "red"}, alpha=0.7)
        plt.title(f"PCA Projection (Explained Var: {np.sum(pca.explained_variance_ratio_):.2%})")
        plt.savefig(os.path.join(self.output_dir, "viz_manifold_pca.png"))
        plt.close()

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=min(30, len(self.df)-1), random_state=42)
        X_tsne = tsne.fit_transform(self.X)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=self.df['Ground_Truth'], 
                        palette={"Isomorphic": "green", "Non-Isomorphic": "red"}, alpha=0.7)
        plt.title("t-SNE Manifold Projection")
        plt.savefig(os.path.join(self.output_dir, "viz_manifold_tsne.png"))
        plt.close()

    def plot_confusion_matrix_grid(self, matrices):
        """Plots all confusion matrices in a single grid."""
        n_models = len(matrices)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        
        for i, (name, cm) in enumerate(matrices.items()):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Iso", "Iso"])
            disp.plot(ax=axes[i], cmap='Blues', values_format='d', colorbar=False)
            axes[i].set_title(name, fontsize=14, fontweight='bold')
            axes[i].grid(False)
            
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.suptitle("Confusion Matrices (Cross-Validation)", fontsize=20, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ml_confusion_matrices_grid.png"), bbox_inches='tight')
        plt.close()

    def save_individual_confusion_matrix(self, name, cm):
        """Saves a single confusion matrix."""
        plt.figure(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Iso", "Iso"])
        disp.plot(cmap='Blues', values_format='d', colorbar=True)
        plt.title(f"Confusion Matrix: {name}")
        plt.grid(False)
        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(self.matrix_dir, f"cm_{safe_name}.png"))
        plt.close()

    def plot_feature_importance(self, model, name):
        """Plots feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(8, 5))
            sns.barplot(x=model.feature_importances_, y=self.feature_names, 
                        hue=self.feature_names, legend=False, palette="viridis")
            plt.title(f"Feature Importance: {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"viz_importance_{name.replace(' ', '_')}.png"))
            plt.close()

    def plot_model_comparison_bar(self, df):
        """Bar chart of model accuracies with layout fix."""
        plt.figure(figsize=(12, 6))
        
        # Plot
        sns.barplot(data=df, x="Accuracy", y="Model", hue="Model", legend=False, palette="magma")
        
        # Formatting
        plt.xlim(0, 1.0)
        plt.title("Model Accuracy Comparison (5-Fold CV)")
        plt.xlabel("Accuracy Score")
        plt.ylabel("")  # Hide Y-label "Model" as it's redundant
        
        # Fix for truncated labels
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ml_model_accuracy.png"), bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self, roc_data):
        plt.figure(figsize=(10, 8))
        for name, (fpr, tpr, roc_auc) in roc_data.items():
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, "ml_roc_curves.png"))
        plt.close()

    def plot_pr_curves(self, pr_data):
        plt.figure(figsize=(10, 8))
        for name, (precision, recall) in pr_data.items():
            plt.plot(recall, precision, lw=2, label=f'{name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "ml_pr_curves.png"))
        plt.close()

    # ==========================================
    # MODEL EXECUTION
    # ==========================================

    def run_model_comparison(self):
        print("\nRunning ML Model Comparison (5-Fold CV)...")
        results = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        roc_data = {}
        pr_data = {}
        confusion_matrices = {}
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            try:
                # 1. Cross Validation Prediction (for Matrix and Curves)
                if hasattr(model, "predict_proba"):
                    y_probas = cross_val_predict(model, self.X, self.y, cv=cv, method="predict_proba")[:, 1]
                    y_pred = (y_probas > 0.5).astype(int)
                else:
                    # For models without predict_proba
                    y_pred = cross_val_predict(model, self.X, self.y, cv=cv)
                    y_probas = cross_val_predict(model, self.X, self.y, cv=cv, method="decision_function")

                # 2. Metrics
                acc = accuracy_score(self.y, y_pred)
                fpr, tpr, _ = roc_curve(self.y, y_probas)
                roc_auc = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(self.y, y_probas)
                cm = confusion_matrix(self.y, y_pred)
                
                # 3. Store Data
                roc_data[name] = (fpr, tpr, roc_auc)
                pr_data[name] = (precision, recall)
                confusion_matrices[name] = cm
                
                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "AUC": roc_auc
                })
                
                # 4. Feature Importance (Fit on full data for visualization)
                model.fit(self.X, self.y)
                self.plot_feature_importance(model, name)
                
            except Exception as e:
                print(f"    Error in CV for {name}: {e}")
                
        # --- Save All Plots ---
        
        # 1. Performance Table
        res_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        res_df.to_csv(os.path.join(self.output_dir, "model_performance.csv"), index=False)
        
        # 2. ROC Curves
        self.plot_roc_curves(roc_data)
        
        # 3. PR Curves
        self.plot_pr_curves(pr_data)
        
        # 4. Bar Chart
        self.plot_model_comparison_bar(res_df)
        
        # 5. Confusion Matrices (Grid and Individual)
        print("  Generating Confusion Matrices...")
        self.plot_confusion_matrix_grid(confusion_matrices)
        for name, cm in confusion_matrices.items():
            self.save_individual_confusion_matrix(name, cm)
            
        return res_df

    def generate_full_report(self, res_df):
        report_path = os.path.join(self.output_dir, "final_analysis_report.txt")
        with open(report_path, "w") as f:
            f.write("QUANTUM GRAPH ISOMORPHISM: ULTIMATE ANALYSIS REPORT\n")
            f.write("===================================================\n\n")
            f.write("1. DATA SUMMARY\n")
            f.write(f"   Total Runs: {len(self.df)}\n")
            f.write(f"   Isomorphic Samples: {sum(self.y)}\n")
            f.write(f"   Non-Isomorphic Samples: {len(self.y) - sum(self.y)}\n\n")
            f.write("2. MODEL PERFORMANCE RANKING\n")
            f.write(res_df.to_string(index=False))
            f.write("\n\n")
            if not res_df.empty:
                best = res_df.iloc[0]
                f.write(f"   Best Performing Model: {best['Model']} ({best['Accuracy']:.2%})\n")
        print(f"Report saved to {report_path}")

    def run(self):
        print("Starting Ultimate Visualization Suite...")
        self.plot_energy_distributions()
        self.plot_manifolds()
        res_df = self.run_model_comparison()
        self.generate_full_report(res_df)
        print("Analysis pipeline complete.")

if __name__ == "__main__":
    base = "graph_isomorphism_results"
    if os.path.exists(base):
        runs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        runs.sort(key=os.path.getmtime)
        if runs:
            latest_run = runs[-1]
            print(f"Targeting: {latest_run}")
            analyzer = UltimateIsoAnalyzer(latest_run)
            analyzer.run()
        else:
            print("No run data found.")
    else:
        print("Results directory not found.")