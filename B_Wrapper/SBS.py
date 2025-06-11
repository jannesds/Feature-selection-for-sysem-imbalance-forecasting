import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import numpy as np

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    smape_val = np.mean(np.where(denominator == 0, 0, diff / denominator)) * 100
    return smape_val
#smape_scorer = make_scorer(smape, greater_is_better=False)
scoring='neg_mean_absolute_error'
def run_sbs(
    model,
    df,
    target,
    features,
    cv,
    n_features_to_select=5,
    scoring=scoring,
    plot=True
):
    # Create output folders
    fig_dir = "figures_sbs"
    log_dir = "logs_sbs"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    forward = False
    floating = True

    if forward:
        selection_tag = "SFFS" if floating else "SFS"
    else:
        selection_tag = "SBFS" if floating else "SBS"
        
    # Timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"{selection_tag}_log_{timestamp}.txt")

    with open(log_file_path, 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        try:
    
            # Scale features
            scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
            y = df[target]
            
            # SBS
            sbs = SFS(
                estimator=model(len(features)),
                k_features=n_features_to_select,
                forward=forward,
                floating=floating,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                verbose=2,
                pre_dispatch='2*n_jobs'
            )
            sbs.fit(X_scaled, y)
    
            print(f" \n {selection_tag} Summary:")
            print(f"Started with {len(features)} features, features to select: {n_features_to_select}.\n")
    
            subsets = sbs.subsets_
            sorted_keys = sorted(subsets.keys(), reverse=True)
            
            for k in sorted_keys:
                info = subsets[k]
                removed = set(features) - set(info['feature_names'])
                kept = list(info['feature_names'])
                step_num = len(features) - k
                print(f"Step {step_num}: {k} features → CV Score: {info['avg_score']:.4f}")
                if removed:
                    print(f"   🔻 Removed ({len(removed)}): {', '.join(sorted(removed))}")
                if kept:
                    print(f"   ✅ Kept ({len(kept)}): {', '.join(sorted(kept))}")
    
            selected_features = [features[i] for i in sbs.k_feature_idx_]
            metric_dic = sbs.get_metric_dict(confidence_interval=0.95)
            df_metric = pd.DataFrame.from_dict(metric_dic, orient="index")
    
            print("\n✅ Final selected features:")
            for i, feat in enumerate(selected_features, 1):
                print(f"{i}. {feat}")
            print(f"\nBest CV score (Negative MAE): {df_metric.loc[n_features_to_select, 'avg_score']:.4f}")
    
            print(f"✅ Console output saved to: {log_file_path}")
        finally:
            sys.stdout = original_stdout

    # Plotting (outside redirection so images still render in notebooks)
    if plot:
        k_features = sorted(metric_dic.keys())
        scores = [metric_dic[k]['avg_score'] for k in k_features]

        metric_name = (
            scoring._score_func.__name__.upper()
            if hasattr(scoring, "_score_func")
            else str(scoring).replace("neg_", "Negative ").upper() )
        color = "#1FABD5"
        #Plot 1
        # Classic line plot: CV score vs number of features
        plt.figure(figsize=(8, 5))
        plt.plot(k_features, scores, marker='o', color=color)
        plt.title(f"Sequential {'Forward' if forward else 'Backward'}{' Floating' if floating else ''} Selection | Scoring: {metric_name}")
        plt.xlabel('Number of Features')
        plt.ylabel(f'Performance - {metric_name}')
        plt.grid(True)
        plt.tight_layout()
        
        scoring_tag = metric_name.lower().replace(" ", "_")

            
        filename = f"{selection_tag}_{scoring_tag}_MLP_lineplot_{timestamp}.png"
        line_plot_path = os.path.join(fig_dir, filename)
        plt.savefig(line_plot_path, dpi=300)
        print(f"📁 Line plot saved: {line_plot_path}")
        plt.show()


        # Plot 2
        fig, ax = plot_sfs(metric_dic, kind='std_dev', color= color)
        plt.title(f"Sequential {'Forward' if forward else 'Backward'}{' Floating' if floating else ''} Selection (w. Std Dev)| Scoring: {metric_name}")
        plt.xlabel("Number of Features")
        plt.ylabel(f"Performance - {metric_name}")
        plt.grid(True)
        plt.tight_layout()
        
        filename = f"{selection_tag}_{scoring_tag}_confidence_intervals_{timestamp}.png"
        conf_plot_path = os.path.join(fig_dir, filename)
        fig.savefig(conf_plot_path, dpi=300)
        print(f"📁 Plot saved: {conf_plot_path}")
        plt.show()

        # Plot 3
        feature_counts = Counter()
        for subset in sbs.subsets_.values():
            for name in subset['feature_names']:
                feature_counts[name] += 1

        freq_complete = {feature: feature_counts.get(feature, 0) for feature in features}
        freq_df = pd.DataFrame.from_dict(freq_complete, orient='index', columns=['frequency'])
        freq_df.index.name = 'feature'
        freq_df.reset_index(inplace=True)
        freq_df.sort_values('frequency', ascending=True, inplace=True)

        plt.figure(figsize=(10, max(6, len(freq_df) * 0.4)))
        ax = sns.barplot(data=freq_df, x="frequency", y="feature", color=color)
        ax.set_title(f"Feature Frequency during Selection in Sequential {'Forward' if forward else 'Backward'}{' Floating' if floating else ''} Selection")
        ax.set_xlabel("Number of times feature was selected")
        ax.set_ylabel("Feature name")

        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

        plt.tight_layout()
        freq_plot_path = os.path.join(fig_dir, f"{selection_tag}_{scoring_tag}_feature_frequency_{timestamp}.png")
        plt.savefig(freq_plot_path, dpi=300)
        print(f"📁 Plot saved: {freq_plot_path}")
        plt.show()

    return selected_features, df_metric
