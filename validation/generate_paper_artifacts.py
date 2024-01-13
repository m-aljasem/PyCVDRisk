import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def setup_style():
    """Set sophisticated style for paper-ready figures."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def load_data(base_dir: Path, models: list) -> dict:
    """Load comparison data for all models."""
    data = {}
    for model in models:
        path = base_dir / "validation" / model / f"{model}_comparison.csv"
        if path.exists():
            df = pd.read_csv(path)
            # Ensure columns exist (handle naming variations)
            if "python_risk" in df.columns and "r_risk" in df.columns:
                 data[model] = df
            else:
                print(f"Warning: Columns missing in {model}")
        else:
            print(f"Warning: {path} not found")
    return data

def plot_scatter(data: dict, output_path: Path):
    """Generate multi-panel scatter plot grid."""
    models = list(data.keys())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()
    
    for i, model in enumerate(models):
        ax = axes[i]
        df = data[model]
        
        # Plot data
        sns.scatterplot(
            data=df, x="r_risk", y="python_risk", 
            ax=ax, alpha=0.3, s=15, color="#2E86C1", edgecolor='none'
        )
        
        # Diagonal line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        # Calculate stats
        r_val, _ = stats.pearsonr(df["r_risk"], df["python_risk"])
        n = len(df)
        rmse = np.sqrt(((df["python_risk"] - df["r_risk"]) ** 2).mean())
        
        # Annotate
        ax.text(
            0.05, 0.95, 
            f"$R$ = {r_val:.5f}\n$N$ = {n:,}\nRMSE = {rmse:.4f}", 
            transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="lightgray")
        )
        
        processed_model_name = model.replace("score2op", "SCORE2-OP").replace("score2dm", "SCORE2-DM").replace("score2", "SCORE2").replace("who", "WHO").replace("globorisk", "Globorisk")
        ax.set_title(processed_model_name, fontweight="bold")
        ax.set_xlabel("R Package Risk (%)")
        ax.set_ylabel("Python Risk (%)")
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved scatter plots to {output_path}")

def plot_bland_altman(data: dict, output_path: Path):
    """Generate multi-panel Bland-Altman plot grid."""
    models = list(data.keys())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()
    
    for i, model in enumerate(models):
        ax = axes[i]
        df = data[model]
        
        mean = (df["python_risk"] + df["r_risk"]) / 2
        diff = df["python_risk"] - df["r_risk"]
        
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        
        # Plot
        sns.scatterplot(
            x=mean, y=diff, 
            ax=ax, alpha=0.3, s=15, color="#C0392B", edgecolor='none'
        )
        
        # Lines
        ax.axhline(mean_diff, color="k", linestyle="-", linewidth=1.5, label="Mean Bias")
        ax.axhline(loa_upper, color="k", linestyle="--", alpha=0.5)
        ax.axhline(loa_lower, color="k", linestyle="--", alpha=0.5)
        
        # Annotations
        ax.text(ax.get_xlim()[1], mean_diff, f" Mean: {mean_diff:.4f}", va="center")
        ax.text(ax.get_xlim()[1], loa_upper, f" +1.96SD: {loa_upper:.4f}", va="center", fontsize=8)
        ax.text(ax.get_xlim()[1], loa_lower, f" -1.96SD: {loa_lower:.4f}", va="center", fontsize=8)
        
        processed_model_name = model.replace("score2op", "SCORE2-OP").replace("score2dm", "SCORE2-DM").replace("score2", "SCORE2").replace("who", "WHO").replace("globorisk", "Globorisk")
        ax.set_title(processed_model_name, fontweight="bold")
        ax.set_xlabel("Mean of Python and R (%)")
        ax.set_ylabel("Difference (Python - R) (%)")
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved Bland-Altman plots to {output_path}")

def generate_summary_table(data: dict, output_dir: Path):
    """Generate and export summary statistics table."""
    summary_list = []
    
    for model, df in data.items():
        n = len(df)
        diff = df["python_risk"] - df["r_risk"]
        abs_diff = np.abs(diff)
        
        r_val, _ = stats.pearsonr(df["r_risk"], df["python_risk"])
        
        processed_model_name = model.replace("score2op", "SCORE2-OP").replace("score2dm", "SCORE2-DM").replace("score2", "SCORE2").replace("who", "WHO").replace("globorisk", "Globorisk")
        
        summary_list.append({
            "Model": processed_model_name,
            "N": n,
            "Pearson R": r_val,
            "Mean Bias (%)": np.mean(diff),
            "SD Bias (%)": np.std(diff),
            "Mean Abs Diff (%)": np.mean(abs_diff),
            "Max Abs Diff (%)": np.max(abs_diff),
            "% Match (<1.0pt)": (abs_diff < 1.0).mean() * 100
        })
    
    summary_df = pd.DataFrame(summary_list)
    
    # Formatting
    summary_df = summary_df.round(4)
    
    # Export CSV
    summary_df.to_csv(output_dir / "validation_summary.csv", index=False)
    
    # Export Markdown
    summary_df.to_markdown(output_dir / "validation_summary.md", index=False)
    print(f"Saved summary tables to {output_dir}")

def main():
    base_dir = Path(__file__).parent.parent
    figures_dir = base_dir / "validation" / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    models = ["score2", "score2op", "score2dm", "who", "globorisk"]
    
    print("Loading data...")
    data = load_data(base_dir, models)
    
    setup_style()
    
    print("Generating Figure 1: Scatter Plots...")
    plot_scatter(data, figures_dir / "validation_scatter.png")
    
    print("Generating Figure 2: Bland-Altman Plots...")
    plot_bland_altman(data, figures_dir / "validation_bland_altman.png")
    
    print("Generating Tables...")
    generate_summary_table(data, base_dir / "validation")
    
    print("All artifacts generated successfully.")

if __name__ == "__main__":
    main()
