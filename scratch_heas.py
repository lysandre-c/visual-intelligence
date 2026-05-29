import json
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, '/home/vanschal/visual-intelligence')
import matplotlib
matplotlib.use('Agg')

from src.metrics.heas import heas_table
from src.analysis.plots import plot_heas_table, save_figure

def main():
    results_path = Path('/home/vanschal/visual-intelligence/results/full/all_results.json')
    with open(results_path, 'r') as f:
        all_results = json.load(f)
        
    human_baselines = {
        'geometric': 0.99,
        'color': 0.99,
        'angle': 0.99,
        'motion': 0.99,
        'impossible': 0.99,
    }
    
    # We use control_ceiling_threshold=-1 as per the yaml config
    df = heas_table(all_results, human_baselines, control_ceiling_threshold=-1)
    
    print(df.to_string())
    
    # Plot and save
    fig = plot_heas_table(df, title="HEAS Table (0.99 baselines)")
    out_dir = Path("HEAS_NEW_VIZ")
    out_dir.mkdir(exist_ok=True)
    save_figure(fig, out_dir / "heas_table_0.99.png")
    print(f"Plot saved to {out_dir / 'heas_table_0.99.png'}")

if __name__ == '__main__':
    main()
