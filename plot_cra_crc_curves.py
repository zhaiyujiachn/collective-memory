import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import make_interp_spline

# Configure matplotlib for scientific style without LaTeX
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "DejaVu Serif"],
    "text.usetex": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "figure.figsize": (7, 6)  # 将宽度减半，从14减为7
})

def read_json_file(file_path):
    """Read JSON file and return its content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def plot_combined_curves(cra_data, crc_data, dataset_name, output_path):
    """Plot CRA and CRC data as two subplots in one figure with science style"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 6))  # 将宽度减半，从14减为7
    
    # Process both datasets
    for ax, data, curve_type, color in [(ax1, cra_data, 'CRA', 'red'), (ax2, crc_data, 'CRC', 'green')]:
        # Store all data points for calculating average
        all_points = []
        
        # Plot 100 transparent gray lines
        for curve_data in data:
            # Ensure each curve has 10 data points
            if len(curve_data) == 10:
                x = np.arange(1, 11)  # x-axis from 1 to 10
                y = np.array(curve_data)
                
                # Plot gray transparent line
                ax.plot(x, y, color='gray', alpha=0.2, linewidth=0.5)
                
                # Collect data points for average calculation
                all_points.append(y)
        
        # Calculate average curve
        if all_points:
            avg_curve = np.mean(all_points, axis=0)
            x = np.arange(1, 11)
            
            # Create smoother curve with spline interpolation
            x_smooth = np.linspace(1, 10, 100)
            spl = make_interp_spline(x, avg_curve, k=3)  # cubic spline interpolation
            y_smooth = spl(x_smooth)
            
            # Plot smooth fitted curve without markers
            ax.plot(x_smooth, y_smooth, color=color, linewidth=2)
        
        # Set subplot properties
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel(curve_type, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to: {output_path}")
    plt.close()

def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory
    output_dir = os.path.join(base_dir, 'output_figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # APS directory files
    aps_cra_path = os.path.join(base_dir, 'final_code_data', 'APS', 'cra_list.json')
    aps_crc_path = os.path.join(base_dir, 'final_code_data', 'APS', 'crc_list.json')
    
    # Medline directory files
    medline_cra_path = os.path.join(base_dir, 'final_code_data', 'Medline', 'cra_list.json')
    medline_crc_path = os.path.join(base_dir, 'final_code_data', 'Medline', 'crc_list.json')
    
    # DBLP directory files
    dblp_cra_path = os.path.join(base_dir, 'final_code_data', 'DBLP', 'cra_list.json')
    dblp_crc_path = os.path.join(base_dir, 'final_code_data', 'DBLP', 'crc_list.json')
    
    # Read and plot APS data
    print("Processing APS data...")
    aps_cra_data = read_json_file(aps_cra_path)
    aps_crc_data = read_json_file(aps_crc_path)
    
    if aps_cra_data and aps_crc_data:
        plot_combined_curves(
            aps_cra_data,
            aps_crc_data,
            'APS',
            os.path.join(output_dir, 'aps_combined_curves.png')
        )
    
    # Read and plot Medline data
    print("\nProcessing Medline data...")
    medline_cra_data = read_json_file(medline_cra_path)
    medline_crc_data = read_json_file(medline_crc_path)
    
    if medline_cra_data and medline_crc_data:
        plot_combined_curves(
            medline_cra_data,
            medline_crc_data,
            'Medline',
            os.path.join(output_dir, 'medline_combined_curves.png')
        )
    
    # Read and plot DBLP data
    print("\nProcessing DBLP data...")
    dblp_cra_data = read_json_file(dblp_cra_path)
    dblp_crc_data = read_json_file(dblp_crc_path)
    
    if dblp_cra_data and dblp_crc_data:
        plot_combined_curves(
            dblp_cra_data,
            dblp_crc_data,
            'DBLP',
            os.path.join(output_dir, 'dblp_combined_curves.png')
        )
    
    print("\nAll figures generated! Please check the output_figures directory.")

if __name__ == "__main__":
    main()