import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

def analyze_attitude_data(file_path, output_dir=None):
    """Analyze attitude data file and generate separate time series plots for roll and pitch angles"""
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(file_path), 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading file: {file_path}")
    data = np.load(file_path)
    
    # Get filename for title
    file_name = os.path.basename(file_path)
    
    # Check data structure
    print(f"Data structure: {data.dtype}")
    print(f"Number of data points: {len(data)}")
    
    # Filter out NaN values (these are episode separators)
    valid_mask = ~np.isnan(data['pitch'])
    valid_data = data[valid_mask]
    
    # Detect episode boundaries (through time gaps)
    time_diffs = np.diff(valid_data['time'])
    episode_boundaries = np.where(time_diffs > 0.1)[0] + 1
    num_episodes = len(episode_boundaries) + 1
    print(f"Detected {num_episodes} episodes")
    
    # Create figure with two subplots, one for pitch and one for roll
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    
    # Plot 1: Pitch angle over time
    ax1.plot(valid_data['time'], valid_data['pitch_deg'], color='blue', linewidth=1.5)
    
    # Mark episode boundaries on pitch plot
    for boundary in episode_boundaries:
        ax1.axvline(x=valid_data['time'][boundary], color='gray', linestyle='--', alpha=0.7)
    
    ax1.set_title('Pitch Angle During Late Training')
    ax1.set_ylabel('Pitch Angle (deg)')
    ax1.grid(True)
    
    # Plot 2: Roll angle over time
    ax2.plot(valid_data['time'], valid_data['roll_deg'], color='red', linewidth=1.5)
    
    # Mark episode boundaries on roll plot
    for boundary in episode_boundaries:
        ax2.axvline(x=valid_data['time'][boundary], color='gray', linestyle='--', alpha=0.7)
    
    ax2.set_title('Roll Angle During Late Training')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Roll Angle (deg)')
    ax2.grid(True)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"separate_attitude_timeseries_{os.path.splitext(file_name)[0]}.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot to: {fig_path}")
    
    # Calculate basic statistics for console output
    pitch_mean = np.mean(valid_data['pitch_deg'])
    pitch_std = np.std(valid_data['pitch_deg'])
    pitch_max_abs = np.max(np.abs(valid_data['pitch_deg']))
    
    roll_mean = np.mean(valid_data['roll_deg'])
    roll_std = np.std(valid_data['roll_deg'])
    roll_max_abs = np.max(np.abs(valid_data['roll_deg']))
    
    # Output basic statistics to console
    print("\nBasic Statistics:")
    print(f"Pitch: Mean={pitch_mean:.2f}°, Std Dev={pitch_std:.2f}°, Max Abs={pitch_max_abs:.2f}°")
    print(f"Roll: Mean={roll_mean:.2f}°, Std Dev={roll_std:.2f}°, Max Abs={roll_max_abs:.2f}°")
    
    return fig_path

def analyze_all_files(directory, pattern="late_training_attitude_*.npy"):
    """Analyze all files in directory matching the pattern"""
    files = glob.glob(os.path.join(directory, pattern))
    
    if not files:
        print(f"No files matching {pattern} found in directory {directory}")
        return
    
    print(f"Found {len(files)} files to analyze")
    
    for file_path in files:
        print(f"\nAnalyzing file: {os.path.basename(file_path)}")
        analyze_attitude_data(file_path)

def main():
    parser = argparse.ArgumentParser(description='Plot separate roll and pitch angles over time from attitude data')
    parser.add_argument('--file', type=str, help='Path to single data file')
    parser.add_argument('--dir', type=str, help='Path to directory containing multiple data files')
    parser.add_argument('--pattern', type=str, default="late_training_attitude_*.npy", help='File matching pattern')
    parser.add_argument('--output', type=str, help='Output directory path')
    
    args = parser.parse_args()
    
    if args.file:
        analyze_attitude_data(args.file, args.output)
    elif args.dir:
        analyze_all_files(args.dir, args.pattern)
    else:
        print("Please specify either --file or --dir parameter")
        parser.print_help()

if __name__ == "__main__":
    main()
