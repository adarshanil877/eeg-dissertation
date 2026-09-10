import mne
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Load data
file_path = "data/c01_cleaned.set"
raw = mne.io.read_raw_eeglab(file_path, preload=True)

# 2. Pick target frontal channels
target_channels = ['Fz', 'FCz', 'F2', 'F3', 'F4', 'F6']
available_channels = [ch for ch in target_channels if ch in raw.ch_names]
raw.pick(available_channels)

# 3. Find onset times for Condition 1 and Condition 2
cond1_time = None
cond2_time = None

for annot in raw.annotations:
    desc = str(annot['description']).strip().lower()
    if desc in ['condition 1', '1', 'condition1'] and cond1_time is None:
        cond1_time = annot['onset']
    if desc in ['condition 2', '2', 'condition2'] and cond2_time is None:
        cond2_time = annot['onset']
    if cond1_time is not None and cond2_time is not None:
        break

if cond1_time is None or cond2_time is None:
    raise ValueError("Could not locate both Condition 1 and Condition 2 in dataset annotations.")

# 4. Create Side-by-Side Figure for Slides
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

def plot_condition_window(ax, target_time, condition_label):
    # Short context window: 2s before epoch start (-2.9s total) to 1.5s after condition onset
    p_start = max(0, target_time - 2.9)
    p_end = target_time + 1.5
    
    data, times = raw[:, :]
    mask = (times >= p_start) & (times <= p_end)
    sub_times = times[mask]
    sub_data = data[:, mask]
    
    spacing = 50e-6
    for i, ch_name in enumerate(available_channels):
        ax.plot(sub_times, sub_data[i] + (i * spacing), color='black', lw=1)
        
    ax.set_yticks([i * spacing for i in range(len(available_channels))])
    ax.set_yticklabels(available_channels, fontsize=10, fontweight='bold')
    
    y_limits = ax.get_ylim()
    box_height = y_limits[1] - y_limits[0]
    
    # Plot conditions floating at top
    for annot in raw.annotations:
        annot_time = annot['onset']
        annot_desc = str(annot['description'])
        if p_start <= annot_time <= p_end and annot_desc != 'boundary':
            is_target = annot_desc.lower() in ['condition 1', 'condition 2', '1', '2']
            line_color = 'green' if is_target else 'blue'
            
            ax.axvline(x=annot_time, color=line_color, linestyle='--', alpha=0.7, lw=1.5)
            ax.text(
                annot_time, 
                y_limits[1] + (spacing * 0.15), 
                f" {annot_desc}", 
                color=line_color, 
                fontweight='bold', 
                fontsize=9, 
                ha='left', 
                va='bottom'
            )
            
    # Draw Highlight Box (-0.9s to 0s)
    epoch_start = target_time - 0.9
    box = patches.Rectangle(
        (epoch_start, y_limits[0]),
        width=0.9,
        height=box_height,
        linewidth=1.8,
        edgecolor='red',
        facecolor='red',
        alpha=0.18,
        zorder=3
    )
    ax.add_patch(box)
    
    # Badge Label inside Box
    ax.text(
        (epoch_start + target_time) / 2, 
        y_limits[1] - (spacing * 0.3), 
        "Epoch\n(-0.9s to 0s)", 
        color='darkred', 
        fontweight='bold', 
        fontsize=8.5, 
        ha='center',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", lw=1, alpha=0.9)
    )
    
    ax.set_ylim(y_limits[0], y_limits[1] + (spacing * 0.7))
    ax.set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')
    ax.set_title(condition_label, fontsize=12, fontweight='bold', pad=20)
    ax.grid(True, linestyle=':', alpha=0.5)

# Render Panels
plot_condition_window(ax1, cond1_time, "Condition 1 Target Window")
plot_condition_window(ax2, cond2_time, "Condition 2 Target Window")

plt.tight_layout()
plt.savefig("eeg_cond1_cond2_side_by_side.png", dpi=300)
plt.show()