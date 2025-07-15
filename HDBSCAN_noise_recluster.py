# Implementation of BirdNET embedding clustering algorithm using HDBSCAN from Sci-kit learn
# Written by Nathaniel Su (nbs63) 7/9/2025

import os
import numpy as np 
from sklearn.cluster import HDBSCAN, Birch, OPTICS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
import pandas as pd
import shutil
from tqdm.auto import tqdm
import mplcursors
from alive_progress import alive_bar, animations, config_handler

# =============================================== PARAMETER CONFIG =============================================== #


# These directories start from base directory from where you run this script ie. where you "cd" to in terminal
embedding_path = '70_days_embeddings/70_days_sounds'
# Desired base filename and directory for csv file
# base_name = 'cluster_output.csv'

# Directory to store sorted sound files by label (parameters are prepended)
base_folder_name = '70_days'
save_path = '70_days_clustering_output'

# Directory where sound files are stored (sound file names must match that of the embeddings 
# excluding the automatic suffix BirdNET appends during embeddings extraction)
input_path = '70_days_sounds'

# Set to True to sort sound files into their own label folers
# input_path must contain a directory of wav files that corresponds to embeddings 
sort_sounds = False

# Selection table file name to generate a raven selection table with corresponding label annotations
# selection numbers must match that of the embeddings
# set to None to skip this 
selection_name = 'combined_target_selections'

recluster_noise = True

# Whether cluster visualization will be generated and shown
make_graph = True
# True for a 3d visualization, False for 2d
make_3d = True

# The minimum number of samples in a group for that group to be considered a cluster; groupings 
# smaller than this size will be left as noise. (int, default=5 )
min_clst_size = 30

# The parameter k used to calculate the distance between a point x_p and its k-th nearest neighbor.
# When None, defaults to min_cluster_size. (int, default=None)
min_samples = 10

# A distance threshold. Clusters below this value will be merged. (float, default=0.0)
cluster_selection_epsilon = 0.1

# Size to reduce the dimensionality of data (Default=None, uses original 320 dimension BirdNET embeddings)
n_components = 20

n_neighbors = 15

min_dist = 0.1

# The method used to select clusters from the condensed tree. The standard approach for HDBSCAN* 
# is to use an Excess of Mass ("eom") algorithm to find the most persistent clusters. Alternatively 
# you can instead select the clusters at the leaves of the tree ("leaf")-- this provides the most fine grained and 
# homogeneous clusters.
cluster_selection_method="eom"


# ============================================= PARAMETER CONFIG END ============================================= #


# Creating output folder name from parameters
base_folder = f"{n_components}_{cluster_selection_epsilon}epil_{min_samples}min_{cluster_selection_method}_{base_folder_name}"

arrays = []
files = []

# Define a custom bar style
bird_bar = animations.bar_factory(
    #fill='',      
    tip='🕊️',
    borders=('🐣', '🌳')       
  
)

config_handler.set_global(length=30, spinner='wait3', bar=bird_bar)

selection_table = None
# Makes sure selection_name has the right file ending
selection_name = selection_name.rsplit('.', 1)[0] + '.txt' if not selection_name.endswith('.txt') else selection_name

total = os.listdir(embedding_path)
with alive_bar(len(total), title = "Loading Audio Embeddings") as bar:
    for file_name in total:
        if file_name.endswith('.txt') and file_name != selection_name:
            file_path = os.path.join(embedding_path, file_name)
            # Load the array from the text file
            try:
                array = np.loadtxt(file_path, delimiter=',')
                arrays.append(array)
                # print(f"Loading {file_name}")
                # Remove known suffix from filename
                base_name = file_name.replace('.birdnet.embeddings.txt', '')
                base_name = base_name.split('_')
                files.append(base_name[0])
                bar()
            except Exception as e:
                bar.text = f"Error loading {file_name}: {e}"
        else:
            bar()


#Combine list of array as numpy array
stacked_array = np.stack(arrays)

print(f"Loaded {len(arrays)} embeddings.")

# Run clustering algorithim (HDBSCAN) and reduce dimensionality if specified
print(f"{'UMAP reducing to '+ str(n_components) if n_components != None else 'Using birdNET base 1024'} dimension")
umap_reducer = umap.UMAP(
    n_components=n_components,        
    n_neighbors=n_neighbors,         # tradeoff between local vs global structure
    min_dist=min_dist,           # tighter packing = better cluster separation
    metric='euclidean'      # compatible with HDBSCAN
)
X_reduced = stacked_array if n_components == None else umap_reducer.fit_transform(stacked_array)

# Run Clustering algo
print(f"Starting HDBSCAN")
hdb = HDBSCAN(min_cluster_size = min_clst_size, cluster_selection_method=cluster_selection_method, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)
labels = hdb.fit_predict(X_reduced)
# print("Finished HDBSCAN")
print(f"Found {len(labels)} labels")
# print(labels)

# Loading selection table
if selection_name != None:
    # If there is a selection table to be referenced, load it onto a DataFrame (Has to be a raven selection table with selection numbers or else it may break!)
    print("Loading selection table..")
    selection_path = os.path.join(embedding_path, selection_name)
    selection_table = pd.read_csv(selection_path, delimiter='\t')

# Save labels to corresponding files
df = pd.DataFrame({
    'filename': files,
    'cluster': labels
})
df = df.drop_duplicates()

# Create a dictionary mapping file name to correct label
label_dict = dict(zip(df.filename, df.cluster))

# Changing output folder name if it already exists
os.makedirs(save_path, exist_ok=True)
output_path = os.path.join(save_path, base_folder)
if os.path.exists(output_path):
    i = 1
    while True:
        new_name = f"{base_folder}_{i}"
        new_path = os.path.join(save_path, new_name)
        if not os.path.exists(new_path):
            output_path = new_path
            break
        i += 1
os.makedirs(output_path, exist_ok=True)

# Save the DataFrame correspondin files to labels
file_path = os.path.join(output_path, "Cluster_output.csv")
df.to_csv(file_path, index=False)

# Generating and saving DataFrame counts of each label
unique, counts = np.unique(labels, return_counts=True)
num_labels = pd.DataFrame({'Label': unique, 'Count': counts})
file_path = os.path.join(output_path, "Label_counts.csv")
num_labels_sorted = num_labels.sort_values(by='Count', ascending=False)
num_labels_sorted.to_csv(file_path, index=False)

# Save parameters/settings used for the algorithim and save paths
param = pd.DataFrame({
'Minimum Cluster Size' : min_clst_size,
'Minimum Samples' : min_samples,
'Cluster Selection Epsilon' : cluster_selection_epsilon,
'Number of Dimensions (None for base)' : n_components,
"Cluster Selection Method" : cluster_selection_method,
"N Neighbors" : n_neighbors,
"Minimum Distance" : min_dist},
index=[0])
file_path = os.path.join(output_path, "Parameters.csv")
param.to_csv(file_path, index=False)

# Save label annotations to a Raven selection table (if one exists)
if selection_name != None:
    selection_table['Label'] = 'Unknown'
    total_selections = df['filename']
    selection_table['Selection'] = selection_table['Selection'].astype('Int64')
    selection_table['Channel'] = selection_table['Channel'].astype('Int64')
    for selection in total_selections:
        selection_num = int(selection.split('.')[1])
        selection_table.loc[selection_table['Selection'] == selection_num, 'Label'] = label_dict[selection]
    selection_save_name = base_folder + '_annotations.txt'
    selection_save_path = os.path.join(output_path, selection_save_name)
    selection_table.to_csv(selection_save_path, sep='\t', index=False)

# Create a set of the different label numbers to use for creating folders
#set_labels = np.unique(labels)
print(f"Labels have {len(unique)} unique values")
#print(set_labels)

if sort_sounds:
# Creating label folders
    for label in unique:
        folder_path = os.path.join(output_path, str(label))
        os.makedirs(folder_path, exist_ok=True)

    # Organize files based on their labels 
    sound_count = 0
    skipped_files = []

    # Iterate through input sound file directory 
    total_sounds = os.listdir(input_path)
    with alive_bar(len(total_sounds), title = "Sorting sound files") as bar:
        for sound_name in total_sounds:
            if sound_name.endswith('.wav'):
                sound_path = os.path.join(input_path, sound_name)
                try: 
                    sound_label = label_dict[sound_name.replace('.wav', '')]
                    for folder in os.listdir(output_path):
                        full_path = os.path.join(output_path, folder)
                        if os.path.isdir(full_path) and str(sound_label) == folder:
                            #copy2 should preserve more metadata as opposed to normal shutil copy
                            shutil.copy2(sound_path, full_path) 
                            bar()
                            sound_count += 1
                            break
                except KeyError:
                    print(f"Warning: {sound_name} was not part of embeddings list...skipping") 
                    skipped_files.append(sound_name)
                    continue
                # Search for the right label folder to copy into
                
            else:
                print(f"Skipped {sound_name} (not a valid .wav file)")
                skipped_files.append(sound_name)
                bar()
    print(f"Sorted and copied {sound_count} files to correct label folders")
    print(f"Output saved to {output_path}")
    # print(f"Skipped files: {skipped_files}")
    skipped = pd.DataFrame({'Skipped Files' : skipped_files})
    skipped_path = os.path.join(output_path, 'Skipped_files.csv')
    skipped.to_csv(skipped_path, index=False)

# Create UMAP visualization
if make_graph:
    print(f"Generating {'3D' if make_3d else '2D'} cluster visualization...")

    # UMAP: 2D or 3D based on toggle
    dimension = 3 if make_3d else 2
    X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=dimension).fit_transform(X_reduced)

    # Filter noise
    mask = labels != -1
    X_clean = X_umap[mask]
    labels_clean = labels[mask]
    print(f"Filtered {len(labels) - len(labels_clean)} sounds classified as noise")
    X_clean = X_umap
    labels_clean = labels

    # Plot
    fig = plt.figure(figsize=(10, 8))

    if make_3d:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_clean[:, 0], X_clean[:, 1], X_clean[:, 2],
                             c=labels_clean, cmap='Spectral', s=10)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
    else:
        plt.scatter(X_clean[:, 0], X_clean[:, 1], c=labels_clean, cmap='Spectral', s=10)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

    plt.title("HDBSCAN Clusters (Noise filtered)")
    plt.colorbar(scatter if make_3d else None, label="Cluster Label")
    plt.grid(True)
    mplcursors.cursor(hover=True)
    plt.show()

if recluster_noise and selection_name != None:
    print("Starting Noise Reclustering... ")
    noise_mask = labels == -1
    X_noise = X_reduced[noise_mask]
    noise_clusterer = HDBSCAN(
        min_cluster_size=20,
        min_samples=5,
        cluster_selection_epsilon=0.2,
        cluster_selection_method='eom'
    )
    noise_labels = noise_clusterer.fit_predict(X_noise)
    #noise_files = files[noise_mask]
    noise_files = df.loc[df['cluster'] == -1, 'filename'].tolist()
    df = pd.DataFrame({
    'filename': noise_files,
    'cluster': noise_labels
    })
    df = df.drop_duplicates()

    # Create a dictionary mapping file name to correct label
    noise_dict = dict(zip(df.filename, df.cluster))

    noise_path = os.path.join(output_path, "Noise_reclustering")
    os.makedirs(noise_path, exist_ok=True)

    # Save the DataFrame correspondin files to labels
    file_path = os.path.join(noise_path, "Noise_cluster_output.csv")
    df.to_csv(file_path, index=False)

    # Generating and saving DataFrame counts of each label
    unique, counts = np.unique(noise_labels, return_counts=True)
    num_labels = pd.DataFrame({'Label': unique, 'Count': counts})
    file_path = os.path.join(noise_path, "Noise_label_counts.csv")
    num_labels_sorted = num_labels.sort_values(by='Count', ascending=False)
    num_labels_sorted.to_csv(file_path, index=False)

    # Save label annotations to a Raven selection table (if one exists)
    if selection_name != None:
        selection_table['Label'] = 'Unknown'
        total_selections = df['filename']
        selection_table['Selection'] = selection_table['Selection'].astype('Int64')
        selection_table['Channel'] = selection_table['Channel'].astype('Int64')
        for selection in total_selections:
            selection_num = int(selection.split('.')[1])
            selection_table.loc[selection_table['Selection'] == selection_num, 'Label'] = noise_dict[selection]
        selection_save_name = base_folder + '_noise_reclustering.txt'
        selection_save_path = os.path.join(noise_path, selection_save_name)
        selection_table.to_csv(selection_save_path, sep='\t', index=False)

print("Finished clustering")

