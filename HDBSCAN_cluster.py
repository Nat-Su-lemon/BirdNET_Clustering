""" 
    HDBSCAN Clustering Algorithm 
    Written by Nathaniel Su (nbs63)
"""

import os
import numpy as np 
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
import pandas as pd
import shutil
from tqdm.auto import tqdm
import mplcursors
from alive_progress import alive_bar, animations, config_handler
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import threading
import time
import sys
import io
import datetime

def run_pipeline(params, progress_callback=None, eta_callback=None, log_file=None):
    # Example usage of params
    embedding_path = params['embedding_path']
    base_folder_name = params['base_folder_name']
    save_path = params['save_path']
    input_path = params['sounds_path']
    sort_sounds = params['sort_sounds']
    selection_name = params['selection_name']
    recluster_noise = params['recluster_noise']
    make_graph = params['make_graph']
    make_3d = params['make_3d']
    min_clst_size = params['min_clst_size']
    min_samples = params['min_samples']
    cluster_selection_epsilon = params['cluster_selection_epsilon']
    n_components = params['n_components']
    n_neighbors = params['n_neighbors']
    min_dist = params['min_dist']
    cluster_selection_method = params['cluster_selection_method']

    if not input_path:
        sort_sounds = False

    print("Running clustering with:")
    for k, v in params.items():
        print(f"{k}: {v}")
    print("\n\n")


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

    config_handler.set_global(length=30, spinner='wait3', bar=bird_bar, force_tty=True, file=sys.__stdout__)

    selection_table = None
    # Makes sure selection_name has the right file ending
    # selection_name = selection_name + '.txt' if not selection_name.endswith('.txt') else selection_name

    print("Loading Audio Embeddings")
    total = os.listdir(embedding_path)
    total_time = len(total) if not sort_sounds or not input_path else len(os.listdir(input_path)) + len(total)
    start_time = time.time()
    with alive_bar(len(total), title = "Loading Audio Embeddings") as bar:
        i = 0
        for file_name in total:
            step_start = time.time()
            if file_name.endswith('.txt') and file_name != selection_name.rsplit("/", 1)[1]:
                file_path = os.path.join(embedding_path, file_name)
                # Load the array from the text file
                try:
                    array = np.loadtxt(file_path, delimiter=',')
                    arrays.append(array)
                    # print(f"Loading {file_name}")
                    # Remove known suffix from filename
                    base_name = file_name.replace('.birdnet.embeddings.txt', '')
                    base_name = base_name.split('_', 2)
                    files.append(base_name[0])
                    elapsed = time.time() - start_time
                    remaining = (elapsed / (i + 1)) * (total_time - (i + 1))
                    if eta_callback:
                        eta_callback(f"Loading Embeddings: {int(remaining)}s")
                    if progress_callback:
                        progress_callback((i + 1) / total_time * 100)
                    bar()
                except Exception as e:
                    bar.text = f"Error loading {file_name}: {e}"
            else:
                elapsed = time.time() - start_time
                remaining = (elapsed / (i + 1)) * (total_time - (i + 1))
                if eta_callback:
                    eta_callback(f"Loading Embeddings: {int(remaining)}s")
                if progress_callback:
                    progress_callback((i + 1) / total_time * 100)
                bar()
            i += 1


    #Combine list of array as numpy array
    stacked_array = np.stack(arrays)
    eta_callback(f"Waiting for UMAP and HDBSCAN")
    print(f"Loaded {len(arrays)} embeddings.")
    start_time = time.time()
    # Run clustering algorithim (HDBSCAN) and reduce dimensionality if specified
    print(f"{'UMAP reducing to '+ str(n_components) if n_components != None else 'Using birdNET base 1024'} dimension")
    umap_reducer = umap.UMAP(
        n_components=n_components,        
        n_neighbors=n_neighbors,         # tradeoff between local vs global structure
        min_dist=min_dist,           # tighter packing = better cluster separation
        metric='euclidean'      # compatible with HDBSCAN
    )
    X_reduced = stacked_array if n_components == None else umap_reducer.fit_transform(stacked_array)
    elapsed = time.time() - start_time
    print(f"Finished UMAP in {round(elapsed, 2)} Seconds")

    # Run Clustering algo
    start_time = time.time()
    print(f"Starting HDBSCAN")
    hdb = HDBSCAN(min_cluster_size = min_clst_size, cluster_selection_method=cluster_selection_method, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)
    labels = hdb.fit_predict(X_reduced)
    elapsed = time.time() - start_time
    print(f"Finished HDBSCAN in {round(elapsed, 2)} Seconds")
    print(f"Found {len(labels)} labels")
    # print(labels)

    # Loading selection table
    if selection_name != None:
        # If there is a selection table to be referenced, load it onto a DataFrame (Has to be a raven selection table with selection numbers or else it may break!)
        print("Loading selection table..")
        # selection_path = os.path.join(embedding_path, selection_name)
        selection_table = pd.read_csv(selection_name, delimiter='\t')

    # Save labels to corresponding files
    df = pd.DataFrame({
        'filename': files,
        'cluster': labels
    })
    df = df.drop_duplicates()

    # Create a dictionary mapping file name to correct label
    label_dict = dict(zip(df.filename, df.cluster))

    # Changing output folder name if it already exists
    # os.makedirs(save_path, exist_ok=True)
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
            match = re.match(r"sel\.(\d+)", selection)
            if match:
                selection_num = int(match.group(1))
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
        start_time = time.time()
        with alive_bar(len(total_sounds), title = "Sorting sound files") as bar:
            # i = 0
            for sound_name in total_sounds:
                step_start = time.time()
                if sound_name.endswith('.wav'):
                    sound_path = os.path.join(input_path, sound_name)
                    try: 
                        sound_label = label_dict[sound_name.replace('.wav', '')]
                        for folder in os.listdir(output_path):
                            full_path = os.path.join(output_path, folder)
                            if os.path.isdir(full_path) and str(sound_label) == folder:
                                #copy2 should preserve more metadata as opposed to normal shutil copy
                                shutil.copy2(sound_path, full_path) 
                                elapsed = time.time() - start_time
                                remaining = (elapsed / (i + 1)) * (total_time - (i + 1))
                                if eta_callback:
                                    eta_callback(f"Sorting Sounds: {int(remaining)}s")
                                if progress_callback:
                                    progress_callback((i + 1) / total_time * 100)
                                bar()
                                sound_count += 1
                                break
                    except KeyError:
                        print(f"Warning: {sound_name} was not part of embeddings list...skipping") 
                        skipped_files.append(sound_name)
                        elapsed = time.time() - start_time
                        remaining = (elapsed / (i + 1)) * (total_time - (i + 1))
                        if eta_callback:
                            eta_callback(f"Sorting Sounds: {int(remaining)}s")
                        if progress_callback:
                            progress_callback((i + 1) / total_time * 100)
                        bar()
                        continue
                    # Search for the right label folder to copy into
                    
                else:
                    print(f"Skipped {sound_name} (not a valid .wav file)")
                    skipped_files.append(sound_name)
                    elapsed = time.time() - start_time
                    remaining = (elapsed / (i + 1)) * (total_time - (i + 1))
                    if eta_callback:
                        eta_callback(f"Sorting Sounds: {int(remaining)}s")
                    if progress_callback:
                        progress_callback((i + 1) / total_time * 100)
                    bar()
                i += 1
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

