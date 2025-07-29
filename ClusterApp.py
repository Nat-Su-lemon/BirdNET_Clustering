"""
    Clustering Application for birdNET embeddings. Has functionality to export to Raven Selection Tables
    Written by Nathaniel Su (nbs63) 7/29/2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import threading
import time
import sys
import io
import datetime
from HDBSCAN_gui import ClusteringGUI

if __name__ == '__main__':
    root = tk.Tk()
    app = ClusteringGUI(root)
    root.mainloop()