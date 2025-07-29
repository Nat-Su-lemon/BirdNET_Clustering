""" 
    GUI Wrapper for HDBSCAN Clustering Algorithm 
    Written by Nathaniel Su (nbs63)

"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import threading
import time
import sys
import io
import datetime
import webbrowser
from HDBSCAN_cluster import run_pipeline

# ------------------ GUI ------------------
class RedirectText(io.StringIO):
    def __init__(self, text_widget, log_file=None):
        super().__init__()
        self.text_widget = text_widget
        self.log_file = log_file

    def write(self, string):
        self.text_widget.after(0, self.text_widget.insert, tk.END, string)
        self.text_widget.after(0, self.text_widget.see, tk.END)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(string)

    def flush(self):
        pass

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

class ClusteringGUI:
    def __init__(self, root):
        self.root = root
        root.title("Clustering Config GUI 1.0")
        root.geometry("700x850")

        # Defaults: use Python None where appropriate
        self.default_values = {
            "embedding_path": "",
            "base_folder_name": "",
            "save_path": "",
            "input_path": "",
            "sort_sounds": False,
            "selection_name": "",
            "recluster_noise": False,
            "make_graph": True,
            "make_3d": True,
            "min_clst_size": 5,
            "min_samples": None,              #Can be None
            "cluster_selection_epsilon": 0.0,
            "n_components": 20,
            "n_neighbors": 15,
            "min_dist": None,                 # Can be None
            "cluster_selection_method": "eom",
        }

        self.vars = self.init_variables()
        self.build_gui()

    def init_variables(self):
        # For fields that can be None, use StringVar, store "None" as string
        def none_to_str(val):
            return "None" if val is None else str(val)

        return {
            "embedding_path": tk.StringVar(value=self.default_values["embedding_path"]),
            "base_folder_name": tk.StringVar(value=self.default_values["base_folder_name"]),
            "save_path": tk.StringVar(value=self.default_values["save_path"]),
            "input_path": tk.StringVar(value=self.default_values["input_path"]),
            "sort_sounds": tk.BooleanVar(value=self.default_values["sort_sounds"]),
            "selection_name": tk.StringVar(value=self.default_values["selection_name"]),
            "recluster_noise": tk.BooleanVar(value=self.default_values["recluster_noise"]),
            "make_graph": tk.BooleanVar(value=self.default_values["make_graph"]),
            "make_3d": tk.BooleanVar(value=self.default_values["make_3d"]),

            "min_clst_size": tk.StringVar(value=none_to_str(self.default_values["min_clst_size"])),
            "min_samples": tk.StringVar(value=none_to_str(self.default_values["min_samples"])),
            "cluster_selection_epsilon": tk.StringVar(value=none_to_str(self.default_values["cluster_selection_epsilon"])),
            "n_components": tk.StringVar(value=none_to_str(self.default_values["n_components"])),
            "n_neighbors": tk.StringVar(value=none_to_str(self.default_values["n_neighbors"])),
            "min_dist": tk.StringVar(value=none_to_str(self.default_values["min_dist"])),

            "cluster_selection_method": tk.StringVar(value=self.default_values["cluster_selection_method"]),
        }

    def build_gui(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        

        descriptions = {
            "embedding_path": "Folder containing the embeddings (from BirdNET).",
            "base_folder_name": "Name for the directory to store sorted sound files.",
            "save_path": "Directory to save the output files.",
            "input_path": "Folder containing original sound files.",
            "sort_sounds": "Whether to sort files into label folders.",
            "selection_name": "File name for Raven selection table (optional).",
            "recluster_noise": "Whether to re-cluster the noise group.",
            "make_graph": "Generate and show cluster visualization.",
            "make_3d": "Use 3D instead of 2D visualization.",
            "min_clst_size": "Minimum number of samples per cluster.",
            "min_samples": "Min samples for HDBSCAN core point (use 'None' to default).",
            "cluster_selection_epsilon": "Merge clusters closer than this distance.",
            "n_components": "UMAP dimensionality reduction output size.",
            "n_neighbors": "UMAP local connectivity parameter.",
            "min_dist": "UMAP minimum distance between embedded points.",
            "cluster_selection_method": "Cluster selection strategy (eom = robust, leaf = fine-grained)."
        }

        row = 0
        for key, var in self.vars.items():
            label = key.replace('_', ' ').capitalize()
            desc = descriptions.get(key, "")
            lbl_widget = ttk.Label(frame, text=label)
            lbl_widget.grid(row=row, column=0, sticky='w')

            if key == "cluster_selection_method":
                widget = ttk.Combobox(frame, textvariable=var, values=["eom", "leaf"], state="readonly")
                widget.grid(row=row, column=1, sticky='ew')
            elif isinstance(var, (tk.StringVar, tk.IntVar, tk.DoubleVar)):
                widget = ttk.Entry(frame, textvariable=var, width=40)
                widget.grid(row=row, column=1, sticky='ew')
                if key in ["embedding_path", "save_path", "input_path"]: # if key in ["embedding_path", "base_folder_name", "save_path", "input_path"]:
                    ttk.Button(frame, text="Browse", command=lambda v=var: self.browse_folder(v)).grid(row=row, column=2)
                elif key == "selection_name":
                    ttk.Button(frame, text="Browse", command=lambda v=var: self.browse_file(v)).grid(row=row, column=2)
            elif isinstance(var, tk.BooleanVar):
                widget = ttk.Checkbutton(frame, variable=var)
                widget.grid(row=row, column=1, sticky='w')

            if desc:
                ToolTip(lbl_widget, desc)
                ToolTip(widget, desc)

            row += 1

        ttk.Button(frame, text="Run Clustering", command=self.start_pipeline_thread).grid(row=row, column=0, pady=10)
        ttk.Button(frame, text="Save Settings", command=self.save_settings).grid(row=row, column=1)
        row += 1
        ttk.Button(frame, text="Load Settings", command=self.load_settings).grid(row=row, column=1)
        ttk.Button(frame, text="Reset to Defaults", command=self.reset_defaults).grid(row=row, column=0, pady=10)

        row += 1
        ttk.Label(frame, text="Progress:").grid(row=row, column=0, sticky='w')
        self.progress = ttk.Progressbar(frame, length=200, mode='determinate', maximum=100)
        self.progress.grid(row=row, column=1, columnspan=2, sticky='ew')

        row += 1
        self.eta_label = ttk.Label(frame, text="ETA: N/A")
        self.eta_label.grid(row=row, column=0, columnspan=3, sticky='w')

        row += 1
        ttk.Label(frame, text="Output:").grid(row=row, column=0, columnspan=3, sticky='w')
        self.output_text = tk.Text(frame, height=15)
        self.output_text.grid(row=row + 1, column=0, columnspan=3, sticky='nsew')

        # Footer text and links
        row += 2
        article = ttk.Label(frame, text="View Reference Article", foreground="blue", cursor="hand2", font=("Arial", 8, "underline"))
        article.grid(row=row, column=0, sticky='w', pady=(10, 0))
        article.bind("<Button-1>", lambda f: self.open_link("https://www.sciencedirect.com/science/article/pii/S1574954125002791?utm#s0080"))

        link = ttk.Label(frame, text="View Code on GitHub", foreground="blue", cursor="hand2", anchor="e", font=("Arial", 8, "underline"))
        link.grid(row=row, column=1, sticky='e', pady=(10, 0))
        link.bind("<Button-1>", lambda e: self.open_link("https://github.com/Nat-Su-lemon/BirdNET_Clustering"))

        # row += 1
        
        # footer_text = ttk.Label(frame, text="HDBSCAN Clustering by Nathaniel Su (nbs63)", anchor="center", justify="center", font=("Arial", 8))
        # footer_text.grid(row=row, column=0, sticky='w', pady=(10, 0))

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)  # if multiple columns

        for r in range(row):
            frame.rowconfigure(r, weight=1)

    def open_link(self, url):
        webbrowser.open_new(url)

    def reset_defaults(self):
        def none_to_str(val):
            return "None" if val is None else str(val)
        for key, default_value in self.default_values.items():
            if key in self.vars:
                self.vars[key].set(none_to_str(default_value))
        messagebox.showinfo("Reset", "Settings have been reset to default values.")

    def browse_folder(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def browse_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            var.set(path)

    def gather_params(self):
        params = {}
        for key, var in self.vars.items():
            value = var.get()
            if key == "min_samples" and value == "None":
                params[key] = None
            else:
                try:
                    params[key] = float(value)
                    if params[key].is_integer():
                        params[key] = int(params[key])
                except:
                    params[key] = value
        return params

    def save_settings(self):
        file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not file:
            return
        params = self.gather_params()
        with open(file, "w") as f:
            json.dump(params, f, indent=4)
        messagebox.showinfo("Saved", f"Settings saved to {file}")

    def load_settings(self):
        file = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not file:
            return
        with open(file, "r") as f:
            params = json.load(f)
        for key, value in params.items():
            if key in self.vars:
                self.vars[key].set(str(value) if value is not None else "None")
        messagebox.showinfo("Loaded", f"Settings loaded from {file}")

    def start_pipeline_thread(self):
        self.progress["value"] = 0
        self.eta_label.config(text="ETA: N/A")
        thread = threading.Thread(target=self.run_pipeline_thread)
        thread.start()

    def run_pipeline_thread(self):
        params = self.gather_params()
        log_path = params.get("save_path", "") or "."
        self.log_file = log_path.rstrip("/\\") + "/cluster_output.log"
        sys.stdout = RedirectText(self.output_text, log_file=self.log_file)

        def update_progress(val):
            self.root.after(0, lambda: self.progress.configure(value=val))

        def update_eta(eta_text):
            self.root.after(0, lambda: self.eta_label.config(text=eta_text))

        run_pipeline(params, progress_callback=update_progress, eta_callback=update_eta, log_file=self.log_file)
        self.root.after(0, lambda: self.progress.configure(value=100))
        self.root.after(0, lambda: self.eta_label.config(text="Done"))

if __name__ == '__main__':
    root = tk.Tk()
    app = ClusteringGUI(root)
    root.mainloop()