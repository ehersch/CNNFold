import os
import tkinter as tk
from tkinter import messagebox, ttk
import time
from folding_algorithms import (
    NussinovAlgorithm,
    ZukerAlgorithm,
    OptimizedNussinovAlgorithm,
    OptimizedNussinovWithFourRussians,
)
import nn_folding


class NussinovApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RNA Folding Prediction")
        self.root.state("zoomed")
        self.root.configure(bg="light blue")

        self.title_label = tk.Label(
            root,
            text="RNA Folding Prediction",
            font=("Arial", 20, "bold"),
            bg="light blue",
            fg="black",
        )
        self.title_label.pack(pady=20)

        self.subtitle_label = tk.Label(
            root,
            text="Created by Mark Beckmann, Ethan Hersch, Ben Bigdelle, and Malli Gutta for CS 4775",
            font=("Arial", 12),
            bg="light blue",
            fg="black",
        )
        self.subtitle_label.pack()

        input_frame = tk.Frame(root, bg="light blue")
        input_frame.pack(pady=20)

        self.input_label = tk.Label(
            input_frame,
            text="Insert an RNA sequence:",
            font=("Arial", 14),
            bg="light blue",
            fg="black",
        )
        self.input_label.pack(side=tk.LEFT, padx=10)

        self.sequence_entry = tk.Entry(
            input_frame,
            width=40,
            font=("Arial", 14),
            bg="white",
            fg="black",
            insertbackground="black",
        )
        self.sequence_entry.pack(side=tk.LEFT, padx=10)

        self.generate_button = tk.Button(
            input_frame,
            text="Generate!",
            font=("Arial", 14, "bold"),
            bg="black",
            fg="black",
            activebackground="dark gray",
            activeforeground="white",
            command=self.generate_structure,
        )
        self.generate_button.pack(side=tk.LEFT, padx=10)

        algorithm_frame = tk.Frame(root, bg="light blue")
        algorithm_frame.pack(pady=20)

        tk.Label(
            algorithm_frame,
            text="Choose Algorithm:",
            font=("Arial", 14),
            bg="light blue",
            fg="black",
        ).pack(side=tk.LEFT, padx=10)
        self.algorithm_var = tk.StringVar(value="Nussinov")
        algorithms = [
            "Nussinov",
            "Zuker",
            "Optimized Nussinov",
            "Four Russians",
            "Neural Network",
        ]
        self.algorithm_menu = tk.OptionMenu(
            algorithm_frame, self.algorithm_var, *algorithms
        )
        self.algorithm_menu.config(bg="white", fg="black", font=("Arial", 12))
        self.algorithm_menu.pack(side=tk.LEFT, padx=10)

        main_frame = tk.Frame(root, bg="light blue")
        main_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        left_frame = tk.Frame(main_frame, bg="light blue")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(
            left_frame,
            text="Optimal Structure",
            font=("Arial", 14, "bold"),
            bg="light blue",
            fg="black",
        ).pack(pady=10)
        self.structure_result = tk.Text(
            left_frame,
            font=("Courier", 14),
            bg="white",
            fg="black",
            wrap=tk.WORD,
            width=50,
            height=5,
        )
        self.structure_result.pack(pady=10)

        tk.Label(
            left_frame,
            text="DP Table",
            font=("Arial", 14, "bold"),
            bg="light blue",
            fg="black",
        ).pack(pady=10)
        self.dp_table_tree = ttk.Treeview(left_frame, show="headings", height=15)
        self.dp_table_tree.pack(pady=10, fill=tk.BOTH, expand=True)

        tk.Label(
            left_frame,
            text="Execution Time",
            font=("Arial", 14, "bold"),
            bg="light blue",
            fg="black",
        ).pack(pady=10)
        self.execution_time_result = tk.Label(
            left_frame,
            text="",
            font=("Arial", 12),
            bg="white",
            fg="black",
            width=40,
            height=2,
            anchor="center",
        )
        self.execution_time_result.pack(pady=10)

    def generate_structure(self):
        sequence = self.sequence_entry.get().strip().upper()
        valid_bases = {"G", "U", "A", "C"}
        if not sequence:
            messagebox.showerror("Error", "Please enter an RNA sequence.")
            return
        if any(base not in valid_bases for base in sequence):
            messagebox.showerror("Error", "Invalid RNA sequence. Use only G, U, A, C.")
            return

        try:
            algorithm = self.algorithm_var.get()
            start_time = time.time()
            if algorithm == "Nussinov":
                predictor = NussinovAlgorithm(sequence)
                structure = predictor.run()
                dp_table = predictor.dp_table
            elif algorithm == "Zuker":
                predictor = ZukerAlgorithm(sequence)
                structure = predictor.run()
                dp_table = predictor.dp_table
            elif algorithm == "Optimized Nussinov":
                predictor = OptimizedNussinovAlgorithm(sequence)
                structure = predictor.run()
                dp_table = predictor.dp_table
            elif algorithm == "Four Russians":
                predictor = OptimizedNussinovWithFourRussians(sequence)
                structure = predictor.run()
                dp_table = predictor.dp_table
            elif algorithm == "Neural Network":
                predictor = nn_folding.RNASecondaryStructurePredictor(max_len=390)
                predictor.build_model()
                # model_path = os.path.join("src", "rna_structure_model.keras")
                predictor.model.load_weights("rna_structure_model.keras")
                structure = predictor.predict_structure(sequence)
                dp_table = None
            else:
                raise ValueError("Invalid algorithm selection.")

            execution_time = time.time() - start_time

            self.structure_result.delete(1.0, tk.END)
            self.structure_result.insert(tk.END, structure)

            if dp_table:
                num_columns = len(dp_table[0])
                self.dp_table_tree["columns"] = [str(i) for i in range(num_columns)]

                for col in range(num_columns):
                    self.dp_table_tree.heading(str(col), text=str(col))
                    self.dp_table_tree.column(str(col), width=40, anchor=tk.CENTER)

                for row in self.dp_table_tree.get_children():
                    self.dp_table_tree.delete(row)

                for row_idx, row in enumerate(dp_table):
                    self.dp_table_tree.insert("", "end", values=row)

            self.execution_time_result.config(
                text=f"Execution Time: {execution_time:.2f} seconds"
            )

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = NussinovApp(root)
    root.mainloop()
