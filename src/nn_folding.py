import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import LevenshteinDistance
from folding_algorithms import ZukerAlgorithm, NussinovAlgorithm, OptimizedNussinovAlgorithm, OptimizedNussinovWithFourRussians

class RNASecondaryStructurePredictor:
    def __init__(self, max_len=None, char_to_idx=None, structure_to_idx=None):
        if char_to_idx is None:
            char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        if structure_to_idx is None:
            structure_to_idx = {'.': 0, '(': 1, ')': 2}

        self.max_len = max_len
        self.char_to_idx = char_to_idx
        self.structure_to_idx = structure_to_idx
        self.idx_to_structure = {v: k for k, v in structure_to_idx.items()}
        self.model = None

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path, header=None, names=['sequence', 'structure'])
        sequences = df['sequence'].values
        structures = df['structure'].values
        return sequences, structures

    def fit_transform_data(self, sequences, structures):
        if self.max_len is None:
            self.max_len = max(len(s) for s in sequences)

        X = np.array([self._one_hot_encode(seq, self.max_len) for seq in sequences])
        Y = np.array([self._encode_structure(st, self.max_len) for st in structures])

        return X, Y

    def _one_hot_encode(self, seq, max_len):
        oh = np.zeros((max_len, len(self.char_to_idx)), dtype=np.float32)
        for i, nucleotide in enumerate(seq):
            if nucleotide == 'T':
                nucleotide = 'U'
            oh[i, self.char_to_idx.get(nucleotide, 0)] = 1.0
        return oh

    def _encode_structure(self, struct, max_len):
        out = np.zeros((max_len, len(self.structure_to_idx)), dtype=np.float32)
        for i, ch in enumerate(struct):
            out[i, self.structure_to_idx[ch]] = 1.0
        return out

    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Input(shape=(self.max_len, len(self.char_to_idx))))
        self.model.add(layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        self.model.add(layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Conv1D(filters=128, kernel_size=7, padding='same', activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.TimeDistributed(layers.Dense(len(self.structure_to_idx), activation='softmax')))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def _validate_and_correct_structure(self, structure):
        """
        Ensures the RNA structure is valid:
        1. Balances parentheses.
        2. Removes unmatched parentheses.
        """
        stack = []
        corrected = list(structure)

        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    stack.pop()  # Matched parenthesis
                else:
                    corrected[i] = '.'  # Replace unmatched closing parenthesis with '.'

        # Replace unmatched opening parentheses with '.'
        for i in stack:
            corrected[i] = '.'

        return ''.join(corrected)

    def predict_structure(self, sequence):
        X_new = np.array(self._one_hot_encode(sequence, self.max_len))[None, ...]
        Y_pred = self.model.predict(X_new, verbose=0)
        pred_idx = np.argmax(Y_pred, axis=-1)[0]
        pred_str = ''.join(self.idx_to_structure[i] for i in pred_idx[:len(sequence)])
        return self._validate_and_correct_structure(pred_str)

    def evaluate_levenshtein(self, sequences, structures):
        """
        Compute average Levenshtein distance between predictions and ground truths.
        """
        distances = []
        lev_calc = LevenshteinDistance([])
        for seq, true_st in zip(sequences, structures):
            pred_st = self.predict_structure(seq)
            dist = lev_calc.calculate(true_st, pred_st)
            distances.append(dist)
        return np.mean(distances)


if __name__ == "__main__":
    predictor = RNASecondaryStructurePredictor()

    sequences, structures = predictor.load_data('../data/RNA_strands.csv')
    sequences = [seq.replace('T', 'U') for seq in sequences]
    
    X, Y = predictor.fit_transform_data(sequences, structures)
    X_train, X_val, Y_train, Y_val, seq_train, seq_val, st_train, st_val = train_test_split(X, Y, sequences, structures, test_size=0.2, random_state=42)

    predictor.build_model()
    predictor.model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=32)
    
    predictor.model.save('rna_structure_model.keras')

    val_levenshtein = predictor.evaluate_levenshtein(seq_val, st_val)
    print(f"Validation Levenshtein Distance (CNN): {val_levenshtein}")

    test_seq = "GCGGCACCGUCCGCUCAAACAAACGG"
    predicted = predictor.predict_structure(test_seq)
    print("Sequence: ", test_seq)
    print("Predicted Structure: ", predicted)

    lev_calc = LevenshteinDistance([])

    with open('validation_results.txt', 'w') as f:
        f.write("Index\tSequence\tTrue_Structure\tCNN_Pred\tCNN_Lev\tZuker_Pred\tZuker_Lev\tNussinov_Pred\tNussinov_Lev\tOptNussinov_Pred\tOptNussinov_Lev\tFourRus_Pred\tFourRus_Lev\n")

        for i, (seq, true_st) in enumerate(zip(seq_val, st_val)):
            # CNN prediction
            cnn_pred = predictor.predict_structure(seq)
            cnn_lev = lev_calc.calculate(true_st, cnn_pred)

            # Zuker prediction
            zuker = ZukerAlgorithm(seq)
            zuker_pred = zuker.run()
            zuker_lev = lev_calc.calculate(true_st, zuker_pred)

            # Nussinov prediction
            nussinov = NussinovAlgorithm(seq)
            nussinov_pred = nussinov.run()
            nussinov_lev = lev_calc.calculate(true_st, nussinov_pred)

            # OptimizedNussinovAlgorithm prediction
            opt_nussinov = OptimizedNussinovAlgorithm(seq)
            opt_nussinov_pred = opt_nussinov.run()
            opt_nussinov_lev = lev_calc.calculate(true_st, opt_nussinov_pred)

            # OptimizedNussinovWithFourRussians prediction
            four_rus = OptimizedNussinovWithFourRussians(seq)
            four_rus_pred = four_rus.run()
            four_rus_lev = lev_calc.calculate(true_st, four_rus_pred)

            # Write the results for this sequence
            f.write(f"{i}\t{seq}\t{true_st}\t{cnn_pred}\t{cnn_lev}\t{zuker_pred}\t{zuker_lev}\t{nussinov_pred}\t{nussinov_lev}\t{opt_nussinov_pred}\t{opt_nussinov_lev}\t{four_rus_pred}\t{four_rus_lev}\n")

    print("Validation results written to validation_results.txt")
