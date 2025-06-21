class ZukerAlgorithm:
    def __init__(self, sequence, score_matrix=1, custom_score_matrix=None):
        """
        Initialize the Zuker algorithm with an RNA sequence.
        """
        self.sequence = sequence
        self.n = len(sequence)
        self.dp_table = [[float("inf")] * self.n for _ in range(self.n)]
        self.bulge_table = [[float("inf")] * self.n for _ in range(self.n)]
        self.structure = ["."] * self.n
        self.version = score_matrix
        self.custom_score_matrix = custom_score_matrix

    def delta(self, x, y):
        """
        Return the thermodynamic stability of a base pair.
        """
        if not self.custom_score_matrix and self.version == 1:
            pairs = {
                ("A", "U"): -2,
                ("U", "A"): -2,
                ("G", "C"): -3,
                ("C", "G"): -3,
                ("G", "U"): -1,
                ("U", "G"): -1,
            }
        if not self.custom_score_matrix and self.version == 2:
            pairs = {
                ("A", "U"): -0.9,
                ("U", "A"): -0.9,
                ("G", "C"): -2.9,
                ("C", "G"): -2.9,
                ("G", "U"): -1.1,
                ("U", "G"): -1.1,
            }
        if not self.custom_score_matrix and self.version == 3:
            pairs = {
                ("A", "U"): -0.5,
                ("U", "A"): -0.5,
                ("G", "C"): -0.5,
                ("C", "G"): -0.5,
                ("G", "U"): -0.5,
                ("U", "G"): -0.5,
            }

        if not self.custom_score_matrix and self.version == 4:
            pairs = {
                ("A", "U"): -20,
                ("U", "A"): -20,
                ("G", "C"): -20,
                ("C", "G"): -20,
                ("G", "U"): -20,
                ("U", "G"): -20,
            }
        if not self.custom_score_matrix and self.version == 5:
            pairs = {
                ("A", "U"): -1,
                ("U", "A"): -1,
                ("G", "C"): -40,
                ("C", "G"): -40,
                ("G", "U"): -10,
                ("U", "G"): -10,
            }
        if self.custom_score_matrix:
            pairs = {
                ("A", "U"): self.custom_score_matrix[0],
                ("U", "A"): self.custom_score_matrix[0],
                ("G", "C"): self.custom_score_matrix[1],
                ("C", "G"): self.custom_score_matrix[1],
                ("G", "U"): self.custom_score_matrix[2],
                ("U", "G"): self.custom_score_matrix[2],
            }
        return pairs.get((x, y), 0)

    def fill_dp_table(self):
        """
        Fill the dynamic programming table considering both loop stability and optimal energy configurations.
        """
        for i in range(self.n):
            self.dp_table[i][i] = 0
            if i < self.n - 1:
                self.dp_table[i][i + 1] = 0

        for span in range(2, self.n):  # The span between i and j
            for i in range(self.n - span):
                j = i + span
                options = [
                    self.dp_table[i + 1][j],  # Skip i
                    self.dp_table[i][j - 1],  # Skip j
                    self.dp_table[i + 1][j - 1]
                    + self.delta(self.sequence[i], self.sequence[j]),  # Pair i, j
                ]
                # Consider bifurcations
                for k in range(i + 1, j):
                    options.append(self.dp_table[i][k] + self.dp_table[k + 1][j])
                self.dp_table[i][j] = min(options)

    def traceback(self, i, j):
        """
        Reconstruct the secondary structure from the DP table.
        """
        if i >= j:
            return []
        if self.dp_table[i + 1][j] == self.dp_table[i][j]:
            return self.traceback(i + 1, j)
        if self.dp_table[i][j - 1] == self.dp_table[i][j]:
            return self.traceback(i, j - 1)
        if (
            self.dp_table[i + 1][j - 1] + self.delta(self.sequence[i], self.sequence[j])
            == self.dp_table[i][j]
        ):
            return [(i, j)] + self.traceback(i + 1, j - 1)
        for k in range(i + 1, j):
            if self.dp_table[i][k] + self.dp_table[k + 1][j] == self.dp_table[i][j]:
                return self.traceback(i, k) + self.traceback(k + 1, j)
        return []

    def get_structure(self):
        """
        Generate the secondary structure by performing a traceback.
        """
        base_pairs = self.traceback(0, self.n - 1)
        for i, j in base_pairs:
            self.structure[i] = "("
            self.structure[j] = ")"
        return "".join(self.structure)

    def run(self):
        """
        Run the Zuker algorithm to predict the RNA secondary structure.
        """
        self.fill_dp_table()
        return self.get_structure()


class NussinovAlgorithm:
    def __init__(self, sequence, traceback_option=1):
        """
        Initialize the algorithm with an RNA sequence.
        """
        self.sequence = sequence
        self.n = len(sequence)
        self.dp_table = [[None] * self.n for _ in range(self.n)]
        for i in range(1, self.n):
            self.dp_table[i][i - 1] = 0
            self.dp_table[i][i] = 0
        self.dp_table[0][0] = 0
        # print(self.dp_table)
        self.structure = ["."] * self.n
        self.traceback_option = traceback_option

    def delta(self, x, y):
        if x == "U" and y == "A" or x == "A" and y == "U":
            return 1
        if x == "C" and y == "G" or x == "G" and y == "C":
            return 1
        if x == "G" and y == "U" or x == "U" and y == "G":
            return 1
        return 0

    def fill_dp_table(self):
        """
        Fill the DP table according to the rules of the Nussinov algorithm.
        """
        for N in range(1, self.n):
            for i in range(0, self.n - N):
                # print(i)
                j = N + i
                gamma_1 = self.dp_table[i + 1][j]
                gamma_2 = self.dp_table[i][j - 1]
                gamma_3 = self.dp_table[i + 1][j - 1] + self.delta(
                    self.sequence[j], self.sequence[i]
                )
                cur_list = [-float("inf")]
                for k in range(i + 1, j):
                    cur_list += [self.dp_table[i][k] + self.dp_table[k + 1][j]]
                gamma_4 = max(cur_list)
                gamma_list = [gamma_1, gamma_2, gamma_3, gamma_4]
                cur_max = max(gamma_list)
                self.dp_table[i][j] = cur_max

    def traceback_2(self):
        """
        Reconstruct the secondary structure from the DP table.
        """
        base_pairs = []
        stack = []
        # print(self.dp_table)
        stack.append((0, self.n - 1))
        while stack:
            i, j = stack.pop()
            # print(self.dp_table[i][j])
            if i >= j:
                continue
            elif self.dp_table[i + 1][j] == self.dp_table[i][j]:
                stack.append((i + 1, j))
            elif self.dp_table[i][j - 1] == self.dp_table[i][j]:
                stack.append((i, j - 1))

            elif (
                self.dp_table[i + 1][j - 1]
                + self.delta(self.sequence[j], self.sequence[i])
                == self.dp_table[i][j]
            ):
                base_pairs.append((i, j))
                stack.append((i + 1, j - 1))

            else:
                for k in range(i + 1, j):
                    if (
                        self.dp_table[i][k] + self.dp_table[k + 1][j]
                        == self.dp_table[i][j]
                    ):
                        stack.append((k + 1, j))
                        stack.append((i, k))
                        break
        return base_pairs

    def traceback_1(self):
        """
        Reconstruct the secondary structure from the DP table.
        """
        base_pairs = []
        stack = []
        # print(self.dp_table)
        stack.append((0, self.n - 1))
        while stack:
            i, j = stack.pop()
            # print(self.dp_table[i][j])
            if i >= j:
                continue
            elif self.dp_table[i][j - 1] == self.dp_table[i][j]:
                stack.append((i, j - 1))
            else:
                for k in range(i, j):
                    if self.delta(self.sequence[j], self.sequence[k]) == 1:
                        if (
                            self.dp_table[i][k - 1] + self.dp_table[k + 1][j - 1] + 1
                            == self.dp_table[i][j]
                        ):
                            base_pairs.append((k, j))
                            stack.append((k + 1, j - 1))
                            stack.append((i, k - 1))
                            break
        return base_pairs

    def traceback(self):
        if self.traceback_option == 1:
            return self.traceback_1()
        else:
            return self.traceback_2()

    def get_structure(self):
        """
        Generate the secondary structure by performing a traceback.
        """
        base_pairs = self.traceback()
        # print(base_pairs)
        for i, j in base_pairs:
            self.structure[i] = "("
            self.structure[j] = ")"
        return "".join(self.structure)

    def run(self):
        """
        Run the Nussinov algorithm to get the secondary structure.
        """
        self.fill_dp_table()
        return self.get_structure()


class OptimizedNussinovAlgorithm:
    def __init__(self, sequence, traceback_option=1):
        """
        Initialize the algorithm with an RNA sequence.
        """
        self.sequence = sequence
        self.n = len(sequence)
        self.dp_table = [[None] * self.n for _ in range(self.n)]
        self.dp_lookup = {}  # This will store precomputed results for pairs (i, j)
        self.traceback_option = traceback_option

        # Initialize base cases
        for i in range(1, self.n):
            self.dp_table[i][i - 1] = 0
            self.dp_table[i][i] = 0
        self.dp_table[0][0] = 0
        self.structure = ["."] * self.n

        # Precompute delta values for all pairs
        self.delta_lookup = {}
        self.precompute_deltas()

    def delta(self, x, y):
        """
        Get the delta (energy) value for a base pair (x, y).
        """
        if (x, y) in self.delta_lookup:
            return self.delta_lookup[(x, y)]
        if x == "U" and y == "A" or x == "A" and y == "U":
            self.delta_lookup[(x, y)] = 1
            return 1
        if x == "C" and y == "G" or x == "G" and y == "C":
            self.delta_lookup[(x, y)] = 1
            return 1
        if x == "G" and y == "U" or x == "U" and y == "G":
            self.delta_lookup[(x, y)] = 1
            return 1
        self.delta_lookup[(x, y)] = 0
        return 0

    def precompute_deltas(self):
        """
        Precompute delta values for all possible base pairings (i, j).
        """
        nucleotides = ["A", "U", "C", "G"]
        for x in nucleotides:
            for y in nucleotides:
                self.delta(x, y)

    def fill_dp_table(self):
        """
        Fill the DP table using the Four Russians method for precomputing results.
        """
        for N in range(1, self.n):
            for i in range(0, self.n - N):
                j = N + i
                # First check if the result for this pair (i, j) has been computed
                if (i, j) in self.dp_lookup:
                    self.dp_table[i][j] = self.dp_lookup[(i, j)]
                else:
                    # Compute the dp values and cache it in the lookup table
                    gamma_1 = self.dp_table[i + 1][j]
                    gamma_2 = self.dp_table[i][j - 1]
                    gamma_3 = self.dp_table[i + 1][j - 1] + self.delta(
                        self.sequence[j], self.sequence[i]
                    )
                    cur_list = [-float("inf")]
                    for k in range(i + 1, j):
                        cur_list += [self.dp_table[i][k] + self.dp_table[k + 1][j]]
                    gamma_4 = max(cur_list)
                    gamma_list = [gamma_1, gamma_2, gamma_3, gamma_4]
                    cur_max = max(gamma_list)
                    self.dp_table[i][j] = cur_max
                    # Cache the result for future reference
                    self.dp_lookup[(i, j)] = cur_max

    def traceback_2(self):
        """
        Reconstruct the secondary structure from the DP table.
        """
        base_pairs = []
        stack = []
        # print(self.dp_table)
        stack.append((0, self.n - 1))
        while stack:
            i, j = stack.pop()
            # print(self.dp_table[i][j])
            if i >= j:
                continue
            elif self.dp_table[i + 1][j] == self.dp_table[i][j]:
                stack.append((i + 1, j))
            elif self.dp_table[i][j - 1] == self.dp_table[i][j]:
                stack.append((i, j - 1))

            elif (
                self.dp_table[i + 1][j - 1]
                + self.delta(self.sequence[j], self.sequence[i])
                == self.dp_table[i][j]
            ):
                base_pairs.append((i, j))
                stack.append((i + 1, j - 1))

            else:
                for k in range(i + 1, j):
                    if (
                        self.dp_table[i][k] + self.dp_table[k + 1][j]
                        == self.dp_table[i][j]
                    ):
                        stack.append((k + 1, j))
                        stack.append((i, k))
                        break
        return base_pairs

    def traceback_1(self):
        """
        Reconstruct the secondary structure from the DP table.
        """
        base_pairs = []
        stack = []
        # print(self.dp_table)
        stack.append((0, self.n - 1))
        while stack:
            i, j = stack.pop()
            # print(self.dp_table[i][j])
            if i >= j:
                continue
            elif self.dp_table[i][j - 1] == self.dp_table[i][j]:
                stack.append((i, j - 1))
            else:
                for k in range(i, j):
                    if self.delta(self.sequence[j], self.sequence[k]) == 1:
                        if (
                            self.dp_table[i][k - 1] + self.dp_table[k + 1][j - 1] + 1
                            == self.dp_table[i][j]
                        ):
                            base_pairs.append((k, j))
                            stack.append((k + 1, j - 1))
                            stack.append((i, k - 1))
                            break
        return base_pairs

    def traceback(self):
        if self.traceback_option == 1:
            return self.traceback_1()
        else:
            return self.traceback_2()

    def get_structure(self):
        """
        Generate the secondary structure by performing a traceback.
        """
        base_pairs = self.traceback()
        for i, j in base_pairs:
            self.structure[i] = "("
            self.structure[j] = ")"
        return "".join(self.structure)

    def run(self):
        """
        Run the Nussinov algorithm to get the secondary structure.
        """
        self.fill_dp_table()
        return self.get_structure()


class OptimizedNussinovWithFourRussians:
    def __init__(self, sequence):
        """
        Initialize the algorithm with an RNA sequence.
        """
        self.sequence = sequence
        self.n = len(sequence)
        self.dp_table = [[0] * self.n for _ in range(self.n)]
        self.structure = ["."] * self.n

    def delta(self, x, y):
        """
        Scoring function for base pairs (matches Nussinov's scoring).
        """
        valid_pairs = {
            ("A", "U"),
            ("U", "A"),
            ("C", "G"),
            ("G", "C"),
            ("G", "U"),
            ("U", "G"),
        }
        return 1 if (x, y) in valid_pairs else 0

    def fill_dp_table(self):
        """
        Fill the DP table using the Four Russians optimization.
        """
        for span in range(1, self.n):
            for i in range(self.n - span):
                j = i + span
                options = [
                    self.dp_table[i + 1][j],
                    self.dp_table[i][j - 1],
                    self.dp_table[i + 1][j - 1]
                    + self.delta(self.sequence[i], self.sequence[j]),
                ]
                for k in range(i + 1, j):
                    options.append(self.dp_table[i][k] + self.dp_table[k + 1][j])
                self.dp_table[i][j] = max(options)

    def traceback(self):
        """
        Reconstruct the secondary structure from the DP table.
        """
        base_pairs = []
        stack = [(0, self.n - 1)]
        while stack:
            i, j = stack.pop()
            if i >= j:
                continue
            if self.dp_table[i + 1][j] == self.dp_table[i][j]:
                stack.append((i + 1, j))
            elif self.dp_table[i][j - 1] == self.dp_table[i][j]:
                stack.append((i, j - 1))
            elif (
                self.dp_table[i + 1][j - 1]
                + self.delta(self.sequence[i], self.sequence[j])
                == self.dp_table[i][j]
            ):
                base_pairs.append((i, j))
                stack.append((i + 1, j - 1))
            else:
                for k in range(i + 1, j):
                    if (
                        self.dp_table[i][k] + self.dp_table[k + 1][j]
                        == self.dp_table[i][j]
                    ):
                        stack.append((i, k))
                        stack.append((k + 1, j))
                        break
        return base_pairs

    def get_structure(self):
        """
        Generate the secondary structure by performing a traceback.
        """
        base_pairs = self.traceback()
        for i, j in base_pairs:
            self.structure[i] = "("
            self.structure[j] = ")"
        return "".join(self.structure)

    def run(self):
        """
        Run the Four Russians optimized Nussinov algorithm.
        """
        self.fill_dp_table()
        return self.get_structure()


def main():
    # Example RNA sequence. The first is found from TB
    sequence = "GCAUCUAUGC"
    sequence_1 = "GGGAAAUCC"
    sequence_2 = "AAUCUGUUACGCA"
    # expected
    sequence_3 = "GCACGACG"

    # This is confirmed correct structure for the cat
    # This data was found from https://www.kaggle.com/code/pyjeffa/a-zuker-like-rna-folding-algorithm
    tRNA_Cat, tRNA_Cat_ref = (
        "UGGCUCAGUCAGUAGAGCAUGAGAUUCUUGAUCUCAGGGCCAUGAGUUCAAGCCCAU",
        "..((((........)))).(((((.......))))).....(((((......)))))",
    )

    # This is found from https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05238-8
    # This is the correct folding too below
    rna_4, rna_4_ref = (
        "AGAGAAUUU",
        "(((...)))",
    )

    # This was found using http://rna.tbi.univie.ac.at//cgi-bin/RNAWebSuite/RNAfold.cgi?PAGE=3&ID=zbztDEz9DU
    # This is the correct folding too below
    rna_5, rna_5_ref = (
        "AUAUUUAGGAU",
        "...........",
    )

    # Initialize and run the Nussinov algorithm
    nussinov = NussinovAlgorithm(rna_5)
    structure = nussinov.run()

    # Output the result
    print(f"RNA Sequence: {rna_4}")
    print(f"Nussinov Secondary Structure: {structure}")

    optimized_nussinov = OptimizedNussinovAlgorithm(tRNA_Cat)
    optimized_stucture = optimized_nussinov.run()

    # Output the result
    # print(f"RNA Sequence: {tRNA_Cat}")
    print(f"Optimized Nussinov Secondary Structure: {optimized_stucture}")

    optimized_nussinov_with_four_russians = OptimizedNussinovWithFourRussians(tRNA_Cat)
    optimized_four_rus_stucture = optimized_nussinov_with_four_russians.run()

    # Output the result
    # print(f"RNA Sequence: {tRNA_Cat}")
    print(
        f"Optimized Nussinov With Four Russians Secondary Structure: {optimized_four_rus_stucture}"
    )

    # Initialize and run the Nussinov algorithm
    zuker = ZukerAlgorithm(tRNA_Cat)
    zuker_structure = zuker.run()

    # Output the result
    # print(f"RNA Sequence: {tRNA_Cat}")
    print(f"Zuker Secondary Structure: {zuker_structure}")


if __name__ == "__main__":
    main()
