class LevenshteinDistance:
    def __init__(self, sequences):
        """
        Initialize with a list of RNA sequences (foldings).
        """
        self.sequences = sequences
        self.num_sequences = len(sequences)

    def calculate(self, seq1, seq2):
        """
        Calculate the Levenshtein distance between two sequences.
        """
        m = len(seq1)
        n = len(seq2)

        d = [[0]*(n+1) for _ in range(m+1)]

        for i in range(1,m+1):
            d[i][0] = i

        for j in range(1,n+1):
            d[0][j] = j

        subCost = 0
        
        for j in range(1,n+1):
            for i in range(1,m+1):
                if seq1[i-1] == seq2[j-1]:
                    subCost = 0
                else:
                    subCost = 1

                deletion = d[i-1][j] + 1
                insertion = d[i][j-1] + 1
                sub = d[i-1][j-1] + subCost
                d[i][j] = min(deletion,insertion,sub)

        return d[m][n]

    def calculate_pairwise_distances(self):
        """
        Calculate Levenshtein distances between all pairs of RNA sequences.
        """
        distances = {}
        for i in range(self.num_sequences):
            for j in range(i + 1, self.num_sequences):
                seq1 = self.sequences[i]
                seq2 = self.sequences[j]
                dist = self.calculate(seq1, seq2)
                distances[(i, j)] = dist
        return distances

    def calculate_multiple_distances(self):
        """
        Calculate Levenshtein distances between all RNA foldings.
        """
        result = []
        pairwise_distances = self.calculate_pairwise_distances()

        for (i, j), dist in pairwise_distances.items():
            result.append(
                f"Levenshtein Distance between Sequence {i} and Sequence {j}: {dist}"
            )

        return result
