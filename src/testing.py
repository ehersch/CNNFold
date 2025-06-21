import csv
from pprint import pprint
from folding_algorithms import (
    NussinovAlgorithm,
    ZukerAlgorithm,
    OptimizedNussinovAlgorithm,
    OptimizedNussinovWithFourRussians,
)
from nn_folding import RNASecondaryStructurePredictor as CNN_Fold
from utils import LevenshteinDistance
import argparse


def run_tests(csv_path, output_csv_path="", num=2, custom_scores=None):
    """
    Run folding algorithm tests and compute Levenshtein distance to the expected structures.
    """
    if num == 0:
        rna_strands = []
        expected_foldings = []

        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                rna_strands.append(row[0])
                expected_foldings.append(row[1])

        results = []
        average_distances = {
            "Zuker 1": 0,
            "Zuker 2": 0,
            "Zuker 3": 0,
            "Zuker 4": 0,
            "Zuker 5": 0,
        }
        total = 0

        for i, (strand, expected) in enumerate(zip(rna_strands, expected_foldings)):
            zuker_1 = ZukerAlgorithm(strand, 1)
            zuker_2 = ZukerAlgorithm(strand, 2)
            zuker_3 = ZukerAlgorithm(strand, 3)
            zuker_4 = ZukerAlgorithm(strand, 4)
            zuker_5 = ZukerAlgorithm(strand, 5)

            zuker_predicted_1 = zuker_1.run()
            zuker_predicted_2 = zuker_2.run()
            zuker_predicted_3 = zuker_3.run()
            zuker_predicted_4 = zuker_4.run()
            zuker_predicted_5 = zuker_5.run()

            levenshtein = LevenshteinDistance([])
            distances = {
                "Zuker 1": levenshtein.calculate(zuker_predicted_1, expected),
                "Zuker 2": levenshtein.calculate(zuker_predicted_2, expected),
                "Zuker 3": levenshtein.calculate(zuker_predicted_3, expected),
                "Zuker 4": levenshtein.calculate(zuker_predicted_4, expected),
                "Zuker 5": levenshtein.calculate(zuker_predicted_5, expected),
            }
            for alg in distances:
                average_distances[alg] += distances[alg]
            total += 1

            results.append(
                {
                    "Index": i + 1,
                    "RNA Strand": strand,
                    "Expected": expected,
                    "Predicted": {
                        "Zuker 1": zuker_predicted_1,
                        "Zuker 2": zuker_predicted_2,
                        "Zuker 3": zuker_predicted_3,
                        "Zuker 4": zuker_predicted_4,
                        "Zuker 5": zuker_predicted_5,
                    },
                    "Distances": distances,
                }
            )

        for result in results:
            print(f"Index: {result['Index']}")
            print(f"RNA Strand: {result['RNA Strand']}")
            print(f"Expected Structure: {result['Expected']}")
            print("Predictions:")
            pprint(result["Predicted"], indent=4)
            print("Levenshtein Distances:")
            pprint(result["Distances"], indent=4)
            print("-" * 70)
            print()

        print("Average Levenshtein Distances (normalized):")
        for key, total_distance in average_distances.items():
            print(f"{key}: {total_distance / total:.2f}")

        if output_csv_path != "":
            with open(output_csv_path, mode="w", newline="") as csvfile:
                fieldnames = [
                    "Index",
                    "RNA Strand",
                    "Expected",
                    "Zuker 1 Prediction",
                    "Zuker 2 Prediction",
                    "Zuker 3 Prediction",
                    "Zuker 4 Prediction",
                    "Zuker 5 Prediction",
                    "Zuker 1 Distance",
                    "Zuker 2 Distance",
                    "Zuker 3 Distance",
                    "Zuker 4 Distance",
                    "Zuker 5 Distance",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    writer.writerow(
                        {
                            "Index": result["Index"],
                            "RNA Strand": result["RNA Strand"],
                            "Expected": result["Expected"],
                            "Zuker 1 Prediction": result["Predicted"]["Zuker 1"],
                            "Zuker 2 Prediction": result["Predicted"]["Zuker 2"],
                            "Zuker 3 Prediction": result["Predicted"]["Zuker 3"],
                            "Zuker 4 Prediction": result["Predicted"]["Zuker 4"],
                            "Zuker 5 Prediction": result["Predicted"]["Zuker 5"],
                            "Zuker 1 Distance": result["Distances"]["Zuker 1"],
                            "Zuker 2 Distance": result["Distances"]["Zuker 2"],
                            "Zuker 3 Distance": result["Distances"]["Zuker 3"],
                            "Zuker 4 Distance": result["Distances"]["Zuker 4"],
                            "Zuker 5 Distance": result["Distances"]["Zuker 5"],
                        }
                    )
    if num == 1:
        rna_strands = []
        expected_foldings = []

        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                rna_strands.append(row[0])
                expected_foldings.append(row[1])

        results = []
        average_distances = {
            "Nussinov 1": 0,
            "Nussinov 2": 0,
        }
        total = 0

        cnn_fold = CNN_Fold(max_len=390)
        cnn_fold.build_model()
        cnn_fold.model.load_weights("rna_structure_model.h5")

        for i, (strand, expected) in enumerate(zip(rna_strands, expected_foldings)):
            nussinov_1 = NussinovAlgorithm(strand, 1)
            nussinov_2 = NussinovAlgorithm(strand, 2)

            nussinov_pred_1 = nussinov_1.run()
            nussinov_pred_2 = nussinov_2.run()

            levenshtein = LevenshteinDistance([])
            distances = {
                "Nussinov 1": levenshtein.calculate(nussinov_pred_1, expected),
                "Nussinov 2": levenshtein.calculate(nussinov_pred_2, expected),
            }
            for alg in distances:
                average_distances[alg] += distances[alg]
            total += 1

            results.append(
                {
                    "Index": i + 1,
                    "RNA Strand": strand,
                    "Expected": expected,
                    "Predicted": {
                        "Nussinov 1": nussinov_pred_1,
                        "Nussinov 2": nussinov_pred_2,
                    },
                    "Distances": distances,
                }
            )

        for result in results:
            print(f"Index: {result['Index']}")
            print(f"RNA Strand: {result['RNA Strand']}")
            print(f"Expected Structure: {result['Expected']}")
            print("Predictions:")
            pprint(result["Predicted"], indent=4)
            print("Levenshtein Distances:")
            pprint(result["Distances"], indent=4)
            print("-" * 70)
            print()

        print("Average Levenshtein Distances (normalized):")
        for key, total_distance in average_distances.items():
            print(f"{key}: {total_distance / total:.2f}")

        if output_csv_path != "":
            with open(output_csv_path, mode="w", newline="") as csvfile:
                fieldnames = [
                    "Index",
                    "RNA Strand",
                    "Expected",
                    "Nussinov 1 Prediction",
                    "Nussinov 2 Prediction",
                    "Nussinov 1 Distance",
                    "Nussinov 2 Distance",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    writer.writerow(
                        {
                            "Index": result["Index"],
                            "RNA Strand": result["RNA Strand"],
                            "Expected": result["Expected"],
                            "Nussinov 1 Prediction": result["Predicted"]["Nussinov 1"],
                            "Nussinov 2 Prediction": result["Predicted"]["Nussinov 2"],
                            "Nussinov 1 Distance": result["Distances"]["Nussinov 1"],
                            "Nussinov 2 Distance": result["Distances"]["Nussinov 2"],
                        }
                    )
    if num == 2:
        rna_strands = []
        expected_foldings = []

        cnn_fold = CNN_Fold(max_len=390)
        cnn_fold.build_model()
        cnn_fold.model.load_weights("rna_structure_model.keras")

        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                rna_strands.append(row[0])
                expected_foldings.append(row[1])

        results = []
        average_distances = {
            "Nussinov": 0,
            "Zuker": 0,
            "OptimizedNussinov": 0,
            "FourRussians": 0,
            "CNN_Fold": 0,
        }
        total = 0

        for i, (strand, expected) in enumerate(zip(rna_strands, expected_foldings)):
            nussinov = NussinovAlgorithm(strand)
            zuker = ZukerAlgorithm(strand)
            optimized_nussinov = OptimizedNussinovAlgorithm(strand)
            optimized_nussinov_four_russians = OptimizedNussinovWithFourRussians(strand)
            nussinov_predicted = nussinov.run()
            zuker_predicted = zuker.run()
            optimized_predicted = optimized_nussinov.run()
            four_russians_predicted = optimized_nussinov_four_russians.run()
            cnn_fold_predicted = cnn_fold.predict_structure(strand)

            levenshtein = LevenshteinDistance([])
            distances = {
                "Nussinov": levenshtein.calculate(nussinov_predicted, expected),
                "Zuker": levenshtein.calculate(zuker_predicted, expected),
                "OptimizedNussinov": levenshtein.calculate(
                    optimized_predicted, expected
                ),
                "FourRussians": levenshtein.calculate(
                    four_russians_predicted, expected
                ),
                "CNN_Fold": levenshtein.calculate(cnn_fold_predicted, expected),
            }
            for alg in distances:
                average_distances[alg] += distances[alg]
            total += 1

            results.append(
                {
                    "Index": i + 1,
                    "RNA Strand": strand,
                    "Expected": expected,
                    "Predicted": {
                        "Nussinov": nussinov_predicted,
                        "Zuker": zuker_predicted,
                        "OptimizedNussinov": optimized_predicted,
                        "FourRussians": four_russians_predicted,
                        "CNN_Fold": cnn_fold_predicted,
                    },
                    "Distances": distances,
                }
            )

        for result in results:
            print(f"Index: {result['Index']}")
            print(f"RNA Strand: {result['RNA Strand']}")
            print(f"Expected Structure: {result['Expected']}")
            print("Predictions:")
            pprint(result["Predicted"], indent=4)
            print("Levenshtein Distances:")
            pprint(result["Distances"], indent=4)
            print("-" * 70)
            print()

        print("Average Levenshtein Distances (normalized):")
        for key, total_distance in average_distances.items():
            print(f"{key}: {total_distance / total:.2f}")

        if output_csv_path != "":
            with open(output_csv_path, mode="w", newline="") as csvfile:
                fieldnames = [
                    "Index",
                    "RNA Strand",
                    "Expected",
                    "Nussinov Prediction",
                    "Zuker Prediction",
                    "OptimizedNussinov Prediction",
                    "FourRussians Prediction",
                    "CNN_Fold Prediction",
                    "Nussinov Distance",
                    "Zuker Distance",
                    "OptimizedNussinov Distance",
                    "FourRussians Distance",
                    "CNN_Fold Distance",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    writer.writerow(
                        {
                            "Index": result["Index"],
                            "RNA Strand": result["RNA Strand"],
                            "Expected": result["Expected"],
                            "Nussinov Prediction": result["Predicted"]["Nussinov"],
                            "Zuker Prediction": result["Predicted"]["Zuker"],
                            "OptimizedNussinov Prediction": result["Predicted"][
                                "OptimizedNussinov"
                            ],
                            "FourRussians Prediction": result["Predicted"][
                                "FourRussians"
                            ],
                            "CNN_Fold Prediction": result["Predicted"]["CNN_Fold"],
                            "Nussinov Distance": result["Distances"]["Nussinov"],
                            "Zuker Distance": result["Distances"]["Zuker"],
                            "OptimizedNussinov Distance": result["Distances"][
                                "OptimizedNussinov"
                            ],
                            "FourRussians Distance": result["Distances"][
                                "FourRussians"
                            ],
                            "CNN_Fold Prediction": result["Distances"]["CNN_Fold"],
                        }
                    )
    if num == 3:
        rna_strands = []
        expected_foldings = []

        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                rna_strands.append(row[0])
                expected_foldings.append(row[1])

        results = []
        average_distances = {
            "Zuker 1": 0,
            "Zuker 2": 0,
            "Zuker 3": 0,
            "Zuker 4": 0,
            "Zuker 5": 0,
            "Zuker Custom": 0,
        }
        total = 0

        for i, (strand, expected) in enumerate(zip(rna_strands, expected_foldings)):
            zuker_1 = ZukerAlgorithm(strand, 1)
            zuker_2 = ZukerAlgorithm(strand, 2)
            zuker_3 = ZukerAlgorithm(strand, 3)
            zuker_4 = ZukerAlgorithm(strand, 4)
            zuker_5 = ZukerAlgorithm(strand, 5)
            zuker_custom = ZukerAlgorithm(strand, 6, custom_scores)

            zuker_predicted_1 = zuker_1.run()
            zuker_predicted_2 = zuker_2.run()
            zuker_predicted_3 = zuker_3.run()
            zuker_predicted_4 = zuker_4.run()
            zuker_predicted_5 = zuker_5.run()
            zuker_predicted_custom = zuker_custom.run()

            levenshtein = LevenshteinDistance([])
            distances = {
                "Zuker 1": levenshtein.calculate(zuker_predicted_1, expected),
                "Zuker 2": levenshtein.calculate(zuker_predicted_2, expected),
                "Zuker 3": levenshtein.calculate(zuker_predicted_3, expected),
                "Zuker 4": levenshtein.calculate(zuker_predicted_4, expected),
                "Zuker 5": levenshtein.calculate(zuker_predicted_5, expected),
                "Zuker Custom": levenshtein.calculate(zuker_predicted_custom, expected),
            }
            for alg in distances:
                average_distances[alg] += distances[alg]
            total += 1

            results.append(
                {
                    "Index": i + 1,
                    "RNA Strand": strand,
                    "Expected": expected,
                    "Predicted": {
                        "Zuker 1": zuker_predicted_1,
                        "Zuker 2": zuker_predicted_2,
                        "Zuker 3": zuker_predicted_3,
                        "Zuker 4": zuker_predicted_4,
                        "Zuker 5": zuker_predicted_5,
                        "Zuker Custom": zuker_predicted_custom,
                    },
                    "Distances": distances,
                }
            )

        for result in results:
            print(f"Index: {result['Index']}")
            print(f"RNA Strand: {result['RNA Strand']}")
            print(f"Expected Structure: {result['Expected']}")
            print("Predictions:")
            pprint(result["Predicted"], indent=4)
            print("Levenshtein Distances:")
            pprint(result["Distances"], indent=4)
            print("-" * 70)
            print()

        print("Average Levenshtein Distances (normalized):")
        for key, total_distance in average_distances.items():
            print(f"{key}: {total_distance / total:.2f}")

        if output_csv_path != "":
            with open(output_csv_path, mode="w", newline="") as csvfile:
                fieldnames = [
                    "Index",
                    "RNA Strand",
                    "Expected",
                    "Zuker 1 Prediction",
                    "Zuker 2 Prediction",
                    "Zuker 3 Prediction",
                    "Zuker 4 Prediction",
                    "Zuker 5 Prediction",
                    "Zuker Custom Prediction",
                    "Zuker 1 Distance",
                    "Zuker 2 Distance",
                    "Zuker 3 Distance",
                    "Zuker 4 Distance",
                    "Zuker 5 Distance",
                    "Zuker Custom Distance",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    writer.writerow(
                        {
                            "Index": result["Index"],
                            "RNA Strand": result["RNA Strand"],
                            "Expected": result["Expected"],
                            "Zuker 1 Prediction": result["Predicted"]["Zuker 1"],
                            "Zuker 2 Prediction": result["Predicted"]["Zuker 2"],
                            "Zuker 3 Prediction": result["Predicted"]["Zuker 3"],
                            "Zuker 4 Prediction": result["Predicted"]["Zuker 4"],
                            "Zuker 5 Prediction": result["Predicted"]["Zuker 5"],
                            "Zuker Custom Prediction": result["Predicted"][
                                "Zuker Custom"
                            ],
                            "Zuker 1 Distance": result["Distances"]["Zuker 1"],
                            "Zuker 2 Distance": result["Distances"]["Zuker 2"],
                            "Zuker 3 Distance": result["Distances"]["Zuker 3"],
                            "Zuker 4 Distance": result["Distances"]["Zuker 4"],
                            "Zuker 5 Distance": result["Distances"]["Zuker 5"],
                            "Zuker Custom Distance": result["Distances"][
                                "Zuker Custom"
                            ],
                        }
                    )


import argparse
import sys

if __name__ == "__main__":
    csv_path = "../data/RNA_Strands.csv"
    output_csv_path = "../data/folding_results.csv"

    parser = argparse.ArgumentParser(
        description="Type of test you want to do (compare Zuker score matrices, Nussinov tracebacks, or compare all algorithms)."
    )
    parser.add_argument(
        "--type",
        type=int,
        required=True,
        help="The type of test you want to do: 0 (Zuker score matrices), 1 (Nussinov tracebacks), 2 (compare all algorithms), or 3 (Zuker with custom scoring).",
    )
    parser.add_argument(
        "--custom_scores",
        type=float,
        nargs=3,
        help="Custom score matrix values as three numbers (e.g., 1 2 3). Can only be used with --type 3.",
    )

    args = parser.parse_args()

    if args.custom_scores and args.type != 3:
        parser.error("--custom_scores can only be used when --type is set to 3.")

    run_tests(csv_path, output_csv_path, args.type, args.custom_scores or [1, 1, 1])
