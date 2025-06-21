import os
import csv
import argparse


def parse_bpseq(file_path):
    """Parse a BPSEQ file to extract dot-bracket structure."""
    with open(file_path, "r") as file:
        fasta_content = file.read()

    lines = fasta_content.splitlines()
    sequence = []
    for line in lines:
        if not line.startswith(">"):  # Skip header lines
            line_list = line.strip().split()
            line_list[2] = str(int(line_list[2]) - 1)
            sequence += [line_list[1:]]

    dot_bracket_structure = ["."] * len(sequence)

    counts = [0] * len(sequence)

    for count, (_, idx) in enumerate(sequence):
        if idx == "-1":
            continue
        if int(idx) >= count:
            dot_bracket_structure[int(idx)] = ")"
            counts[int(idx)] += 1
        else:
            dot_bracket_structure[int(idx)] = "("

    # Print or return the result
    final = [seq[0] for seq in sequence]
    return "".join(final), "".join(dot_bracket_structure)


def process_files(bpseq_folder, fasta_folder, output_csv):
    """Process files in BPSEQ and FASTA folders and write results to a CSV file."""
    results = []

    for bpseq_file in os.listdir(bpseq_folder):
        if bpseq_file.endswith(".bpseq"):
            fasta_file = bpseq_file.replace(".bpseq", ".fa")
            bpseq_path = os.path.join(bpseq_folder, bpseq_file)
            fasta_path = os.path.join(fasta_folder, fasta_file)

            if os.path.exists(fasta_path):  # Ensure corresponding FASTA file exists
                sequence, dot_bracket = parse_bpseq(bpseq_path)
                results.append((sequence, dot_bracket))
            else:
                print(f"Warning: No corresponding FASTA file for {bpseq_file}")

    # Write results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Sequence", "Dot-Bracket Structure"])  # Header row
        writer.writerows(results)

    print(f"Dot-bracket results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BPSEQ files to dot-bracket notation and save to a CSV file."
    )
    parser.add_argument(
        "--bpseq_folder",
        required=True,
        help="Path to the folder containing BPSEQ files",
    )
    parser.add_argument(
        "--fasta_folder",
        required=True,
        help="Path to the folder containing FASTA files",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to the output CSV file",
    )

    args = parser.parse_args()

    process_files(args.bpseq_folder, args.fasta_folder, args.output_csv)
