import os
import argparse


def main(file_name):
    base_loc = "/Users/ethanhersch/Desktop/4775_final/CS4775_Project/src/rna_data/test-set-1-bpseq"
    file_path = os.path.join(base_loc, file_name)
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
    print(f"Sequence: {''.join(final)}")
    print(f"Dot-Bracket Structure: {''.join(dot_bracket_structure)}")
    return "".join(final), "".join(dot_bracket_structure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a FASTA file to dot-bracket notation."
    )
    parser.add_argument("file_name", help="The name of the FASTA file")

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the main function with the file name passed as a command-line argument
    main(args.file_name)
