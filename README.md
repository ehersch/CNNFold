# CS4775_Project (Migrated from Cornell GitHub for public viewing)

An implementation and further exploration of the Nussinov Algorithm and other RNA folding prediction methods, including deep learning.

First clone our repository by running

```
git clone https://github.com/ehersch/CNNFold.git
```

(this may take a while)

# Setting up the environment

We have an `environment.yml` file to create a conda environment. Follow these steps to set up the environment:

## Prerequisites

Make sure you have Conda installed. You can download and install it from [Conda's official website](https://docs.conda.io/en/latest/miniconda.html).

## Steps to Set Up the Environment

Everything is done in bash, so if you're currently on zsh, either switch from command line or use
VSCode's bash terminal.

1. **Create the Environment**
   Run the following command to create the Conda environment using the `environment.yml` file (this may take a while):
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the Environment
   Once the environment is created, activate it:
   ```bash
   conda activate CS4775
   ```

# Data Instructions

Our relevant dataset (https://zenodo.org/records/7788537) has already been parsed
and this is included in the 'data' directory in the 'RNA_strands.csv' file has
contains this processed data. It is not necessary to redo this process as the
data is already contained in this CSV.

Yet, to reproduce these results do the following:

Download 'test-set-1-bpseq.zip' and 'test-set-1-fasta.zip' from https://zenodo.org/records/7788537
and unzip both

Move both files to the 'CS4775_PROJECT/data' directory

```bash
   cd data
```

```bash
   python parser.py --bpseq_folder test-set-1-bpseq --fasta_folder test-set-1-fasta --output_csv <CSV_NAME>.csv
```

And your results will appear in 'data/<CSV_NAME>.csv'

# Testing Instructions

To test the functionality of the program, follow these steps:

### Step 1: Navigate to the Source Directory

Change your working directory to `src`:

```bash
cd src
```

Run the following commands:

```bash
   python testing.py --type 0

python testing.py --type 1

python testing.py --type 2

python testing.py --type 3 --custom_scores s1 s2 s3
```

These algorithms run on our base dataset in 'data/RNA_strands.csv'

The first option allows the user to compare 5 different score matrices for the
Zuker algorithm.

The second option allows the user to compare the results of the two versions of
traceback from the Nussnov algorithm.

The third option allows the user to compare the results for running on the
Nussinov, Zuker, Optimized Nussnov, Four Russians algorithms.

The last option allows the user to create a custom score matrix for the Zuker
algorithm. These scores are penalties, so make sure all three entries are
negative. Here, s1 is the ('A', 'U') score, s2 is the ('G', 'C') score, and s3
is the ('U', 'G') score.

Optionally, feel free to run GUI.py to test our algorithms with custom
sequences. Simply open the GUI via

```bash
python gui.py
```

input your sequence, select your algorithm, and press Generate! in order to view
the folding structure and DP table used to generate the result.
