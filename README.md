# Replicating Hansen (2015), "Punishment and Deterrence:
Evidence from Drunk Driving"

## The paper and its headline claim

- **Paper:** Hansen, Benjamin. 2015. "Punishment and Deterrence: Evidence from Drunk Driving." American Economic Review 105 (4): 1581–1617.
DOI: 10.1257/aer.20130189
- **Identification strategy:** Regression Discontinuity.
- **Headline claim:** A 2.1 percentage point decrease in DUI in the next four years for drivers who surpassed the 0.08 DUI BAC threshold. Located in Table 3, in the regression coefficient under column 1 "All tested drivers"
- **My replication:** −0.0221 (SE 0.0041), against the paper's −0.021: a gap of −5.45% as `python reproduce.py` prints it, meaning my estimate is slightly larger in magnitude. The gap is about a quarter of a standard error. The other seven replicated estimates, including the 0.15 threshold, are in `output/headline_table.csv` and the report.

## Reproduce the headline table

Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python reproduce.py
```

`python reproduce.py` is the one command. Once `MY_PAPER` is set in `download_data.py`,
it downloads the data if `data/` does not have it, runs `notebooks/01_positioning_and_replication.ipynb` and then
`notebooks/02_robustness.ipynb` top to bottom in a fresh kernel, and prints the headline
table from `output/headline_table.csv`. Executed copies of the notebooks, with their
outputs, go to `output/executed/`.

- **Expected runtime:** about 6 minutes, measured from a clean clone in Google Colab (5 minutes 32 seconds). Notebook 01 takes about 10 seconds; almost all the rest is notebook 02, where `rdrobust`'s data-driven bandwidth selection is slow on this sample. The run has not frozen if notebook 02 takes several minutes.

### Running a notebook in Google Colab
Each notebook has a commented-out setup cell at the top. Uncomment it and run it first: it clones this repository so the notebook can find `download_data.py`, and in notebook 02 it also runs notebook 01 (to create `output/headline_table.csv`) and installs `rdrobust`. Keep it commented out when running `python reproduce.py`.


## Data: source and licence

- **Source:** https://raw.githubusercontent.com/scunning1975/mixtape/master/hansen_dwi.dta
- **Included or downloaded:** Downloaded. The data are not stored in this repository; `download_data.py` fetches the file on the first run, and `.gitignore` excludes `data/*.dta`. The file is public, in a repository whose MIT-style licence would permit including a copy, but downloading keeps the repository small and the source traceable to a single public URL. The underlying data were collected by Hansen from Washington State administrative records.

The template downloads by default: `.gitignore` excludes `data/*.dta`, and
`download_data.py` fetches the files on the first run. Two facts to check before you
decide:

- The five files from `scunning1975/mixtape` (castle, card, lee, hansen, abortion) are in a
  repository under an MIT-style licence that permits copies if its notice is kept. The
  underlying data were collected by the original authors.
- The Dehejia-Wahba files on `users.nber.org/~rdehejia` state no licence. Download them;
  do not commit them.

If you decide to include the data, delete the `data/*.dta` line from `.gitignore`, commit
the files with the licence notice, and say so above.

## Repository map

| Path | What it is |
|---|---|
| `reproduce.py` | The one command: runs the notebooks in name order, prints the headline table |
| `download_data.py` | Data URLs for the six papers; `python download_data.py --paper card` fetches one into `data/` |
| `notebooks/01_positioning_and_replication.ipynb` | Sections 1 and 2: positioning notes, the replication, the theirs-vs-mine table, the discrepancy log. **No AI.** |
| `notebooks/02_robustness.ipynb` | Section 3: four robustness exercises. AI allowed, logged in `ai-appendix.md` |
| `output/headline_table.csv` | The theirs-vs-mine table, written by notebook 01 |
| `output/` | Every table and figure the report uses |
| `report/` | The report PDF |
| `ai-appendix.md` | P.R.I.M.E. entries and full transcripts for Sections 3 to 6, and the no-AI statement for Sections 1 and 2 |
| `AUDIT.md` | How to run the peer audit you file on a classmate's repository |
| `requirements.txt` | Pinned package versions |

TODO: add a row for every file you add.

## Use of AI

Sections 1 and 2 were completed without AI.

AI Was used in Section 3. It was used to create a generic reusable function built on my own written section 2 regressions that could be used across each exercise. The first function call was written manually for each exercise, then AI was asked to write a for loop that looped that call over each function specification and save the results in a table.
