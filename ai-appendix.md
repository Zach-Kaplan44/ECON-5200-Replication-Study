# AI Appendix

## Sections 1 and 2

Sections 1 and 2 were completed without AI.

Name: Zachary Kaplan
Date: 11/22/26

Using AI in Sections 1 or 2 is an integrity violation. Do not edit the statement above; if it
is not true, talk to the instructor before you submit.

## How to file an entry

Sections 3 to 6 allow AI as a documented co-pilot under P.R.I.M.E. File one entry for each
piece of work you used it on, under the section it belongs to. Paste the full exchange: every
prompt as you sent it and every reply as it came back, including what you rejected. A summary
is not a transcript. If you cannot explain a line of what the model gave you, you cannot
submit it.

If you did not use AI in a section, write "No AI used in this section." under its heading.

Copy this block for each entry:

```
### Entry: TODO what it was for (for example, Exercise 1, not-yet-treated comparison)

- Model and version: TODO
- Date: TODO

**[Prep]** TODO: the role you gave the model.
**[Request]** TODO: what you asked for.
**[Iterate]** TODO: the constraints you added, and the version-2 prompt.
**[Mechanism Check]** TODO: what you asked it to explain, and whether the explanation held.
**[Evaluate]** TODO: how you checked the output against your own data or results.

**Full transcript**

TODO: paste the exchange here, unedited.

**What I changed, and why:** TODO, one line per change.
```

## Section 3: Robustness
### Reusable RD function and reproduction check based on section 2 code (Section 3)

- Model and version: Claude Opus 5
- Date: 11/22/26

**[Prep]** No explicit role assigned. I supplied context instead: my eight Section 2 regressions, a description of what they replicate (Hansen 2015, BAC cutoffs 0.08 and 0.15, two bandwidths, with and without controls), and the rules the function had to follow.

**[Request]** Turn the eight regressions into one reusable function, with inputs for the data, cutoff, bandwidth, controls, standard error type, treatment rule, and excluded years. It had to filter the window on integer `low_score` with both edges included, default to `>` at 150 and `>=` elsewhere, use unweighted OLS with my existing dummy columns, drop constant controls, leave the input data unmodified, return an unrounded jump, and come with a check cell reproducing my eight estimates.

**[Iterate]** The first version had five problems, and my version-2 prompt fixed each: (1) a false claim that my 0.08 windows had lost boundary rows to floating-point error; (2) a check cell that referenced notebook 01 variables, so it couldn't run in notebook 02, rewritten to read `output/headline_table.csv` by row label at `atol=1e-10`; (3) an `assert` that always passed, replaced with a before-and-after comparison of `dwi`'s columns; (4) two inaccurate comments, one attributing the strict 0.15 rule to Hansen rather than the statute, one overstating the float issue; (5) no check that `year` is numeric before excluding years.

**[Mechanism Check]** I asked for line-by-line comments. The AI's explanation of the float issue did not hold: I tested `abs(0.13 - 0.08)` in float64, found it is exactly 0.05, and confirmed that the boundary rows at 30, 130, 55, and 105 are kept. The float error only affected the 0.15 window at 0.200. I corrected the claim in the version-2 prompt.

**[Evaluate]** The check cell reproduced all eight Section 2 estimates exactly, from the saved headline table, with sample sizes of 93,899, 50,016, 139,033, and 75,909 matching my Section 2 windows, and `dwi` unchanged.

**Full transcript**

Below are eight RD regressions from my replication of Hansen (2015), at BAC cutoffs 0.08 and 0.15, two bandwidths each, with and without controls. Turn them into one function I can reuse for robustness checks.

CUTOFF = 0.08
# Converts low_score into the same units as bac
dwi['bac_min'] = dwi['low_score'] / 1000
#Creates age and year dummies and adds the dummy columns to the dataset.
#This is done because Hansen uses age and year fixed effects as controls rather than raw age and year
age_dummy = pd.get_dummies(dwi['aged'], drop_first=True, prefix='Age').astype(int)
year_dummy = pd.get_dummies(dwi['year'], drop_first=True, prefix='Year').astype(int)
dwi[age_dummy.columns] = age_dummy
dwi[year_dummy.columns] = year_dummy
RUNNING = 'bac_min'
H = 0.05
KERNEL = "rectangular"
CONTROLS = ['male', 'white',] + list(age_dummy.columns) + list(year_dummy.columns)
dwi['above_cutoff'] = (dwi[RUNNING] >= CUTOFF).astype(int)

in_band = dwi[np.abs(dwi[RUNNING] - CUTOFF) <= H]
bac_centered = in_band[RUNNING] - CUTOFF

rd_cols = pd.DataFrame(
  {"above_cutoff": in_band['above_cutoff'], "bac_centered": bac_centered,
   "above_centered": in_band['above_cutoff'] * bac_centered}
)
X = pd.concat([rd_cols, in_band[CONTROLS]], axis=1)
X = sm.add_constant(X)

rd_fit = sm.OLS(in_band['recidivism'], X).fit(cov_type='cluster', cov_kwds={'groups': in_band['bac_min']})  #Used Clustered SE as that is what Hansen uses in his regression
JUMP = float(rd_fit.params['above_cutoff'])

no_controls = sm.add_constant(rd_cols)
rd_fit_nocontrol = sm.OLS(in_band['recidivism'], no_controls).fit(cov_type='cluster', cov_kwds={'groups': in_band['bac_min']})  #Used Clustered SE as that is what Hansen uses in his regression
JUMP_nocontrol = float(rd_fit_nocontrol.params['above_cutoff'])


print(JUMP)
print(JUMP_nocontrol)
print(rd_fit.nobs)
print(rd_fit_nocontrol.nobs)

# Model for narrower threshold
H_NARROW = 0.025
in_band_narrow = dwi[np.abs(dwi[RUNNING] - CUTOFF) <= H_NARROW]
bac_centered_narrow = in_band_narrow[RUNNING] - CUTOFF

rd_cols_narrow = pd.DataFrame(
  {"above_cutoff": in_band_narrow['above_cutoff'], "bac_centered": bac_centered_narrow,
   "above_centered": in_band_narrow['above_cutoff'] * bac_centered_narrow}
)

X_narrow = sm.add_constant(pd.concat([rd_cols_narrow, in_band_narrow[CONTROLS]], axis=1))

rd_fit_narrow = sm.OLS(in_band_narrow['recidivism'], X_narrow).fit(cov_type='cluster', cov_kwds={'groups': in_band_narrow['bac_min']})
JUMP_narrow = float(rd_fit_narrow.params['above_cutoff'])

no_controls_narrow = sm.add_constant(rd_cols_narrow)
rd_fit_narrow_nocontrol = sm.OLS(in_band_narrow['recidivism'], no_controls_narrow).fit(
    cov_type='cluster', cov_kwds={'groups': in_band_narrow['bac_min']})
JUMP_narrow_nocontrol = float(rd_fit_narrow_nocontrol.params['above_cutoff'])

print(JUMP_narrow)
print(JUMP_narrow_nocontrol)
print(rd_fit_narrow.nobs)

# Hansen's higher cutoff, 0.15 also ran with no controls and at both bandwidths

CUTOFF2 = 0.15
# Changed filter to directly on low_score to avoid float error at 0.150.
# Also used hard comparison rather than >= as Hansen explicitly makes this a hard cutoff instead.
dwi['above_cutoff2'] = (dwi['low_score'] > 150).astype(int)


in_band2 = dwi[np.abs(dwi['low_score'] - 150) <= 50]
bac_centered2 = in_band2[RUNNING] - CUTOFF2

rd_cols2 = pd.DataFrame(
  {"above_cutoff2": in_band2['above_cutoff2'], "bac_centered2": bac_centered2,
   "above_centered2": in_band2['above_cutoff2'] * bac_centered2}
)
X2 = pd.concat([rd_cols2, in_band2[CONTROLS]], axis=1)
X2 = sm.add_constant(X2)

rd_fit2 = sm.OLS(in_band2['recidivism'], X2).fit(cov_type='cluster', cov_kwds={'groups': in_band2['bac_min']})
JUMP2 = float(rd_fit2.params['above_cutoff2'])

no_controls2 = sm.add_constant(rd_cols2)
rd_fit2_nocontrol = sm.OLS(in_band2['recidivism'], no_controls2).fit(cov_type='cluster', cov_kwds={'groups': in_band2['bac_min']})
JUMP2_nocontrol = float(rd_fit2_nocontrol.params['above_cutoff2'])

in_band2_narrow = dwi[np.abs(dwi['low_score'] - 150) <= 25]
bac_centered2_narrow = in_band2_narrow[RUNNING] - CUTOFF2

rd_cols2_narrow = pd.DataFrame(
  {"above_cutoff2": in_band2_narrow['above_cutoff2'], "bac_centered2": bac_centered2_narrow,
   "above_centered2": in_band2_narrow['above_cutoff2'] * bac_centered2_narrow}
)
X2_narrow = pd.concat([rd_cols2_narrow, in_band2_narrow[CONTROLS]], axis=1)
X2_narrow = sm.add_constant(X2_narrow)

rd_fit2_narrow = sm.OLS(in_band2_narrow['recidivism'], X2_narrow).fit(cov_type='cluster', cov_kwds={'groups': in_band2_narrow['bac_min']})
JUMP2_narrow = float(rd_fit2_narrow.params['above_cutoff2'])

no_controls2_narrow = sm.add_constant(rd_cols2_narrow)
rd_fit2_narrow_nocontrol = sm.OLS(in_band2_narrow['recidivism'], no_controls2_narrow).fit(cov_type='cluster', cov_kwds={'groups': in_band2_narrow['bac_min']})
JUMP2_narrow_nocontrol = float(rd_fit2_narrow_nocontrol.params['above_cutoff2'])

print(JUMP2, JUMP2_nocontrol, rd_fit2.nobs)
print(JUMP2_narrow, JUMP2_narrow_nocontrol, rd_fit2_narrow.nobs)

Inputs:

the data, passed in as an argument (dwi)
the cutoff and bandwidth, in integer thousandths of BAC (e.g. 80 and 50)
whether to include controls (my CONTROLS list, or none)
the standard error type: clustered by bac_min, or HC1
the treatment rule: at or above the cutoff, or strictly above
an optional list of years to exclude

Rules for the function:

Filter the window on low_score, including both edges: abs(low_score − cutoff) <= h.
Default treatment rule: the comparison should be a hard> at 150, and a >= at any other cutoff.
Use Rectangular kernel only: unweighted OLS within the window.
Use sm.OLS with my existing dummy columns, not formula syntax.
Drop any control column that is constant within the sample before fitting.
Don’t modify the input data; build the treatment and centered variables on a copy of the window.

The function should return the fitted result and the jump coefficient as an unrounded float. Additionally, write a check cell that calls the function with my all eight settings (differing bandwidths, and confirms it reproduces my estimates. Comment the lines to explain what the code is doing

Here's the function. One thing to flag up front: your 0.08 windows were filtered on bac_min in float space, where 0.13 - 0.08 evaluates to slightly more than 0.05, so the boundary rows at low_score 30, 130 (and 55, 105 for the narrow band) were silently dropped. Filtering on low_score as you asked fixes that, which means the four 0.08 regressions may come back with a few more observations and marginally different coefficients. The 0.15 runs should match exactly since you already filtered those on the integer score.

python
def rd_estimate(
    data,                   # the full dataset; never modified
    cutoff,                 # cutoff in integer thousandths of BAC, e.g. 80 or 150
    h,                      # bandwidth in the same integer units, e.g. 50 or 25
    controls=None,          # list of control column names, or None for no controls
    se="cluster",           # "cluster" (grouped by bac_min) or "HC1"
    rule=None,              # "ge", "gt", or None to use the default rule below
    exclude_years=None,     # optional iterable of years to drop before windowing
    outcome="recidivism",
    score="low_score",
):
    """Local linear RD at `cutoff` with a rectangular kernel of half-width `h`.

    Returns (fitted statsmodels result, jump coefficient as an unrounded float).
    """
    # Default treatment rule: Hansen treats 0.15 as a hard cutoff, so strictly
    # above there, at-or-above at every other cutoff.
    if rule is None:
        rule = "gt" if cutoff == 150 else "ge"
    if rule not in ("ge", "gt"):
        raise ValueError("rule must be 'ge', 'gt', or None")

    # Year exclusions apply before the window so the bandwidth isn't affected
    # by which rows the year filter removes.
    sample = data
    if exclude_years is not None:
        sample = sample[~sample["year"].isin(list(exclude_years))]

    # Window on the integer score, both edges included. Comparing integers keeps
    # the boundary rows that a float comparison on BAC drops.
    window = sample[np.abs(sample[score] - cutoff) <= h].copy()

    # Running variable in BAC units, then recentered so the constant is the
    # intercept at the cutoff and above_cutoff is the jump.
    window["bac_min"] = window[score] / 1000
    window["bac_centered"] = window["bac_min"] - cutoff / 1000

    # Treatment indicator, also built on the integer score to avoid float error
    # at 0.150, where the >= comparison on bac_min is unreliable.
    if rule == "gt":
        window["above_cutoff"] = (window[score] > cutoff).astype(int)
    else:
        window["above_cutoff"] = (window[score] >= cutoff).astype(int)

    # Interaction term: lets the slope differ on either side of the cutoff.
    window["above_centered"] = window["above_cutoff"] * window["bac_centered"]

    # Rectangular kernel = plain unweighted OLS on the rows inside the window.
    X = window[["above_cutoff", "bac_centered", "above_centered"]]

    if controls:
        # Drop controls that are constant in this sample (e.g. year dummies for
        # years that were excluded, or age dummies absent from a narrow window);
        # they're collinear with the constant and would make X singular.
        keep = [c for c in controls if window[c].nunique(dropna=False) > 1]
        X = pd.concat([X, window[keep]], axis=1)

    X = sm.add_constant(X)
    y = window[outcome]

    # Hansen clusters on the running variable; HC1 offered as a robustness check.
    if se == "cluster":
        fit = sm.OLS(y, X).fit(cov_type="cluster",
                               cov_kwds={"groups": window["bac_min"]})
    elif se == "HC1":
        fit = sm.OLS(y, X).fit(cov_type="HC1")
    else:
        raise ValueError("se must be 'cluster' or 'HC1'")

    return fit, float(fit.params["above_cutoff"])

Check cell:

python
# Each entry: label, kwargs for the function, and the estimate from the original cell.
checks = [
    ("0.08  h=.050  controls",    dict(cutoff=80,  h=50, controls=CONTROLS), JUMP),
    ("0.08  h=.050  no controls", dict(cutoff=80,  h=50, controls=None),     JUMP_nocontrol),
    ("0.08  h=.025  controls",    dict(cutoff=80,  h=25, controls=CONTROLS), JUMP_narrow),
    ("0.08  h=.025  no controls", dict(cutoff=80,  h=25, controls=None),     JUMP_narrow_nocontrol),
    ("0.15  h=.050  controls",    dict(cutoff=150, h=50, controls=CONTROLS), JUMP2),
    ("0.15  h=.050  no controls", dict(cutoff=150, h=50, controls=None),     JUMP2_nocontrol),
    ("0.15  h=.025  controls",    dict(cutoff=150, h=25, controls=CONTROLS), JUMP2_narrow),
    ("0.15  h=.025  no controls", dict(cutoff=150, h=25, controls=None),     JUMP2_narrow_nocontrol),
]

print(f"{'setting':<28}{'function':>12}{'original':>12}{'diff':>12}{'N':>9}  match")
for label, kwargs, original in checks:
    fit, jump = rd_estimate(dwi, **kwargs)          # defaults: cluster SEs, default rule
    diff = jump - original
    match = np.isclose(jump, original, rtol=0, atol=1e-10)  # exact to float precision
    print(f"{label:<28}{jump:>12.6f}{original:>12.6f}{diff:>12.2e}"
          f"{int(fit.nobs):>9}  {'yes' if match else 'NO'}")

# Confirm the input frame came back untouched.
assert "above_cutoff" not in rd_estimate(dwi, cutoff=80, h=50)[0].model.data.frame.columns \
    if False else True  # placeholder-free check below
print("\ndwi columns unchanged:", "bac_centered" not in dwi.columns and "above_centered" not in dwi.columns)

Drop that last assert line if it reads as clutter — the print underneath is the part that actually verifies the no-mutation rule.

A couple of notes on the other arguments, since the check cell only exercises the defaults:

se="HC1" and rule="ge"/rule="gt" let you test how sensitive the 0.15 estimate is to the hard-cutoff choice, which is worth a line in a robustness table.
exclude_years=[2007] (or whichever) drops those rows before the window is drawn, so the bandwidth is still a true ±h in score space.

Please make these changes:

A correction to your explanation: I tested the 0.08 windows, and abs(0.13 − 0.08) is exactly 0.05 in float64, so no boundary rows were dropped there. The boundary rows at 30, 130, 55, and 105 are all kept by the <= filter. The float error only affected the 0.15 window, at 0.200. No code change is needed for this, but keep filtering on low_score for consistency.
Rewrite the check cell to run in a separate notebook, where JUMP and the other estimate variables don’t exist. Instead, read the expected values from output/headline_table.csv:
Build the path the same way my notebooks do: ROOT / "output" / "headline_table.csv".
Read it with index_col=0, since the row labels are saved as the index.
Use the my_estimate column, matching each check to its row by label. The labels are:
Table 3, Panel A, col 1: BAC >= 0.08, h = 0.05
Table 3, Panel B, col 1: BAC >= 0.08, h = 0.025
Appendix Table 1, Panel A, col 1: BAC >= 0.08, h = 0.05, no controls
Appendix Table 1, Panel B, col 1: BAC >= 0.08, h = 0.025, no controls
Table 4, Panel A, col 1: BAC > 0.15, h = 0.05
Table 4, Panel B, col 1: BAC > 0.15, h = 0.025
Appendix Table 2, Panel A, col 1: BAC > 0.15, h = 0.05, no controls
Appendix Table 2, Panel B, col 1: BAC > 0.15, h = 0.025, no controls
The values are saved unrounded, so use atol=1e-10.
If the file doesn’t exist, raise a clear error saying to run notebook 01 or python reproduce.py first.
Delete the assert line, and replace the column check with a real one: save list(dwi.columns) before calling the function and confirm it’s identical afterward.
Fix two comments. On the default rule, say the statute defines aggravated DUI as BAC strictly above 0.15 (Hansen’s Table 1), not that Hansen treats it as a hard cutoff. On the window filter, say a float comparison can drop boundary rows, as it did at the 0.15 window, rather than implying it always does.
Before filtering on exclude_years, check that data["year"] is numeric, and raise a clear error if it isn’t.

Briefly explain each change.

Thanks for the correction — I had the float behavior wrong at 0.08. Here's the updated function and check cell.

python
def rd_estimate(
    data,                   # the full dataset; never modified
    cutoff,                 # cutoff in integer thousandths of BAC, e.g. 80 or 150
    h,                      # bandwidth in the same integer units, e.g. 50 or 25
    controls=None,          # list of control column names, or None for no controls
    se="cluster",           # "cluster" (grouped by bac_min) or "HC1"
    rule=None,              # "ge", "gt", or None to use the default rule below
    exclude_years=None,     # optional iterable of years to drop before windowing
    outcome="recidivism",
    score="low_score",
):
    """Local linear RD at `cutoff` with a rectangular kernel of half-width `h`.

    Returns (fitted statsmodels result, jump coefficient as an unrounded float).
    """
    # Default treatment rule: the statute defines aggravated DUI as a BAC
    # strictly above 0.15 (Hansen, Table 1), so strictly above at that cutoff
    # and at-or-above everywhere else.
    if rule is None:
        rule = "gt" if cutoff == 150 else "ge"
    if rule not in ("ge", "gt"):
        raise ValueError("rule must be 'ge', 'gt', or None")

    # Year exclusions apply before the window so the bandwidth isn't affected
    # by which rows the year filter removes.
    sample = data
    if exclude_years is not None:
        # isin() against numbers silently matches nothing if year is stored as
        # strings or categories, which would drop no rows and give a wrong
        # answer with no warning, so check the dtype first.
        if not pd.api.types.is_numeric_dtype(data["year"]):
            raise TypeError(
                f"exclude_years requires a numeric 'year' column, but it has "
                f"dtype {data['year'].dtype}. Convert it first, e.g. "
                f"data['year'] = pd.to_numeric(data['year'])."
            )
        sample = sample[~sample["year"].isin(list(exclude_years))]

    # Window on the integer score, both edges included. Comparing integers is
    # exact; a float comparison on BAC can drop boundary rows, as it does at the
    # 0.15 window, where 0.200 - 0.150 exceeds 0.05 in float64.
    window = sample[np.abs(sample[score] - cutoff) <= h].copy()

    # Running variable in BAC units, then recentered so the constant is the
    # intercept at the cutoff and above_cutoff is the jump.
    window["bac_min"] = window[score] / 1000
    window["bac_centered"] = window["bac_min"] - cutoff / 1000

    # Treatment indicator, also built on the integer score to avoid float error
    # at 0.150, where the >= comparison on bac_min is unreliable.
    if rule == "gt":
        window["above_cutoff"] = (window[score] > cutoff).astype(int)
    else:
        window["above_cutoff"] = (window[score] >= cutoff).astype(int)

    # Interaction term: lets the slope differ on either side of the cutoff.
    window["above_centered"] = window["above_cutoff"] * window["bac_centered"]

    # Rectangular kernel = plain unweighted OLS on the rows inside the window.
    X = window[["above_cutoff", "bac_centered", "above_centered"]]

    if controls:
        # Drop controls that are constant in this sample (e.g. year dummies for
        # years that were excluded, or age dummies absent from a narrow window);
        # they're collinear with the constant and would make X singular.
        keep = [c for c in controls if window[c].nunique(dropna=False) > 1]
        X = pd.concat([X, window[keep]], axis=1)

    X = sm.add_constant(X)
    y = window[outcome]

    # Hansen clusters on the running variable; HC1 offered as a robustness check.
    if se == "cluster":
        fit = sm.OLS(y, X).fit(cov_type="cluster",
                               cov_kwds={"groups": window["bac_min"]})
    elif se == "HC1":
        fit = sm.OLS(y, X).fit(cov_type="HC1")
    else:
        raise ValueError("se must be 'cluster' or 'HC1'")

    return fit, float(fit.params["above_cutoff"])

Check cell:

python
# Expected values come from the saved headline table, not from in-memory
# variables, so this cell runs in a fresh notebook.
table_path = ROOT / "output" / "headline_table.csv"
if not table_path.exists():
    raise FileNotFoundError(
        f"{table_path} not found. Run notebook 01 or `python reproduce.py` "
        f"to generate the headline table before running these checks."
    )

# Row labels were saved as the index, so read them back as the index.
headline = pd.read_csv(table_path, index_col=0)

# Each entry: row label in the table, and the kwargs that should reproduce it.
checks = [
    ("Table 3, Panel A, col 1: BAC >= 0.08, h = 0.05",
     dict(cutoff=80, h=50, controls=CONTROLS)),
    ("Table 3, Panel B, col 1: BAC >= 0.08, h = 0.025",
     dict(cutoff=80, h=25, controls=CONTROLS)),
    ("Appendix Table 1, Panel A, col 1: BAC >= 0.08, h = 0.05, no controls",
     dict(cutoff=80, h=50, controls=None)),
    ("Appendix Table 1, Panel B, col 1: BAC >= 0.08, h = 0.025, no controls",
     dict(cutoff=80, h=25, controls=None)),
    ("Table 4, Panel A, col 1: BAC > 0.15, h = 0.05",
     dict(cutoff=150, h=50, controls=CONTROLS)),
    ("Table 4, Panel B, col 1: BAC > 0.15, h = 0.025",
     dict(cutoff=150, h=25, controls=CONTROLS)),
    ("Appendix Table 2, Panel A, col 1: BAC > 0.15, h = 0.05, no controls",
     dict(cutoff=150, h=50, controls=None)),
    ("Appendix Table 2, Panel B, col 1: BAC > 0.15, h = 0.025, no controls",
     dict(cutoff=150, h=25, controls=None)),
]

# Snapshot the columns so we can confirm the function didn't touch the input.
cols_before = list(dwi.columns)

print(f"{'setting':<28}{'function':>12}{'saved':>12}{'diff':>12}{'N':>9}  match")
all_match = True
for label, kwargs in checks:
    expected = float(headline.loc[label, "my_estimate"])  # saved unrounded
    fit, jump = rd_estimate(dwi, **kwargs)                # defaults: cluster SEs, default rule
    diff = jump - expected
    match = np.isclose(jump, expected, rtol=0, atol=1e-10)
    all_match &= match
    short = label.split(": ")[1]                          # trim the table reference for width
    print(f"{short:<28}{jump:>12.6f}{expected:>12.6f}{diff:>12.2e}"
          f"{int(fit.nobs):>9}  {'yes' if match else 'NO'}")

cols_after = list(dwi.columns)
print(f"\nall eight reproduce: {all_match}")
print(f"dwi columns unchanged: {cols_before == cols_after}")

What changed:

Float explanation. Dropped the claim about boundary rows at 0.08; the window filter still uses low_score for consistency and exactness, with no code change.
Check cell. Expected values now load from ROOT / "output" / "headline_table.csv" with index_col=0, matched by row label to the my_estimate column, compared at atol=1e-10, with a FileNotFoundError pointing at notebook 01 or reproduce.py. The printed label is trimmed to the part after the colon so the table stays narrow; the full label is still what does the lookup.
Column check. The assert is gone. cols_before is captured before the loop and compared to cols_after once all eight calls have run.
Comments. The default-rule comment now cites the statutory definition of aggravated DUI as strictly above 0.15 (Hansen's Table 1), and the window comment says a float comparison can drop boundary rows, naming 0.15 as where it did.
Year dtype. exclude_years now raises a TypeError if data["year"] isn't numeric, since isin against integers would otherwise match nothing and silently drop no rows.

**What I changed, and why:**

- Widened the check cell's label column from 28 to 40 characters, so the printed table aligns.

### Exercise 1, inference (clustered versus HC1 standard errors)

- Model and version: Claude Opus 5
- Date: 11/27/26
**[Prep]** No explicit role assigned. Context came from the earlier part of the same chat: the verified `rd_estimate` function and the `checks` list of eight Section 2 specifications.

**[Request]** Starting from my manual command, which runs the headline specification with HC1 standard errors, write a loop over the eight specifications in `checks`, running each with clustered and HC1 standard errors. Build a table with the jump, both standard errors, their ratio, both p-values, and N; confirm the coefficient is identical under both; save the table to `output/robustness_inference.csv`; and print it to four decimals without overwriting existing variables.

**[Iterate]** One revision: I asked for four more columns, the 95% confidence interval under each standard error type, taken from `conf_int()`.

**[Mechanism Check]** I asked for line-by-line explanations. Two claims were worth checking. First, the AI said a ratio below 1, meaning clustering produces larger standard errors, is "the usual direction" with a coarse running variable. My results went the other way at 0.08 (ratios of 1.04 to 1.18), which matches Kolesár and Rothe's argument that clustering by running-variable values can understate uncertainty; I relied on my own results, not the AI's expectation. Second, it said both intervals use a normal critical value, so their widths are proportional to the standard errors. My output is consistent with that within rounding: the headline intervals' width ratio is about 1.05, matching the standard error ratio of 1.054.

**[Evaluate]** The coefficient check printed True for all eight specifications. The clustered standard errors match my Section 2 values (0.0041, 0.0042, 0.0048, 0.0049, 0.0034, 0.0034, 0.0046, 0.0046), and N matches each Section 2 window.

**Full transcript**

I am running a robustness check to compare Hansen’s clustered standard errors to heteroskedasticity-robust (HC1) ones.
Here’s my manual command, which runs the headline specification with HC1:

fit_hc1, jump_hc1 = rd_estimate(dwi, cutoff=80, h=50, controls=CONTROLS, se="HC1")

Using the latest rd_estimate function and the checks list from earlier in this chat, write a loop over the eight specifications in checks that runs each one twice, with se="cluster" and se="HC1", and builds a DataFrame with one row per specification and these columns:

the jump coefficient
the clustered SE and the HC1 SE
the ratio of HC1 SE to clustered SE
the p-value under each SE type
N

Also confirm the coefficient is identical under both SE types, since only the standard errors should change.
Save the table to ROOT / "output" / "robustness_inference.csv", print it formatted to four decimals, and don’t overwrite any existing variables. Explain each line.

python
# Fresh names throughout so nothing from the earlier cells is overwritten.
inference_rows = []          # one dict per specification, assembled into the table below
coef_mismatches = []         # labels where the two coefficients aren't bit-identical

for spec_label, spec_kwargs in checks:
    # Same specification, fit twice; `se` is passed separately so the dicts
    # inside `checks` are never mutated.
    fit_cl, jump_cl = rd_estimate(dwi, se="cluster", **spec_kwargs)
    fit_rb, jump_rb = rd_estimate(dwi, se="HC1", **spec_kwargs)

    # Only the covariance estimator differs, so the point estimates come from
    # the same OLS solve and should be exactly equal, not merely close.
    if jump_cl != jump_rb:
        coef_mismatches.append(spec_label)

    # Pull the SE and p-value for the jump coefficient out of each fit.
    se_cluster = float(fit_cl.bse["above_cutoff"])
    se_hc1 = float(fit_rb.bse["above_cutoff"])

    inference_rows.append({
        "label": spec_label,
        "coef": jump_cl,                       # identical under both, checked above
        "se_cluster": se_cluster,
        "se_hc1": se_hc1,
        "se_ratio": se_hc1 / se_cluster,       # >1 means clustering tightens the SE
        "p_cluster": float(fit_cl.pvalues["above_cutoff"]),
        "p_hc1": float(fit_rb.pvalues["above_cutoff"]),
        "N": int(fit_cl.nobs),                 # window is the same, so N matches too
    })

# Row labels as the index, matching how the headline table is saved.
inference_df = pd.DataFrame(inference_rows).set_index("label")

# Report the coefficient check before anything else, since a mismatch would
# mean the two runs weren't actually the same specification.
if coef_mismatches:
    raise AssertionError(
        "Coefficient changed with the SE type in: " + "; ".join(coef_mismatches)
    )
print("coefficients identical under cluster and HC1: True\n")

# Write the table out alongside the other saved output.
inference_path = ROOT / "output" / "robustness_inference.csv"
inference_path.parent.mkdir(parents=True, exist_ok=True)   # no-op if output/ exists
inference_df.to_csv(inference_path)                        # full precision on disk
print(f"saved to {inference_path}\n")

# Print rounded to four decimals; N stays an integer because float_format
# only applies to float columns.
print(inference_df.to_string(float_format=lambda v: f"{v:.4f}"))

Line by line:

inference_rows / coef_mismatches collect results and failures as the loop runs, rather than building the DataFrame incrementally.
The loop unpacks checks into its label and kwargs, then calls rd_estimate twice with se passed as a separate keyword. Passing it separately rather than editing spec_kwargs keeps checks reusable for the reproduction cell.
The equality test uses != rather than np.isclose. cov_type only affects the covariance matrix, not the coefficient vector, so the two values come from the same solve and should match to the last bit — an approximate test here would hide a real bug.
fit.bse and fit.pvalues are indexed by parameter name, so ["above_cutoff"] picks out the jump in both.
se_ratio is HC1 over clustered, so a value below 1 means clustering on bac_min widens the interval relative to HC1 — the usual direction when the running variable is coarse and there's within-cell correlation.
N comes from the clustered fit, but either works: both runs use the same window.
The mismatch check raises before the file is written, so a bad run can't leave a stale CSV behind.
mkdir(parents=True, exist_ok=True) makes the cell safe to run in a fresh clone where output/ hasn't been created yet.
to_csv saves unrounded values, matching how headline_table.csv stores estimates, so a later check can compare at atol=1e-10. The rounding is display-only, via float_format in to_string.

One note on reading the result: the p-values will differ between the two columns, sometimes enough to cross conventional thresholds at the narrow bandwidths where cluster counts are small. With 51 or 101 mass points in the running variable, the clustered SE is at the low end of where that estimator is reliable, so treat a large se_ratio swing as information about the estimator rather than about the effect.

Can you add in 4 more columns to the loop:  the 95% confidence interval for the jump under each SE type, as ci_low_cluster, ci_high_cluster, ci_low_hc1, and ci_high_hc1, taken from each fit’s conf_int() for above_cutoff.

python
# Fresh names throughout so nothing from the earlier cells is overwritten.
inference_rows = []          # one dict per specification, assembled into the table below
coef_mismatches = []         # labels where the two coefficients aren't bit-identical

for spec_label, spec_kwargs in checks:
    # Same specification, fit twice; `se` is passed separately so the dicts
    # inside `checks` are never mutated.
    fit_cl, jump_cl = rd_estimate(dwi, se="cluster", **spec_kwargs)
    fit_rb, jump_rb = rd_estimate(dwi, se="HC1", **spec_kwargs)

    # Only the covariance estimator differs, so the point estimates come from
    # the same OLS solve and should be exactly equal, not merely close.
    if jump_cl != jump_rb:
        coef_mismatches.append(spec_label)

    # Pull the SE and p-value for the jump coefficient out of each fit.
    se_cluster = float(fit_cl.bse["above_cutoff"])
    se_hc1 = float(fit_rb.bse["above_cutoff"])

    # conf_int() returns a frame indexed by parameter with columns 0 (lower)
    # and 1 (upper); default alpha=0.05 gives the 95% interval. Each fit builds
    # its interval from its own covariance estimator, so the two differ in width
    # but are centered on the same coefficient.
    ci_cl = fit_cl.conf_int().loc["above_cutoff"]
    ci_rb = fit_rb.conf_int().loc["above_cutoff"]

    inference_rows.append({
        "label": spec_label,
        "coef": jump_cl,                       # identical under both, checked above
        "se_cluster": se_cluster,
        "se_hc1": se_hc1,
        "se_ratio": se_hc1 / se_cluster,       # >1 means clustering tightens the SE
        "p_cluster": float(fit_cl.pvalues["above_cutoff"]),
        "p_hc1": float(fit_rb.pvalues["above_cutoff"]),
        "ci_low_cluster": float(ci_cl[0]),
        "ci_high_cluster": float(ci_cl[1]),
        "ci_low_hc1": float(ci_rb[0]),
        "ci_high_hc1": float(ci_rb[1]),
        "N": int(fit_cl.nobs),                 # window is the same, so N matches too
    })

# Row labels as the index, matching how the headline table is saved.
inference_df = pd.DataFrame(inference_rows).set_index("label")

# Report the coefficient check before anything else, since a mismatch would
# mean the two runs weren't actually the same specification.
if coef_mismatches:
    raise AssertionError(
        "Coefficient changed with the SE type in: " + "; ".join(coef_mismatches)
    )
print("coefficients identical under cluster and HC1: True\n")

# Write the table out alongside the other saved output.
inference_path = ROOT / "output" / "robustness_inference.csv"
inference_path.parent.mkdir(parents=True, exist_ok=True)   # no-op if output/ exists
inference_df.to_csv(inference_path)                        # full precision on disk
print(f"saved to {inference_path}\n")

# Print rounded to four decimals; N stays an integer because float_format
# only applies to float columns.
print(inference_df.to_string(float_format=lambda v: f"{v:.4f}"))

The CI columns sit after the p-values, so the cluster and HC1 blocks each read SE → p → interval. The full table is saved unrounded as before.

Two practical notes:

Twelve columns against those long row labels will wrap in most notebook outputs. If that's annoying, print(inference_df[["coef", "ci_low_cluster", "ci_high_cluster", "ci_low_hc1", "ci_high_hc1"]].to_string(float_format=...)) gives a readable interval-only view without changing what gets saved.
conf_int() uses a normal critical value when cov_type is anything other than the default nonrobust, so both intervals are coefficient ± 1.96 × SE. That means ci_high − ci_low is exactly proportional to the SE, and the interval ratio will reproduce se_ratio — the intervals add no independent information, but they're the form most readers want in a table.
**What I changed, and why:**

- No changes to the generated code


### Exercise 2, placebo cutoffs

- Model and version: Claude Opus 5
- Date: 11/22/26

**[Prep]** No explicit role assigned. Context came from earlier in the same chat, including the verified `rd_estimate` function.

**[Request]** Starting from my manual command, a placebo at 0.050, write a loop running the same specification (bandwidth 0.025, controls, clustered standard errors) at placebo cutoffs of 0.040, 0.050, 0.110, 0.115, 0.120, and 0.180, plus the real cutoffs at 0.08 and 0.15 for comparison. Before fitting each placebo, raise an error if its window contains a real cutoff. Build a table with the cutoff, whether it's real or a placebo, the jump, its standard error, p-value, and 95% interval, and N; save it to `output/robustness_placebo.csv`; and display it with set decimal places.

**[Iterate]** None beyond the initial prompt.

**[Mechanism Check]** I asked for line-by-line explanations. Two small claims didn't quite hold. It said the 0.110 window, 0.085 to 0.135, is the tightest fit against the real cutoffs, but the 0.050 window, 0.025 to 0.075, is equally close: both edges are 0.005 from 0.08. It also referred to "eight cutoffs tested at the 5% level" when discussing chance rejections, but only six are placebos. Neither affects the code. Its warning that the low-BAC windows would have small samples and wide intervals was borne out: 9,727 and 16,417 stops.

**[Evaluate]** The window check raised no errors. The real-cutoff rows (-0.0193 at 0.08 and -0.0099 at 0.15) match my Section 2 narrow-bandwidth estimates and Exercise 1, with matching N (50,016 and 75,909).

**Full transcript**

My second robustness check is for the placebo effect, to see if BAC cutoffs that are not legal thresholds cause a jump.
Here’s the first call,  for a placebo at 0.050:

fit_p50, jump_p50 = rd_estimate(dwi, cutoff=50, h=25, controls=CONTROLS)

Using the latest rd_estimate function from earlier in this chat, write a loop that runs the same specification (bandwidth 25, with CONTROLS, clustered SEs) at these cutoffs, in thousandths: placebos at 40, 50, 110, 115, 120, and 180, plus the real cutoffs at 80 and 150 for comparison.
Before fitting each placebo, check that its window (cutoff ± 25) doesn’t contain 80 or 150, and raise an error if it does.
Build a DataFrame with one row per cutoff and these columns:

the cutoff (in BAC units) and whether it’s real or a placebo
the jump coefficient, its SE, and its p-value
the 95% confidence interval, from conf_int()
N

Save the full-precision table to ROOT / "output" / "robustness_placebo.csv". Then display a readable version, with coefficients, SEs, and intervals to four decimals and p-values to three. Don’t overwrite any existing variables. Explain each line.

python
# All new names so nothing from the earlier cells (including fit_p50) is touched.
PLACEBO_H = 25                       # same bandwidth for every row, in thousandths
REAL_CUTOFFS = (80, 150)             # the two statutory thresholds
placebo_cutoffs = (40, 50, 110, 115, 120, 180)
placebo_rows = []                    # one dict per cutoff, assembled into the table below

# Run the real cutoffs alongside the placebos so the table is self-contained;
# sorted so the rows read left to right along the BAC scale.
placebo_specs = sorted(
    [(c, "placebo") for c in placebo_cutoffs] + [(c, "real") for c in REAL_CUTOFFS]
)

for cut, kind in placebo_specs:
    # A placebo window that straddles a real threshold would pick up the actual
    # treatment effect leaking in, so the row would be meaningless. Check the
    # window bounds before fitting, and only for the placebos.
    if kind == "placebo":
        contaminating = [r for r in REAL_CUTOFFS if abs(r - cut) <= PLACEBO_H]
        if contaminating:
            raise ValueError(
                f"Placebo cutoff {cut} with h={PLACEBO_H} spans the window "
                f"[{cut - PLACEBO_H}, {cut + PLACEBO_H}], which contains the real "
                f"cutoff(s) {contaminating}. Narrow the bandwidth or drop this cutoff."
            )

    # Identical specification throughout: h=25, CONTROLS, clustered SEs. The
    # treatment rule falls out of the function's default, so the placebos use
    # >= and the 0.15 row uses >, as in the headline table.
    fit_pl, jump_pl = rd_estimate(dwi, cutoff=cut, h=PLACEBO_H, controls=CONTROLS)

    # 95% interval for the jump, built from this fit's clustered covariance.
    ci_pl = fit_pl.conf_int().loc["above_cutoff"]

    placebo_rows.append({
        "cutoff_bac": cut / 1000,                          # report in BAC units
        "kind": kind,                                      # "real" or "placebo"
        "coef": jump_pl,
        "se": float(fit_pl.bse["above_cutoff"]),
        "p_value": float(fit_pl.pvalues["above_cutoff"]),
        "ci_low": float(ci_pl[0]),
        "ci_high": float(ci_pl[1]),
        "N": int(fit_pl.nobs),
    })

# Index on the cutoff so rows are identified by the thing that varies.
placebo_df = pd.DataFrame(placebo_rows).set_index("cutoff_bac")

# Save unrounded, matching how the other output tables are written.
placebo_path = ROOT / "output" / "robustness_placebo.csv"
placebo_path.parent.mkdir(parents=True, exist_ok=True)     # no-op if output/ exists
placebo_df.to_csv(placebo_path)
print(f"saved to {placebo_path}\n")

# Display only: four decimals for the estimate columns, three for p-values.
# Per-column formatters, since to_string's float_format is applied globally.
placebo_formatters = {
    "coef":    lambda v: f"{v:.4f}",
    "se":      lambda v: f"{v:.4f}",
    "ci_low":  lambda v: f"{v:.4f}",
    "ci_high": lambda v: f"{v:.4f}",
    "p_value": lambda v: f"{v:.3f}",
}
print(placebo_df.to_string(formatters=placebo_formatters))

Line by line:

PLACEBO_H and REAL_CUTOFFS are named rather than inlined so the contamination check and the fits can't drift apart if you change the bandwidth.
placebo_specs merges the two groups and sorts by cutoff, which puts the real thresholds in place among the placebos instead of appended at the end. Reading down the column, a credible result is a visible jump at 0.08 and 0.15 and nothing elsewhere.
The contamination check compares each placebo to both real cutoffs and raises with the window bounds and the offending cutoff named. All six placebos clear it at h=25 — 110 is the tightest, with a window of [85, 135] — so this is a guard for when you change the bandwidth, not something that fires now.
The check is skipped for the real rows, which would otherwise fail against themselves.
rd_estimate is called with no se argument so it uses its clustered default, and no rule, so 150 gets > and everything else >=.
conf_int().loc["above_cutoff"] pulls the lower and upper bounds as columns 0 and 1, same as in the inference table.
cutoff_bac divides by 1000 for display; the function still works in integer thousandths internally.
to_csv writes full precision; the rounding lives entirely in formatters, which takes a per-column dict so p-values can use three decimals while the rest use four.

Two things to watch when you read the output:

The 0.04 and 0.05 windows reach down to a low_score of 15 and 25. If your sample has few arrests at very low BAC, those rows will have small N and wide intervals, and a null there is weak evidence rather than a clean pass. The N column will tell you which rows to discount.
With eight cutoffs tested at the 5% level, roughly one spurious rejection is unremarkable. The pattern across rows is the evidence, not any single p-value — a lone significant placebo at 0.115 with the real cutoffs also significant is a different story from placebos significant and 0.08 not.
**What I changed, and why:**

- No changes to the generated code

### Exercise 3, the 0.15 treatment rule

- Model and version: Claude Opus 5
- Date: 11/22/26

**[Prep]** No explicit role assigned. Context came from earlier in the same chat, including the verified `rd_estimate` function and the saved headline table.

**[Request]** Starting from my manual command, the headline 0.15 specification with 0.150 counted as treated, write a loop over the four 0.15 specifications under both treatment rules. Build a table with the rule, the jump, its standard error, p-value, and 95% interval, N, Hansen's estimate from the headline table, and the gap from it. Confirm the strict-rule rows reproduce my saved estimates exactly, confirm N is identical under both rules, print the number of stops at exactly 0.150, and save the table to `output/robustness_rule015.csv`.

**[Iterate]** None beyond the initial prompt.

**[Mechanism Check]** I asked for comments on each line. Two of its explanations were worth noting. It said the N check is guaranteed to pass, since the function builds the window before applying the rule; that's correct from the function's structure, so the check guards against future code changes rather than testing the data. It also described a positive gap as meaning "my estimate is larger." Because the estimates are negative, a positive gap actually means mine is smaller in magnitude than Hansen's, which is how I report it.

**[Evaluate]** The strict-rule rows reproduced my headline table exactly, N was identical under both rules, and 1,527 stops sit at exactly 0.150. The strict-rule estimates match my Section 2 values.

**Full transcript**

The third robustness check is testing how the 0.150 estimate depends on the coding choice of whether a BAC of exactly 0.150 counts as aggravated DUI. Hansen’s Table 1 defines aggravated as strictly above 0.15 (rule="gt", my replication estimate), but his text uses “above” loosely, and >= coding would treat 0.150 as aggravated (rule="ge").

This is the manual function call: 

fit_ge, jump_ge = rd_estimate(dwi, cutoff=150, h=50, controls=CONTROLS, rule='ge')

Using the latest rd_estimate function and the headline table at ROOT / "output" / "headline_table.csv" from earlier in this chat, write a loop over the four 0.15 specifications (bandwidth 50 and 25, with CONTROLS and without), running each under both rule="gt" and rule="ge". Build a DataFrame with one row per specification and rule, and these columns:

the rule
the jump coefficient, its SE, its p-value, and its 95% confidence interval from conf_int()
N
Hansen’s estimate for that specification, from the paper_estimate column of the headline table, matched by label
the gap from Hansen’s estimate

Also:

Confirm the rule="gt" rows reproduce my my_estimate values in the headline table exactly.
Confirm N is the same under both rules, since only the treatment coding changes, not the window.
Print how many stops have low_score == 150, the number whose treatment status changes.

Save the full-precision table to ROOT / "output" / "robustness_rule015.csv". Then display a readable version, with coefficients, SEs, intervals, and gaps to four decimals, and p-values to three. Don’t overwrite any existing variables. Comment each line explaining the code.

python
# All new names so nothing from the earlier cells (including fit_ge) is touched.
rule015_path_in = ROOT / "output" / "headline_table.csv"
if not rule015_path_in.exists():
    raise FileNotFoundError(
        f"{rule015_path_in} not found. Run notebook 01 or `python reproduce.py` "
        f"to generate the headline table before running this check."
    )

# Row labels were saved as the index, same as in the reproduction cell.
rule015_headline = pd.read_csv(rule015_path_in, index_col=0)

# The four 0.15 specifications: each label paired with the kwargs that produce it.
# The labels read "BAC > 0.15" because the headline table was built with rule="gt";
# the "ge" rows below deliberately depart from that coding.
rule015_specs = [
    ("Table 4, Panel A, col 1: BAC > 0.15, h = 0.05",
     dict(cutoff=150, h=50, controls=CONTROLS)),
    ("Table 4, Panel B, col 1: BAC > 0.15, h = 0.025",
     dict(cutoff=150, h=25, controls=CONTROLS)),
    ("Appendix Table 2, Panel A, col 1: BAC > 0.15, h = 0.05, no controls",
     dict(cutoff=150, h=50, controls=None)),
    ("Appendix Table 2, Panel B, col 1: BAC > 0.15, h = 0.025, no controls",
     dict(cutoff=150, h=25, controls=None)),
]

rule015_rows = []        # one dict per (specification, rule) pair
rule015_repro_fail = []  # labels where the "gt" run misses the saved my_estimate
rule015_n_fail = []      # labels where N differs between the two rules

for spec_label, spec_kwargs in rule015_specs:
    # Hansen's own number and my saved replication number for this specification.
    paper_est = float(rule015_headline.loc[spec_label, "paper_estimate"])
    mine_est = float(rule015_headline.loc[spec_label, "my_estimate"])

    n_by_rule = {}  # collect N per rule so the two can be compared after the inner loop

    for rule_name in ("gt", "ge"):
        # Same window and same controls both times; only the treatment coding
        # at low_score == 150 changes. `rule` is passed separately so the dicts
        # in rule015_specs stay reusable.
        fit_r, jump_r = rd_estimate(dwi, rule=rule_name, **spec_kwargs)

        # The "gt" run is the headline coding, so it must land on my saved
        # estimate exactly; a miss means the function or the data has drifted.
        if rule_name == "gt" and not np.isclose(jump_r, mine_est, rtol=0, atol=1e-10):
            rule015_repro_fail.append(spec_label)

        n_by_rule[rule_name] = int(fit_r.nobs)

        # 95% interval for the jump, from this fit's clustered covariance.
        ci_r = fit_r.conf_int().loc["above_cutoff"]

        rule015_rows.append({
            "label": spec_label,
            "rule": rule_name,
            "coef": jump_r,
            "se": float(fit_r.bse["above_cutoff"]),
            "p_value": float(fit_r.pvalues["above_cutoff"]),
            "ci_low": float(ci_r[0]),
            "ci_high": float(ci_r[1]),
            "N": int(fit_r.nobs),
            "paper_estimate": paper_est,
            "gap_from_paper": jump_r - paper_est,   # signed: positive means my estimate is larger
        })

    # The window is defined by low_score alone, so recoding treatment can't
    # change which rows are in the sample. A difference here would mean the
    # rule is leaking into the window filter.
    if n_by_rule["gt"] != n_by_rule["ge"]:
        rule015_n_fail.append(spec_label)

# Index on the specification and the rule, so each row is uniquely identified.
rule015_df = pd.DataFrame(rule015_rows).set_index(["label", "rule"])

# Raise on either failure before writing, so a bad run leaves no stale CSV.
if rule015_repro_fail:
    raise AssertionError(
        'rule="gt" did not reproduce my_estimate in: ' + "; ".join(rule015_repro_fail)
    )
if rule015_n_fail:
    raise AssertionError(
        "N differed between rules in: " + "; ".join(rule015_n_fail)
    )
print('rule="gt" reproduces my_estimate in all four specifications: True')
print("N identical under both rules in all four specifications: True")

# The stops whose treatment status actually flips between the two codings.
# If this is zero, the two rules are the same regression and every pair of
# rows below will be identical.
rule015_n_at_150 = int((dwi["low_score"] == 150).sum())
print(f"stops with low_score == 150 (treatment flips under 'ge'): {rule015_n_at_150}\n")

# Save unrounded, matching the other output tables.
rule015_path_out = ROOT / "output" / "robustness_rule015.csv"
rule015_path_out.parent.mkdir(parents=True, exist_ok=True)   # no-op if output/ exists
rule015_df.to_csv(rule015_path_out)
print(f"saved to {rule015_path_out}\n")

# Display only: four decimals for the estimate columns, three for p-values.
# Per-column formatters, since to_string's float_format applies to every float.
rule015_formatters = {
    "coef":           lambda v: f"{v:.4f}",
    "se":             lambda v: f"{v:.4f}",
    "ci_low":         lambda v: f"{v:.4f}",
    "ci_high":        lambda v: f"{v:.4f}",
    "gap_from_paper": lambda v: f"{v:.4f}",
    "paper_estimate": lambda v: f"{v:.4f}",
    "p_value":        lambda v: f"{v:.3f}",
}
print(rule015_df.to_string(formatters=rule015_formatters))

A few notes on the design and on reading the output:

The N check is guaranteed to pass as written, since rd_estimate builds the window from low_score before it ever looks at rule. It's worth keeping as a regression test — if someone later moves the treatment construction into the filter, this is what catches it — but it isn't evidence about the data.
The count of stops at exactly low_score == 150 is the whole story for this check. Those are the only rows that change, and they sit exactly at the boundary where the local linear fit is most leveraged, so a handful of them can move the coefficient more than their share of the sample suggests. If the count comes back as zero, the ge and gt rows will be identical and the ambiguity in Hansen's text is moot.
gap_from_paper is signed as mine minus Hansen's, so positive means my estimate is larger. The gt gaps are already in your headline table; what this adds is whether the ge gaps are materially worse, which is the argument that Table 1's definition is the right reading.
The index is a two-level (label, rule) MultiIndex, so rule015_df.xs("ge", level="rule") pulls just the alternative coding, and the CSV round-trips with index_col=[0, 1].
**What I changed, and why:**

- No changes to the generated code


### Exercise 4, re-estimation with `rdrobust`

- Model and version: Claude Opus 5
- Date: 11/22/26

**[Prep]** No explicit role assigned. I supplied the GIVEN `rd()` and `show()` functions from the course starter notebook, so the loop could match their settings.

**[Request]** Starting from my manual command, `rd()` at 0.08 with controls on stops with BAC at or below 0.150, write a loop over both cutoffs with and without controls, calling `rdrobust` directly with `rd()`'s settings. At the aggravated threshold, use a cutoff of 0.1505, so a BAC of 0.150 is untreated as the statute requires, and restrict each cutoff's data so its window can't cross the other threshold. Collect the conventional, bias-corrected, and robust results, the bandwidths, and N; save to `output/robustness_rdrobust.csv`; and display with set decimal places. My prompt repeated one paragraph by mistake, and it didn't list the columns or ask for the comparison with my Section 2 estimates, so the generated table doesn't include that comparison.

**[Iterate]** None beyond the initial cell.

**[Mechanism Check]** I asked for comments on each line. Two of its claims: first, that `rdrobust` can return `NaN` rather than an error when a covariate is collinear, which is why it added a filter for constant controls. I didn't verify this; I kept the filter, since it does no harm, and the match check below confirms it didn't change the result. Second, that a large gap between the uncorrected and bias-corrected estimates would signal curvature. I checked: the gap is about 0.001 at both cutoffs, so the linear fit is adequate at the chosen bandwidth.

**[Evaluate]** The loop's 0.08-with-controls row matches my manual `rd()` call exactly (-0.0168, SE 0.0062, robust p = 0.034, n = 48,199). Without controls, N at 0.08 is 50,016, matching my Section 2 narrow window.

**Full transcript**

The other robustness check re-estimates Hansen's RD with rdrobust which the 2015 paper does not use. rdrobust is already imported in the notebook. The following rd() and show() functions are used in the manual function call:

def rd(cutoff, h=None, kernel="triangular", controls=None, data=None, outcome="recidivism", donut=0.0):
    data = dwi if data is None else data
    d = data[(data[RUNNING] - cutoff).abs() >= donut]
    covs = d[controls] if controls else None
    kw = {"h": h} if h else {}
    r = rdrobust(y=d[outcome], x=d[RUNNING], c=cutoff, p=1, kernel=kernel, covs=covs, **kw)
    return {"jump": r.coef.iloc[0, 0], "se": r.se.iloc[0, 0], "p_robust": r.pv.iloc[2, 0],
            "h": r.bws.iloc[0, 0], "n": int(r.N_h[0] + r.N_h[1])}


def show(label, r):
    print(f"{label:<34} {r['jump']:+.4f} (SE {r['se']:.4f}, robust p {r['p_robust']:.3f}), h = {r['h']:.3f}, n = {r['n']:,}")

The following is the manual function call for one specification:

r08 = rd(0.08, controls=CONTROLS, data=dwi[dwi['low_score'] <=150])
show("0.08, MSE-optimal h, controls", r08)

Write a loop over both cutoffs, with and without CONTROLS:

At 0.08, use only stops with low_score <= 150. At the aggravated threshold, use a cutoff of 0.1505, so a BAC of 0.150 falls on the untreated side as the statute requires, and use only stops with low_score >= 80. These restrictions keep each window from crossing the other threshold.
Call rdrobust directly, with the same settings as rd() (p=1, triangular kernel, covs from the controls, RUNNING as the running variable), so you can collect more of its output than rd() returns.

Write a loop over both cutoffs, with and without CONTROLS:

At 0.08, use only stops with low_score <= 150. At the aggravated threshold, use a cutoff of 0.1505, so a BAC of 0.150 falls on the untreated side as the statute requires, and use only stops with low_score >= 80. These restrictions keep each window from crossing the other threshold.
Call rdrobust directly, with the same settings as rd() (p=1, triangular kernel, covs from the controls, RUNNING as the running variable), so you can collect more of its output than rd() returns.

rdrobust stores its results with rows for conventional, bias-corrected, and robust: use coef, se, ci, and pv accordingly, and bws and N_h for the bandwidth and N.
Save the full-precision table to ROOT / "output" / "robustness_rdrobust.csv", then display a readable version, with estimates and intervals to four decimals, p-values to three, and bandwidths to three. Don’t overwrite any existing variables. Comment each line to explain the code.

python
# All new names so nothing from the earlier cells (r08, rd, show) is touched.
rdr_rows = []   # one dict per (cutoff, controls) combination

# Each entry: label, the cutoff rdrobust centers on, the row restriction that
# keeps this window from reaching the other threshold, and the controls.
#   - 0.08 runs on low_score <= 150, so the window can't cross the aggravated cutoff.
#   - the aggravated threshold is centered at 0.1505 rather than 0.150, which puts
#     a BAC of exactly 0.150 on the untreated side as the statute requires, and
#     runs on low_score >= 80 so the window can't reach back to 0.08.
rdr_specs = [
    ("0.08, controls",        0.08,   dwi["low_score"] <= 150, CONTROLS),
    ("0.08, no controls",     0.08,   dwi["low_score"] <= 150, None),
    ("0.1505, controls",      0.1505, dwi["low_score"] >= 80,  CONTROLS),
    ("0.1505, no controls",   0.1505, dwi["low_score"] >= 80,  None),
]

for rdr_label, rdr_cut, rdr_mask, rdr_controls in rdr_specs:
    # Apply the row restriction; .copy() so nothing downstream touches dwi.
    d = dwi[rdr_mask].copy()

    # Covariates as a frame, or None when running without controls. Constant
    # columns are dropped for the same reason as in rd_estimate: a year or age
    # dummy with no variation in this subsample is collinear with the intercept.
    if rdr_controls:
        covs = d[[c for c in rdr_controls if d[c].nunique(dropna=False) > 1]]
    else:
        covs = None

    # Same settings as rd(): local linear, triangular kernel, MSE-optimal
    # bandwidth chosen by rdrobust itself (no h passed), RUNNING as the score.
    # Called directly rather than through rd() so the full result object is
    # available, not just the five fields rd() returns.
    r = rdrobust(y=d["recidivism"], x=d[RUNNING], c=rdr_cut, p=1,
                 kernel="triangular", covs=covs)

    # rdrobust stacks three rows of inference: 0 conventional, 1 bias-corrected,
    # 2 robust. The point estimate is the conventional one; the robust row
    # carries the bias-corrected estimate with its wider, bias-aware interval,
    # and is the row rdrobust intends you to report inference from.
    rdr_rows.append({
        "label": rdr_label,
        "cutoff": rdr_cut,
        "controls": rdr_controls is not None,
        "coef": float(r.coef.iloc[0, 0]),            # conventional point estimate
        "se": float(r.se.iloc[0, 0]),                # conventional SE
        "p_conventional": float(r.pv.iloc[0, 0]),
        "ci_low": float(r.ci.iloc[0, 0]),            # conventional 95% interval
        "ci_high": float(r.ci.iloc[0, 1]),
        "coef_bc": float(r.coef.iloc[1, 0]),         # bias-corrected estimate
        "se_robust": float(r.se.iloc[2, 0]),         # robust SE
        "p_robust": float(r.pv.iloc[2, 0]),          # the p-value rd() reports
        "ci_low_robust": float(r.ci.iloc[2, 0]),     # robust 95% interval
        "ci_high_robust": float(r.ci.iloc[2, 1]),
        # bws row 0 is the main bandwidth h, row 1 the bias bandwidth b; the two
        # columns are the left and right sides, which can differ.
        "h_left": float(r.bws.iloc[0, 0]),
        "h_right": float(r.bws.iloc[0, 1]),
        "b_left": float(r.bws.iloc[1, 0]),
        "b_right": float(r.bws.iloc[1, 1]),
        # N_h counts effective observations inside h on each side.
        "n_left": int(r.N_h[0]),
        "n_right": int(r.N_h[1]),
        "N": int(r.N_h[0] + r.N_h[1]),
    })

# Index on the label so each row is identified by its specification.
rdr_df = pd.DataFrame(rdr_rows).set_index("label")

# Save unrounded, matching the other output tables.
rdr_path = ROOT / "output" / "robustness_rdrobust.csv"
rdr_path.parent.mkdir(parents=True, exist_ok=True)   # no-op if output/ exists
rdr_df.to_csv(rdr_path)
print(f"saved to {rdr_path}\n")

# Display only: four decimals for estimates and intervals, three for p-values
# and bandwidths. Per-column formatters, since to_string's float_format is global.
rdr_formatters = {
    "coef":           lambda v: f"{v:+.4f}",
    "se":             lambda v: f"{v:.4f}",
    "ci_low":         lambda v: f"{v:+.4f}",
    "ci_high":        lambda v: f"{v:+.4f}",
    "coef_bc":        lambda v: f"{v:+.4f}",
    "se_robust":      lambda v: f"{v:.4f}",
    "ci_low_robust":  lambda v: f"{v:+.4f}",
    "ci_high_robust": lambda v: f"{v:+.4f}",
    "p_conventional": lambda v: f"{v:.3f}",
    "p_robust":       lambda v: f"{v:.3f}",
    "h_left":         lambda v: f"{v:.3f}",
    "h_right":        lambda v: f"{v:.3f}",
    "b_left":         lambda v: f"{v:.3f}",
    "b_right":        lambda v: f"{v:.3f}",
    "cutoff":         lambda v: f"{v:.4f}",
}
print(rdr_df.to_string(formatters=rdr_formatters))

Three things worth checking in the output:

The constant-column drop isn't in rd(), but the row restrictions here can leave a year or age dummy with no variation, and rdrobust doesn't always fail loudly on a collinear covariate — it can return NaN instead. Dropping them keeps the covariate set honest. If you'd rather stay byte-identical to rd(), remove the filter and watch for NaN coefficients.
Compare h_left/h_right against your 0.05 and 0.025 OLS bandwidths. rdrobust picks its own MSE-optimal bandwidth, so if it lands near 0.05 your headline choice is defensible on its own terms; if it lands much narrower, the OLS estimate is averaging over a wider window than the data-driven rule would.
coef and coef_bc won't match. The gap is rdrobust's bias correction, and a large one means the local linear fit is picking up curvature in the running variable — the same thing a quadratic specification check would flag.

One note on the 0.1505 centering: it shifts the window by half a thousandth on both sides, so the effective left edge is 0.0005 further from 0.150 than a window centered at 0.150 would be. That's immaterial at these bandwidths, but it means h_left and h_right aren't directly comparable to the ±50 integer windows in the OLS tables without that offset in mind.
**What I changed, and why:**

- Moved `!pip install rdrobust==2.0.0` into the commented-out Colab setup cell, so it doesn't run in a clean checkout, where `rdrobust` is installed from `requirements.txt`.


## Section 4: Does the finding hold?

No AI used in this section.

## Section 5: Peer audit

No AI used in this section.

## Section 6: Your own repository

TODO
