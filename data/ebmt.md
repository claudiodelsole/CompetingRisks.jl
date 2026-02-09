## Multicenter Bone Marrow Transplantation Data

Random subsample of 400 patients extracted from the bone marrow transplant registry of the European Blood and Marrow Transplant (EBMT) Group.

Available as part of the R package [`crrSC`](https://cran.r-project.org/package=crrSC) by running `data(center)`.

**Variables**

- *id*: id of transplantation center.

- *ftime*: event time.

- *fstatus*: event type;
0 = censored, 1 = acute or chronic GvHD , 2 = death free of GvHD.

- *cells*: source of stem cells;
0 = peripheral blood, 1 = bone marrow.

- *fm*: female donor to male recipient match;
0 = no match, 1 = match.