## Melanoma Survival Data

Survival data collected by Drzewiecki et al. (1980) at the Odense University Hospital, Denmark, on patients diagnosed with melanoma who underwent surgery between 1962 and 1977.

Available as part of the R package [`timereg`](https://cran.r-project.org/package=timereg) by running `data(melanoma)`.

**Variables**

- *no*: patient code.

- *status*: survival status. 
1 = death from melanoma, 2 = alive (censored), 3 = death from other cause.

- *days*: survival time.

- *ulc*: ulceration indicator. 
0 = absent, 1 = present.

- *thick*: tumour thickness (1/100 mm).

- *sex*: 0 = female, 1 = male.