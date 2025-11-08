# GIST Data Structure

## Data Organization

We run our experiments with different source data, now organized as follows:

### PDF Files
- All PDF files are now stored in `data/pdfs/`
- We use Git Large File Storage (LFS) to manage these large files. See the [LFS tutorial](https://github.com/git-lfs/git-lfs/wiki/Tutorial) for details.

### PDF Information
- A JSON file containing information about PDF files is located at `data/docs/pdf_info.json`. This file maps PDF files to their respective samples or gold standards
- Additional information about the PDFs such as urls for download and report type is stored at `data/docs/sample_160_reports_URLs_and_report-type.csv`

### Samples
We have three sample datasets (not available in repository):
- sample_7 (previously "001-seven-sample-reports"): A very small selection of seven human-annotated sustainability reports, useful for initial code development & software testing. More details at the end of this readme.
- sample_39 (previously "002-forty-reports-first-labelling"): Another small selection of forty human-annotated sustainability reports. It is meant to give you a first idea of how well your code performs.
- sample_160 (previously "003-sample_140"): The sample we use for our first annotation task. Contains 160 documents (including the additional reports that may not be relevant financial/CSR reports).

### Gold Standards
We use two gold standard datasets stored in `data/evaluation_dataset/`:
- bbk_2023 (previously "data_labeled_anon.xlsx", not available in repository): A first gold standard covering 39 reports. 
- gist_2025: A second gold standard covering 139 reports.

### Git Large File Storage

Git repositories in general should be small (<1GB at LRZ, 10GB at most) or use
[Git Large File Storage (LFS)](https://gitlab.lrz.de/help/topics/git/lfs/index.md). All pdfs in the subfolders here
are therefore stored in LFS. See the [LFS tutorial](https://github.com/git-lfs/git-lfs/wiki/Tutorial) for details.

Following best practices in FAIR research projects (see 
[here](https://heidiseibold.ck.page/posts/setting-up-a-fair-and-reproducible-project) or 
[here](https://gin-tonic.netlify.app/standard/)), each directory contains
two sub-folders:
- ``ìnput-data``: The original, immutable data dump.
- ``processed-data``: A somehow cleaned version of the input data.

Ideally, we would have guides and codebooks of the data sources stored within each sub-folder, as recommended by the
[TIER Protocol](https://www.projecttier.org/tier-protocol/protocol-4-0/).


### Seven Sample Reports Overview
A small sample, completely non-random.

The following reports do not contain information about Scope 1/Scope 2/Scope 3 CO2 emissions:

- novonordisk
- pepsico

The following reports contain information. Extraction results were (sometimes) correct:

- asml (page 55: contains Scope 1 and Scope 2 for the years 2014-2016)
- puma (page 15: contains all three scopes for the years 2015-2018)

The following reports contain information, but I couldn't extract correct information so far:

- apple (page 66: contains all three scopes for the years 2016-2020). Problem is that there is (partly conflicting information?) on page 12,14,75,89. Even for a human very hard to tell what is correct.
- Chevron (pages 47, 48: contains Scope 1 and Scope 2 for the years 2016-2020): Challenge: a table that runs across three pages
- mercedesbenz (page 153: contains Scope 1, Scope 2 market-based, Scope 2 location-based for the years 2017-2021). Challenges: Mercedes reports on p. 153 Scope 1, Scope 2-market-based, Scope 2-location-based, Total-market-based, and Total-location based. From page 140 one can CALCULATE values for scope 3, separately for cars&vans in 2020 & 2021
