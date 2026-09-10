# Attribution, provenance, and licensing

## Two different things with two different licences

This repository contains **questions**, not images. They are covered separately.

### The questions and answer keys (`rexinthewild.csv`)

Written by the authors of the accompanying paper and released under
**Creative Commons Attribution 4.0 (CC BY 4.0)**. You may use, adapt, and
redistribute them, including commercially, with attribution.

The questions are original text. They describe the figures rather than
reproducing them, and they are not excerpts from the source articles.

### The images

**Not contained in this repository and not redistributed by it.**

Every image belongs to its source article in the PubMed Central Open Access
subset, and each article carries **its own licence**, which varies:

- Many are CC BY, permitting reuse with attribution.
- A substantial number are **CC BY-NC** or **CC BY-NC-ND**, which prohibit
  commercial use and, for ND, derivative works.
- Some are in the OA subset under terms that permit access through PMC but
  restrict redistribution.

**Membership in the PMC Open Access subset does not by itself grant you reuse
rights.** `fetch_images.py` retrieves each image directly from PMC so that you
obtain it under its own terms, rather than under any terms this repository might
imply. Before using the images for anything beyond local evaluation — and
especially before redistributing them, including them in a derived dataset, or
using them commercially — check the licence on each source article.

To look one up, resolve its PMCID from the `pmcid` column:

```
https://www.ncbi.nlm.nih.gov/pmc/articles/<PMCID>/
```

The licence statement appears on the article page, and machine-readable licence
metadata is available through the PMC OA Web Service:

```
https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=<PMCID>
```

## Citing the source articles

If you publish results on this benchmark, cite the benchmark paper. If you
reproduce any individual figure, cite that figure's source article as well —
the `pmcid` column gives you what you need to resolve the full reference.

The 954 questions draw on 381 articles. A citation list for all of them is not
reproduced here; resolve them from the PMCIDs as needed.

## Data provenance

Images were selected from the PMC Open Access subset for being **patient-style
photographs** — clinical or self-captured photographs of people, as opposed to
radiographs, histopathology, dermoscopy, charts, or diagrams.

Questions were drafted against each figure and its caption, then reviewed by
clinicians in two rounds. The first round involved clinicians of varying
experience levels. In the second, questions were reviewed by senior clinicians,
which resulted in revisions to a substantial fraction of items — including
corrections to gold answers — and the removal of questions that could not be
answered from the image alone.

Because gold-answer corrections were part of that process, accuracies computed
against any earlier version of this file are not comparable to those in the
paper. The dataset in this repository is the version the paper reports on.

## NCBI access policy

`fetch_images.py` requires a contact email, sent in the `User-Agent` header,
and rate-limits itself to roughly 3 requests per second. Both are conditions of
NCBI's automated-access policy. Please do not remove them, and do not raise the
request rate — sustained abuse gets IP ranges blocked, which affects everyone
using the same network.

## Reporting problems

Errors in the questions, mismatched answer keys, and images that no longer
retrieve are all worth reporting. Please open an issue in this repository with
the `image_id` and a description.
