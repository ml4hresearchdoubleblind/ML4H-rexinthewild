# Attribution, provenance, and licensing

## Read this before using the images

**Every one of the 381 source articles is licensed for non-commercial use only.
None is plain CC BY.** We verified this against PMC's own licence metadata for
all 381 articles; the per-article breakdown is in `source_articles.csv`.

| Licence | Articles | Questions |
|---|---|---|
| CC BY-NC | 137 (36.0%) | 335 (35.1%) |
| CC BY-NC-SA | 130 (34.1%) | 313 (32.8%) |
| CC BY-NC-**ND** | 114 (29.9%) | 306 (32.1%) |

Two consequences worth being explicit about:

**Commercial use of the images is not permitted** by any source article in this
benchmark. If you are evaluating a commercial model, note that the NC term
restricts the images, and take your own view on whether internal benchmarking
constitutes commercial use — that is a question for your counsel, not for us.

**Roughly a third of the images are ND (no derivative works).** Retrieving and
viewing them is fine. Redistributing modified copies is not. Resizing, cropping,
or re-encoding an image and then publishing or redistributing the result may not
be permitted for those articles. Preprocessing images in memory in order to run
inference is a different matter from publishing a derived image set.

Membership in the PMC Open Access Subset does not by itself grant reuse rights.
`fetch_images.py` retrieves each image directly from PMC so that you obtain it
under its own article's terms, rather than under any terms this repository might
appear to grant.

## Two different things, two different licences

### The questions and answer keys (`rexinthewild.csv`)

Written by the authors of the accompanying paper and released under
**Creative Commons Attribution 4.0 (CC BY 4.0)**.

The questions are original text. They pose clinical questions about what a figure
shows; they are not excerpts from, or reproductions of, the source articles, and
the factual content they describe is not itself copyrightable. This is why the
questions can carry a more permissive licence than the images they refer to.

### The images

**Not contained in this repository and not redistributed by it.** See above.

### The code

MIT, per `LICENSE`.

## Looking up a source article

Resolve any PMCID from the `pmcid` column:

```
https://pmc.ncbi.nlm.nih.gov/articles/<PMCID>/
```

`source_articles.csv` already carries, for each of the 381 articles: its licence
code, DOI, PMC version, formatted citation, and how many images and questions it
contributes. Machine-readable metadata is also available per article at:

```
https://pmc-oa-opendata.s3.amazonaws.com/<PMCID>.<version>/<PMCID>.<version>.json
```

At the time of release, none of the 381 articles was retracted and all were
present in the OA subset. Both facts can change; `is_retracted` in the JSON above
is the field to re-check.

## Citing

If you publish results on this benchmark, cite the accompanying paper. If you
reproduce an individual figure, cite that figure's source article too — the
`citation` column of `source_articles.csv` gives you a formatted reference for
every one.

## Data provenance

Images were selected from the PMC Open Access Subset for being **patient-style
photographs**: clinical or self-captured photographs of people, as opposed to
radiographs, histopathology, dermoscopy, charts, or diagrams.

Questions were drafted against each figure and its caption, then reviewed by
clinicians over two rounds. The first round involved clinicians of varying
experience levels. In the second, questions were reviewed by senior clinicians;
this produced revisions to a substantial fraction of items — including
corrections to gold answers — and the removal of questions that could not be
answered from the image alone.

Because gold-answer corrections were part of that process, accuracies computed
against any earlier version of this file are not comparable to those in the
paper. The file in this repository is the version the paper reports on.

## Where the images come from, technically

NCBI **retired its legacy FTP article-dataset service in August 2026**. PMC
content is now distributed through the PMC Cloud Service on AWS Open Data. Older
scripts built on `ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv` and the
`oa.fcgi` web service no longer work — both endpoints now return 404.
`fetch_images.py` uses the current object layout:

```
https://pmc-oa-opendata.s3.amazonaws.com/<PMCID>.<version>/<image_file_name>
```

Images are individual objects, so no bulk index or tarball is involved.
Documentation: <https://pmc.ncbi.nlm.nih.gov/tools/cloud/>

## Reporting problems

Errors in the questions, mismatched answer keys, and images that no longer
retrieve are all worth reporting. Please open an issue with the `image_id`.
