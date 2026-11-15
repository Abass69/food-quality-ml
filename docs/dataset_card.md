# Dataset card

Source: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), Cortez et al. (2009). DOI: https://doi.org/10.24432/C56S3T. Downloaded 15 November 2026. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The source archive is redistributed unchanged with attribution; transformations below apply only during analysis.

The archive contains 1,599 red and 4,898 white Portuguese Vinho Verde records. Eleven chemical measurements predict a sensory score. There are no spoilage, microbial contamination, storage-time, or image labels. The included `winequality.names` is the original data description.

## Features and units
| Feature | Unit / meaning |
|---|---|
| fixed acidity | g/dm³, tartaric acid |
| volatile acidity | g/dm³, acetic acid |
| citric acid | g/dm³ |
| residual sugar | g/dm³ |
| chlorides | g/dm³, sodium chloride |
| free sulfur dioxide | mg/dm³ |
| total sulfur dioxide | mg/dm³ |
| density | g/cm³ |
| pH | pH scale |
| sulphates | g/dm³, potassium sulphate |
| alcohol | percent by volume |
| wine_type | Added from file identity: red=0, white=1 |

Unit conventions should be checked against the original study before collecting new laboratory measurements. No unit conversion is performed in this pipeline.

## Target and transformations
Source quality is on a 0–10 sensory scale; observed unique-record scores are 3–9. Project bands: lower <=5, standard=6, higher >=7. These are analytical bins, not regulatory or commercial grade definitions. Lower quality does not mean unsafe, and higher quality does not mean safe.

Exact duplicate rows across input features and score are removed: 6,497 original rows become 5,320 rows. No missing values are observed. There are 5,320 distinct input profiles, so no conflicting labels on identical profiles remain in this release. Class counts after deduplication: lower 1,988; standard 2,323; higher 1,009. This deduplication changes the population weighting: the study evaluates unique profiles, not the frequency of all original records.

SHA-256 of source archive:
`3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4`

`source_id` is generated as color plus 1-based CSV data-row position; it is not a producer or sample identifier. No winery, batch, vintage, collection-time or true sample identifiers are available to the pipeline. Exact-profile grouping cannot remove unobserved batch dependence or near-duplicate leakage. The test is internal, not geographical or temporal validation. Dataset-specific associations cannot be assumed to transfer to other beverages or foods.
