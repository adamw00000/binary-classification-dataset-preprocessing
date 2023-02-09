# Outlier detection under False Omission Rate control - datasets

This repository contains test datasets, their summary and visualizations used in the paper. For methods and experiments described in the paper, refer to the [implementation repository](https://github.com/wawrzenczyka/FOR-CTL).

## Dataset files

Datasets used in the paper are collected in `data/` directory. Each of the `.csv` files corresponds to a single dataset with a matching name. Each CSV file contains `N` columns, where first `N - 2` are feature columns, and the last two are `Class` and `BinClass` columns - the former contains class assignment used in the original dataset, whereas the latter contains binarized label - `0` is treated as an outlier and `1` is treated as an inlier.

## Dataset summary

All calculated dataset statistics can be found in the `summary.csv` file in the repository root.

## Visualizations

You can find t-SNE visualization (with both original and binary classes) for each dataset in the `plots/` directory, in both `.png` and vector `.pdf` formats.

## Generation code

Dataset prepocessing code might be found in `preprocess-all.py` file. It uses [datahub.io](https://datahub.io) as a source for the data. the code allows for a simple adding of the new datasets or modyfing existing by defining the transformation function (`preprocessing_fun`) passed to the `process_dataset` function.