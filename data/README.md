# Dataset creation

## Creation from scratch

1. (Optional) Download the .mat files from Chen et al. (2024)
2. Convert the .mat files to .h5ad format using the `mat_to_h5.py` script. This will produce a directory of h5ad files, each containing a single dataset. In the `.obsm` metadata of each file, there is a column called `CCFname` and another called `CCFano` that contains the area name and area number, respectively. In addition, there should an `obsm` entry called `CCF_streamlines` which is a 3d array of position on the cortical butterfly flatmap, with the third dimension being the depth of cells in the cortex.
3. Tokenize! Use the `tokenize.py` script to create a vocabulary and tokenize the data, specifying which files are train and which are test. This will produce a `tokenized` directory containing the tokenized data.