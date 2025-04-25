#!/usr/bin/env python

"""
This script converts MATLAB .mat files containing single-cell RNA-seq data into the HDF5-based AnnData format (.h5ad).
It's specifically designed for files from Chen et al. 2023 (Xiaoyin's lab at the Allen Institute).

Usage:
    python mat_to_h5ad.py --input input.mat --output output.h5ad
    python mat_to_h5ad.py --input_dir input_dir --output_dir output_dir
"""

import argparse
import anndata as ad
import numpy as np
import pickle
import os
from scipy.sparse import csc_matrix, csr_matrix
from pymatreader import read_mat

# Flag to control Ensembl ID conversion
CONVERT_TO_ENSEMBL_NAMES = False


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert MATLAB .mat files to AnnData .h5ad format")
    
    # File options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to a single input .mat file")
    input_group.add_argument("--input_dir", help="Directory containing input .mat files")
    
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", help="Path for the output .h5ad file")
    output_group.add_argument("--output_dir", help="Directory for output .h5ad files")
    
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    
    return parser.parse_args()


def matlab_to_csc(matlab_array, num_rows=None):
    data = matlab_array['data']
    ir = matlab_array['ir']
    jc = matlab_array['jc']
    ir = np.array(ir, dtype=int)  # Convert 'ir' to integer array

    if num_rows is None:
        num_rows = max(ir) + 1
    num_cols = len(jc) - 1
    
    indptr = [0]
    for j in range(num_cols):
        indptr.append(jc[j+1])
    
    indices = ir
    csr = csc_matrix((data, indices, indptr), shape=(num_rows, num_cols))
    
    return csr


def matlab_to_csr(matlab_array, num_rows=None):
    return csr_matrix(matlab_to_csc(matlab_array, num_rows))


def mat_to_h5(source_mat, target_h5ad):
    """Convert a MATLAB .mat file to AnnData .h5ad format"""
    print(f"Converting {os.path.basename(source_mat)} to {os.path.basename(target_h5ad)}")
    
    if CONVERT_TO_ENSEMBL_NAMES:
        try:
            from geneformer.util.parameters import ensembl_ids as ensembl_ids_path
            with open("ensembl_ids.pickle", 'rb') as f:
                ensemble_ids = pickle.load(f)
        except (ImportError, FileNotFoundError):
            print("Warning: Could not load Ensembl IDs")
            ensemble_ids = {}

    # Read the MATLAB file
    f = read_mat(source_mat)
    n_cells = len(f['filt_neurons']['id'])

    # Create AnnData object
    adata_ref = ad.AnnData(matlab_to_csr(f['filt_neurons']['expmat'], n_cells), dtype="float64")
    print(adata_ref)
    
    # Add metadata
    for key in f['filt_neurons'].keys():
        if key != 'expmat':
            result = f['filt_neurons'][key]
            print(key, type(result), result.shape if hasattr(result, 'shape') else "not an array")
            
            if isinstance(result, float) or isinstance(result, int):
                adata_ref.uns[key] = result
            elif hasattr(f['filt_neurons'][key], 'shape') and len(f['filt_neurons'][key]) == len(adata_ref):
                if len(f['filt_neurons'][key].shape) == 2:
                    adata_ref.obsm[key] = f['filt_neurons'][key]
                elif len(f['filt_neurons'][key]) == len(adata_ref):
                    adata_ref.obs[key] = f['filt_neurons'][key]
            else:
                if len(f['filt_neurons'][key]) == len(adata_ref):
                    adata_ref.obs[key] = f['filt_neurons'][key]
                elif len(f['filt_neurons'][key]) == len(adata_ref.var):
                    adata_ref.var[key] = f['filt_neurons'][key]
                else:
                    adata_ref.uns[key] = f['filt_neurons'][key]
    
    # Truncate to 106 genes
    adata_ref = adata_ref[:, :106]
    print(adata_ref)

    # Convert gene names to Ensembl IDs if enabled
    if CONVERT_TO_ENSEMBL_NAMES:
        # Add missing mappings
        ensemble_ids["Tafa1"] = "ENSMUSG00000059187"
        ensemble_ids["Tafa2"] = "ENSMUSG00000044071"
        ensemble_ids["Ccn2"] = "ENSMUSG00000019997"
        
        genes = adata_ref.uns['genes']
        adata_ref.var_names = [ensemble_ids.get(g, g) for g in genes[:106]]

    # Filter low-quality cells
    passes_filter = adata_ref.obs['H1_type'] != 'filtered'
    adata_ref = adata_ref[passes_filter]
    print(f"Filtered out {sum(~passes_filter)} low-quality cells")

    print("Final AnnData object:")
    print(adata_ref)

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(target_h5ad)), exist_ok=True)
    
    # Write the result
    adata_ref.write(target_h5ad)
    print(f"Successfully wrote {target_h5ad}")


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    if args.input:
        # Single file conversion
        if os.path.exists(args.output) and not args.force:
            print(f"Output file {args.output} already exists. Use --force to overwrite.")
            return
        mat_to_h5(args.input, args.output)
    else:
        # Directory conversion
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory {args.input_dir} not found")
            return
            
        os.makedirs(args.output_dir, exist_ok=True)
        
        for filename in os.listdir(args.input_dir):
            if filename.endswith('.mat'):
                source_path = os.path.join(args.input_dir, filename)
                target_path = os.path.join(args.output_dir, filename.replace('.mat', '.h5ad'))
                
                if os.path.exists(target_path) and not args.force:
                    print(f"Output file {target_path} already exists. Skipping. Use --force to overwrite.")
                    continue
                    
                mat_to_h5(source_path, target_path)


if __name__ == '__main__':
    main()
