#!/grid/zador/home/benjami/miniconda3/envs/geneformer2/bin/python

"""
This file is a custom script that converts MATLAB .mat files containing single-cell RNA-seq data into the HDF5-based AnnData format (.h5ad),
assuming that the .mat files are structured in a specific way (namely, that from Chen et al. 2023, i.e. Xiaoyin's lab at the Allen Institute).
"""

import anndata as ad
import numpy as np
import pickle
from pymatreader import read_mat
import os
from geneformer.util.parameters import ensembl_ids as ensembl_ids_path
from scipy.sparse import csc_matrix, csr_matrix

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
    with open("ensembl_ids.pickle",'rb') as f:
        ensemble_ids = pickle.load(f)

    f = read_mat(source_mat)

    n_cells = len(f['filt_neurons']['id'])

    adata_ref = ad.AnnData(matlab_to_csr(f['filt_neurons']['expmat'], n_cells), dtype = "float64")
    print(adata_ref)
    ## add metadata
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
    adata_ref = adata_ref[:, :106]
    print(adata_ref)
    ### convert gene names to ensemble IDs

    ensemble_ids["Tafa1"] = "ENSMUSG00000059187"
    ensemble_ids["Tafa2"] = "ENSMUSG00000044071"
    ensemble_ids["Ccn2"] = "ENSMUSG00000019997"

    genes = adata_ref.uns['genes']
    adata_ref.var_names = [ensemble_ids[g] for g in genes[:106]]

    # remove "bad" cells (less than 20 counts across all genes)
    passes_filter = adata_ref.obs['H1_type'] != 'filtered'
    adata_ref = adata_ref[passes_filter]

    print("Converted the following file:")
    print(adata_ref)

    adata_ref.write(target_h5ad)

if __name__ == '__main__':
    parent_directory = os.environ.get('ROOT_DATA_PATH') # /home/benjami/mnt/zador_nlsas_norepl_data/Ari/transcriptomics

    files = ["filt_neurons_D078_1L_CCFv2_newtypes",
            "filt_neurons_D078_2L_CCFv2_newtypes",
            "filt_neurons_D079_3L_CCFv2_newtypes",
            "filt_neurons_D079_4L_CCFv2_newtypes",
            "filt_neurons_D076_1L_CCFv2_newtypes",
            "filt_neurons_D077_1L_CCFv2_newtypes",
            "filt_neurons_D077_2L_CCFv2_newtypes"]
    
    for file in files:
        print("Loading file: ", file)
        source_mat = os.path.join(parent_directory, "barseq/Raw data Nov 2023", file + ".mat")
        target_h5ad = os.path.join(parent_directory, "barseq/test", file + ".h5ad")

        if os.path.exists(target_h5ad):
            print("File already exists, skipping")
            continue

        mat_to_h5(source_mat, target_h5ad)
        print(" ")

