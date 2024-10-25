"""
Geneformer tokenizer.

Input data:
Required format: raw counts scRNAseq data without feature selection as .h5ad file
Required row (gene) attribute: "var_names"; Ensembl ID for each gene
Required col (cell) attribute: "n_counts"; total read counts in that cell
Optional col (cell) attribute: "filter_pass"; binary indicator of whether cell should be tokenized based on user-defined filtering criteria
Optional col (cell) attributes: any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below

Usage:
  from geneformer import TranscriptomeTokenizer
  tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ_major"}, nproc=4)
  tk.tokenize_data("h5ad_data_file", "output_directory", "output_prefix")
"""

import os
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
from datasets import Dataset
from scipy import sparse

GENE_MEDIAN_FILE = Path(__file__).parent / "gene_median_dictionary.pkl"
TOKEN_DICTIONARY_FILE = Path(__file__).parent / "token_dictionary.pkl"


def tokenize_cell(gene_vector, gene_tokens):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = gene_vector.nonzero()[1]
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector.data)[:2048]
    # tokenize
    sentence_tokens = gene_tokens[nonzero_mask][sorted_indices]
    return sentence_tokens


class TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict,
        nproc=1,
        gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
        prepend_cls=True,   
        prepend_tissue=False,
        prepend_species=None,
        gene_panel=None,
    ):
        """
        Initialize tokenizer.

        Parameters
        ----------
        custom_attr_name_dict : dict
            Dictionary of custom attributes to be added to the dataset.
            Keys are the names of the attributes in the dataset.
            Values are the names of the attributes in the h5ad file.
        nproc : int
            Number of processes to use for dataset mapping.
        gene_median_file : Path
            Path to pickle file containing dictionary of non-zero median
            gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl IDs:token).
        prepend_tissue : bool
            Whether to prepend tissue name to cell type name in the cell_type.
            Requires that the tissue name is in the h5ad file as "tissue", and
            that these are in the token dictionary.
        prepend_species : str
            If not None, adds "mouse" or "human" to the beginning of the cell type, and tokenizes it.
        gene_panel : list
            List of genes to include in the tokenization. If None, all genes are used.
        """
        # dictionary of custom attributes {output dataset column name: input .h5ad column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        self.prepend_species = prepend_species
        self.prepend_tissue = prepend_tissue
        self.prepend_cls = prepend_cls

        self.gene_panel = gene_panel
        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # gene keys for full vocabulary. Only include genes that are in both the median and token dictionaries.
        self.gene_keys = list(set(self.gene_median_dict.keys()) & set(self.gene_token_dict.keys()))
        if self.gene_panel is not None:
            self.gene_keys = list(set(self.gene_keys) & set(self.gene_panel))

        # protein-coding and miRNA gene list dictionary for selecting .h5ad rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    def tokenize_data(self, h5ad_data_path, output_directory, output_prefix, save=True):
        """
        Tokenize .h5ad file and save as tokenized .dataset in output_directory.

        Parameters
        ----------
        h5ad_data_file : Path
            Path to the h5ad file
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset
        output_prefix : str
            Prefix for output .dataset
        """
        if os.path.isdir(h5ad_data_path):
            tokenized_cells, cell_metadata = self.tokenize_files(Path(h5ad_data_path))
        else:
            tokenized_cells, cell_metadata = self.tokenize_file(h5ad_data_path)
        tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata)

        if save:
            return self.save_tokenized_data(tokenized_dataset, output_directory, output_prefix)
        else:
            return tokenized_dataset, None

    def save_tokenized_data(self, tokenized_dataset, output_directory, output_prefix):
        """
        Save tokenized data as .dataset in output_directory.

        Parameters
        ----------
        tokenized_dataset : Dataset
            Tokenized dataset
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset
        output_prefix : str
            Prefix for output .dataset
        """
        output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
        tokenized_dataset.save_to_disk(output_path)

        # save lengths
        lengths = [tokenized_dataset[i]["length"] for i in range(len(tokenized_dataset))]
        lengths_path = (Path(output_directory) / f"{output_prefix}_lengths").with_suffix(".pkl")

        with open(lengths_path, "wb") as f:
            pickle.dump(lengths, f)

        return tokenized_dataset, lengths

    def tokenize_files(self, h5ad_data_directory):
        tokenized_cells = []
        cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.keys()}

        # Iterate through h5ad files in the directory
        for h5ad_file_path in sorted(h5ad_data_directory.glob("*.h5ad")):
            print(f"Tokenizing {h5ad_file_path}")
            file_tokenized_cells, file_cell_metadata = self.tokenize_file(h5ad_file_path)
            tokenized_cells += file_tokenized_cells
            for k in cell_metadata.keys():
                cell_metadata[k] += file_cell_metadata[k]

        return tokenized_cells, cell_metadata


    def tokenize_file(self, h5ad_file_path):
        file_cell_metadata = {
            attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
        }

        adata = ad.read_h5ad(h5ad_file_path)
        if not sparse.issparse(adata.X):
            adata.X = sparse.csr_matrix(adata.X)

        if 'n_counts' not in adata.obs.keys():
            adata.obs['n_counts'] = adata.X.sum(axis=1)

        # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in adata.var_names]
        )[0]

        coding_miRNA_ids = adata.var_names[coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids] + [self.gene_token_dict["<pad>"]]
        ) # it's crucial that <pad> is index -1

        norm_factor_vector = np.array(
            [
                self.gene_median_dict[i]
                for i in coding_miRNA_ids
            ]
        )

        # define coordinates of cells passing filters for inclusion (e.g. QC)
        if 'filter_pass' in adata.obs.keys():
            filter_pass_loc = np.where(adata.obs['filter_pass'] == True)[0]
            counts = adata.obs["n_counts"].values[filter_pass_loc]
            data = adata.X[filter_pass_loc, :]
        else:
            filter_pass_loc = np.arange(adata.shape[0])
            counts = adata.obs["n_counts"].values
            data = adata.X

        tokenized_cells = []
        chunk_size = 250_000  # Number of rows to process in each chunk
        num_rows = data.shape[0]
        num_chunks = (num_rows - 1) // chunk_size + 1

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, num_rows)

            gene_vector_norm = (
                data[start:end, coding_miRNA_loc]
                .multiply(1. / counts[start:end, None])
                .multiply(10_000)
                .multiply(1 / norm_factor_vector[None, :])
            ).toarray()

            # Get the indices for sorting the chunk independently
            sorted_indices_chunk = np.argsort(
                -gene_vector_norm, axis=1
            )[:, :2048]

            # Genes with zero expression are masked with <pad> token
            zero_mask = gene_vector_norm[np.arange(gene_vector_norm.shape[0])[:, None], sorted_indices_chunk] == 0
            sorted_indices_chunk[zero_mask] = -1

            tokenized_cells_chunk = [
                coding_miRNA_tokens[sorted_indices_chunk[i]].tolist()
                for i in range(end - start)
            ]

            # Make the first token the tissue
            if self.prepend_tissue:
                tissue = adata.obs['tissue'][filter_pass_loc][start:end]
                tissue_tokens = [self.gene_token_dict[i] for i in tissue]
                
                for i in range(end - start):
                    tokenized_cells_chunk[i].pop()
                    tokenized_cells_chunk[i].insert(0, tissue_tokens[i])

            # Make the first token the species?
            if self.prepend_species is not None:
                species_token = self.gene_token_dict[self.prepend_species]
                for i in range(end - start):
                    tokenized_cells_chunk[i].pop()
                    tokenized_cells_chunk[i].insert(0, species_token)

            # Add cls token
            if self.prepend_cls:
                cls_token = self.gene_token_dict["<cls>"]
                for i in range(end - start):
                    tokenized_cells_chunk[i].pop()
                    tokenized_cells_chunk[i].insert(0, cls_token)
                
            tokenized_cells.extend(tokenized_cells_chunk)

        # Add custom attributes for cells to dict
        for tokenization_key in file_cell_metadata.keys():
            raw_key = self.custom_attr_name_dict[tokenization_key]
            if raw_key=='organ_major':
                if ('organ_major' not in adata.obs.keys()):
                    if ('tissue' in adata.obs.keys()):
                        file_cell_metadata[tokenization_key] = adata.obs['tissue'][filter_pass_loc].tolist()
                    else:
                        file_cell_metadata[tokenization_key] = ['brain' for i in range(len(filter_pass_loc))]
            if raw_key in adata.obs.keys():
                file_cell_metadata[tokenization_key] = adata.obs[raw_key][
                    filter_pass_loc
                ].tolist()
            elif raw_key in adata.obsm.keys():
                file_cell_metadata[tokenization_key] = adata.obsm[raw_key][
                    filter_pass_loc
                ].tolist()
            else:
                print(f"WARNING: Custom attribute {raw_key} not found in .h5ad file. Setting to 'Unk'")
                file_cell_metadata[tokenization_key] = ['Unk' for i in range(len(filter_pass_loc))]

        return tokenized_cells, file_cell_metadata

    def create_dataset(self, tokenized_cells, cell_metadata):
        # create dict for dataset creation
        dataset_dict = {"input_ids": tokenized_cells}
        dataset_dict.update(cell_metadata)


        # create dataset
        output_dataset = Dataset.from_dict(dataset_dict)

        # measure lengths of dataset
        def measure_length(example):
            example["length"] = len(example["input_ids"])
            return example

        output_dataset_truncated_w_length = output_dataset.map(
            measure_length, num_proc=self.nproc
        )

        return output_dataset_truncated_w_length
