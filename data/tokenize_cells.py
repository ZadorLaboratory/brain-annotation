#!/grid/zador/home/benjami/miniconda3/envs/geneformer/bin/python

from h5ad_tokenizer import TranscriptomeTokenizer
import time
import argparse
import pickle
import os
from datasets import Dataset, interleave_datasets, DatasetDict
import numpy as np
import json

parent_directory = os.environ.get('ROOT_DATA_PATH') # /home/benjami/mnt/zador_data_norepl/Ari/transcriptomics

parser = argparse.ArgumentParser(description='Tokenize loom files')
parser.add_argument('--gene_median_file', type=str, default="files/median_dict.pkl",
                    help='path to gene median file')
parser.add_argument('--token_dictionary_file', type=str, default="files/barseq_token_dict_cls.pkl",
                    help='path to token dictionary file')                    
parser.add_argument('--h5ad_data_directory', type=str, default="barseq/Chen2023",
                    help='path to data directory, relative to ROOT_DATA_PATH') 
parser.add_argument('--output_directory', type=str, default=".",
                    help='path to output directory')
parser.add_argument('--gene-panel-path', type=str, default="files/barseq_gene_panel.pkl",
                    help='Path to a gene panel file. If none, all genes are used. The gene panel file should be a pickle of a list. The gene names should be Ensemble gene IDs.')
parser.add_argument('--nproc', type=int, default=24, help='number of processes')
parser.add_argument('--output_prefix', type=str, default="train_test_barseq", help='output prefix for the tokenized files')

args = parser.parse_args()
print("args:", args)
t0 = time.time()
####### Get area annotations #######
# move up the hierarchy to get rid of the smallest 'areas' which are actually layers
with open('files/area_ancestor_id_map.json', 'r') as f:
    area_ancestor_id_map = json.load(f)
with open('files/area_name_map.json', 'r') as f:
    area_name_map = json.load(f)
area_name_map['0'] = 'outside_brain'
annotation2area_int = {0.0:0} # Map from annotation id to area id. Float to int
for a in area_ancestor_id_map.keys(): 
    higher_area_id = area_ancestor_id_map[str(int(a))][1] if len(area_ancestor_id_map[str(int(a))])>1 else a    
    annotation2area_int[float(a)] = higher_area_id

unique_areas = np.unique(list(annotation2area_int.values())) # List of unique areas (noncontiguous ints)
unique_annos = [area_name_map[str(int(a))] for a in unique_areas] # List of unique area names
area_classes = np.arange(len(unique_areas)) # Create a class for each area (0, 1, 2, ...)
id2id = {float(k):v for (k,v) in zip(unique_areas, area_classes)} # Map from area id to class id
annoation2area_class = {k: id2id[int(v)] for k,v in annotation2area_int.items()} # Map from annotation to area class
id2id_rev = {v:k for k,v in id2id.items()} # Map from class id to area id
area_class2area_name = {k: area_name_map[str(int(v))] for k,v in id2id_rev.items()} # Map from area class to area name
###############################

if args.gene_panel_path != "none":
    with open(args.gene_panel_path, 'rb') as f:
        gene_panel = pickle.load(f)
else:
    gene_panel = None

labels= {'CCF','CCF_streamlines', 'H1_type', 'CCFname', 'CCFparentname', 'id', 'H2_type', 'CCFano', 'H3_type'}
label_dict = {label: label for label in labels}

# All the odd-numbered animals were controls, and the even-numbered animals were enucleated.
# For example, D076-1L is a control, and D076-4L is enucleated.
train_filenames = ['filt_neurons_D076_1L_CCFv2_newtypes.h5ad', 
                   'filt_neurons_D077_1L_CCFv2_newtypes.h5ad',
                    'filt_neurons_D078_1L_CCFv2_newtypes.h5ad',
                    ]
test_filenames = ['filt_neurons_D079_3L_CCFv2_newtypes.h5ad',]

datasets = []
for filename in train_filenames:
    h5ad_data_path = os.path.join(parent_directory, args.h5ad_data_directory, filename)
    print("Tokenizing:", h5ad_data_path)
    tk = TranscriptomeTokenizer(label_dict, 
        gene_median_file=args.gene_median_file,
        token_dictionary_file=args.token_dictionary_file,
        gene_panel = gene_panel,
        nproc=args.nproc)

    tokenized_dataset,_ = tk.tokenize_data(h5ad_data_path, args.output_directory, filename, save=False)

    # Filter dataset to only include cells for which the CCF_streamlines is not nans
    tokenized_dataset = tokenized_dataset.filter(lambda x: not np.isnan(np.sum(x['CCF_streamlines'])))
    tokenized_dataset = tokenized_dataset.map(lambda x: {'area_label': annoation2area_class[x['CCFano']]})

    datasets.append(tokenized_dataset)

dataset = interleave_datasets(datasets)

# get test dataset
datasets = []
for filename in test_filenames:
    h5ad_data_path = os.path.join(parent_directory, args.h5ad_data_directory, filename)
    print("Tokenizing:", h5ad_data_path)
    tk = TranscriptomeTokenizer(label_dict, 
        gene_median_file=args.gene_median_file,
        token_dictionary_file=args.token_dictionary_file,
        gene_panel = gene_panel,
        nproc=args.nproc)

    tokenized_dataset,_ = tk.tokenize_data(h5ad_data_path, args.output_directory, filename, save=False)
    tokenized_dataset = tokenized_dataset.filter(lambda x: not np.isnan(np.sum(x['CCF_streamlines'])))
    tokenized_dataset = tokenized_dataset.map(lambda x: {'area_label': annoation2area_class[x['CCFano']]})

    datasets.append(tokenized_dataset)

test_dataset = interleave_datasets(datasets)

# Merge train and test datasets
final_dataset = DatasetDict({
    'train': dataset,
    'test': test_dataset
})

final_dataset.save_to_disk(os.path.join(args.output_directory, f"{args.output_prefix}.dataset"))
