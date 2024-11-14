from typing import Literal
from scipy.sparse import csr_matrix
import numpy as np
import anndata as ad

from sklearn import manifold
from scipy.stats import special_ortho_group
import colour 
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from collections import defaultdict
import warnings

def compute_hierarchical_averages(df, h3_vectors):
    """
    Compute average vectors for hierarchical types based on their contained H3 types.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with columns H1_type, H2_type, and H3_type
    h3_vectors (dict): Dictionary mapping H3_type to 3D vectors (as numpy arrays or lists)
    
    Returns:
    tuple: (h1_vectors, h2_vectors)
        - h1_vectors: Dictionary mapping H1_type to average vectors
        - h2_vectors: Dictionary mapping H2_type to average vectors
    """
    # Convert all vectors to numpy arrays if they aren't already
    h3_vectors = {k: np.array(v) for k, v in h3_vectors.items()}
    
    # Create mappings for the hierarchy
    h2_to_h1 = df[['H2_type', 'H1_type']].drop_duplicates().set_index('H2_type')['H1_type'].to_dict()
    h3_to_h2 = df[['H3_type', 'H2_type']].drop_duplicates().set_index('H3_type')['H2_type'].to_dict()
    
    # Check for H3 types in vectors that don't appear in DataFrame
    unknown_h3 = set(h3_vectors.keys()) - set(df['H3_type'])
    if unknown_h3:
        warnings.warn(f"Some H3 types in vectors not found in DataFrame and will be ignored: {unknown_h3}")
        # Remove unknown H3 types from vectors
        h3_vectors = {k: v for k, v in h3_vectors.items() if k not in unknown_h3}
    
    # Calculate H2 averages
    h2_vectors = defaultdict(list)
    for h3_type, vector in h3_vectors.items():
        h2_type = h3_to_h2[h3_type]
        h2_vectors[h2_type].append(vector)
    
    h2_vectors = {h2: np.mean(vectors, axis=0) 
                 for h2, vectors in h2_vectors.items()}
    
    # Calculate H1 averages
    h1_vectors = defaultdict(list)
    for h2_type, vector in h2_vectors.items():
        h1_type = h2_to_h1[h2_type]
        h1_vectors[h1_type].append(vector)
    
    h1_vectors = {h1: np.mean(vectors, axis=0) 
                 for h1, vectors in h1_vectors.items()}
    
    return h1_vectors, h2_vectors

def split_hierarchy_inplace(adata,level):
    """Split cell types into hierarchy, e.g. 'IT_5_2' -> 'IT_5' or 'IT_5_2' -> 'IT', depending on level.
    Returns adata with column 'cell_type'"""
    if level==0:
        def filt(x):
            if x == 'non_Exc':
                return x
            return x.split('_')[0]
    elif level==1:
        def filt(x):
            if x == 'non_Exc':
                return x
            return "_".join(x.split('_')[:2])
    else:
        filt = lambda x: x
    
    adata.obs['cell_type'] = list(map(filt, adata.obs['cell_type'].values))
    return adata

def get_colormap(adata, key="cell_type", plot_colorspace=False, include_unknown=False, unknown_color='w',
                deficiency: Literal[None, "Deuteranomaly", "Protanomaly", "Tritanomaly"] = None,
                severity=0):
    """ Returns a dictionary of colors in which the perceptual distance is equal to the type/type dissimilarity.
    The colormap changes each time this is run. 

    Optionally, you can specify a color deficiency (e.g. "Deuteranomaly") and severity (0-100) to create colors
    that are approximately perceptually uniform for a certain form of colorblindness.
    This uses the CVD simulator from Machado et al. (2009).
    
    Similarity is specifically 3d MDS embedding of the "psuedobulk" expression of cells in each cell type.
            What is the average gene expression across types? What is the similarity between those?
            we'll use these to define the cell-cell similarity and then select a colormap in which
            perceptual distance is equal to the type/type dissimilarity.
            
    Uses the LUV color space. Check the code to make this more brighter/vivid
    """
    labels = adata.obs[key].unique()
    if not include_unknown:
        labels = labels[labels!="Unknown"]
    bulks = []
    for label in labels:
        pseudobulk = adata[adata.obs[key]==label].X.mean(0)
        bulks.append(pseudobulk)
    bulks = np.array(np.stack(bulks))
    similarities = np.corrcoef(bulks- bulks.mean(axis=0,keepdims=True))
    if plot_colorspace:
        sns.heatmap(pd.DataFrame(similarities, index = labels, columns=labels))
        plt.show()

    embed3 = manifold.MDS(n_components=3, dissimilarity="precomputed", )
    colors3 = embed3.fit_transform(1-similarities)
    random_3d_rotation = special_ortho_group.rvs(3)
    colors3 = np.matmul(colors3,random_3d_rotation)

    luv=colors3.copy()
    luv[:,0]=luv[:,0]*0.5 + .5 # squish the lightness and make it lighter
    luv[:,1:]*=2 # more vivid
    xyz = colour.Luv_to_XYZ(luv*100)
    colors_rgb = np.maximum(np.minimum(colour.XYZ_to_sRGB(xyz, ),1),0)
    if deficiency is not None:
        matrix = colour.blindness.matrix_cvd_Machado2009(deficiency, severity)
        # this maps normal rgb -> simulated rgb. how can we choose colors in this space?
        raise NotImplementedError
    
    if plot_colorspace:
        embed = manifold.MDS(n_components=2, dissimilarity="precomputed")
        colors = embed.fit_transform(1-similarities)
        plt.scatter(colors[:,0],colors[:,1], 
                    c=colors_rgb, s=100)
        plt.gca().set_facecolor('gray')

    d= {cat: c for cat, c in zip(labels, colors_rgb)}
    if not include_unknown:
        d['Unknown'] = unknown_color
    return d
