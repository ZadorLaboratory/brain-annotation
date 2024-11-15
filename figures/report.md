## Single-cell predictions

### Area classification is much easier for some cell types

Although the average accuracy of area predictions for single cells is about 30%, not all cells are equally easy to spatially locate. Some cell types are much easier than others.

This can be seen by stratifying the accuracy by `h2_type`:

![Percentage by H2 types](single_cell_pct_correct_H2_type.png)

The finest level of types (H3) shows even greater heterogeneity.

#### Not surprising given distributions of cells.

This result shouldn't surprise us much, as some cell types are highly localized in the cortex to a particular area:

![H2 type locations](h2_types_location.png)

#### Why does pooling improve? Controlling for cell type

Pooling improves predictions in part because cells contain independent information (i.e. because of noise), and also because of cell heterogeneity. We can correct for the latter by controlling for cell type to see how much of the improvement is due to noise.

<img src="cell_type_group_size_accuracy.png" alt="Accuracy improvement by H2 types" width="400">
<img src="group_size_single_cell_pooling.png" alt="Accuracy improvement across all types" width="400">


## Brains have different spatial coverage

The spatial coverage of the brain is not uniform across brains. Some brains have more cells in some areas than others:
<img src="CCF_areas_D076_1L.png" alt="Spatial coverage" width="400">
<img src="CCF_areas_D077_1L.png" alt="Spatial coverage" width="400">
<img src="CCF_areas_D078_1L.png" alt="Spatial coverage" width="400">
<img src="CCF_areas_D079_3L.png" alt="Spatial coverage" width="400">