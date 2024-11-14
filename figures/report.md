## Single-cell predictions

### Area classification is much easier for some cell types

Although the average accuracy of area predictions for single cells is about 30%, not all cells are equally easy to spatially locate. Some cell types are much easier than others.

This can be seen by stratifying the accuracy by `h2_type`:

![Percentage by H2 types](single_cell_pct_correct_H2_type.png)

The finest level of types (H3) shows even greater heterogeneity.

#### Not surprising given distributions of cells.

This result shouldn't surprise us much, as some cell types are highly localized in the cortex to a particular area:

![H2 type locations](h2_types_location.png)




## Brains have different spatial coverage

The spatial coverage of the brain is not uniform across brains. Some brains have more cells in some areas than others:
<img src="CCF_areas_D076_1L.png" alt="Spatial coverage" width="400">
<img src="CCF_areas_D077_1L.png" alt="Spatial coverage" width="400">
<img src="CCF_areas_D078_1L.png" alt="Spatial coverage" width="400">
<img src="CCF_areas_D079_3L.png" alt="Spatial coverage" width="400">