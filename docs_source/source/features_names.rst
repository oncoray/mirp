Feature name references
=======================

Features values exported in a `pandas.DataFrame` by `extract_features` are encoded in columns, see e.g. this
`tutorial <https://oncoray.github.io/mirp/tutorial_compute_radiomics_features_mr.html>`_. The column name corresponds
to the name of each feature. However, the naming of features can seem properly arcane to new users.

Each feature name consists of maximum 4 components, in order:

1. The filter and its parameters (if any).
2. The feature family and the feature name.
3. Feature parameters (if any).
4. Discretisation parameters (if any).

Here we will first describe the feature names and their parameters, sorted by feature family, followed by discretisation
parameters and filters, annotated with their Image Biomarker Standardisation Initiative identifiers [Zwanenburg2016]_,
[Depeursinge2020]_.

Features
--------

Morphological features (`HCUG`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three-dimensional morphological features are listed below:

* `morph_volume`: Volume (`RNU0`)
* `morph_vol_approx`: Approximate volume (`YEKZ`)
* `morph_area_mesh`: Surface area (`C0JK`)
* `morph_av`: Surface to volume ratio (`2PR5`)
* `morph_comp_1`: Compactness 1 (`SKGS`)
* `morph_comp_2`: Compactness 2 (`BQWJ`)
* `morph_sph_dispr`: Spherical disproportion (`KRCK`)
* `morph_sphericity`: Sphericity (`QCFX`)
* `morph_asphericity`: Asphericity (`25C7`)
* `morph_com`: Centre of mass shift (`KLMA`)
* `morph_diam`: Maximum 3D diameter (`L0JK`)
* `morph_pca_maj_axis`: Major axis length (`TDIC`)
* `morph_pca_min_axis`: Minor axis length (`P9VJ`)
* `morph_pca_least_axis`: Least axis length	(`7J51`)
* `morph_pca_elongation`: Elongation (`Q3CK`)
* `morph_pca_flatness`: Flatness (`N17B`)
* `morph_vol_dens_aabb`: Volume density - axis-aligned bounding box	(`PBX1`)
* `morph_area_dens_aabb`: Area density - axis-aligned bounding box (`R59B`)
* `morph_vol_dens_aee`: Volume density - approximate enclosing ellipsoid (`6BDE`)
* `morph_area_dens_aee`: Area density - approximate enclosing ellipsoid (`RDD2`)
* `morph_vol_dens_conv_hull`: Volume density - convex hull (`R3ER`)
* `morph_area_dens_conv_hull`: Area density - convex hull (`7T7F`)
* `morph_integ_int`: Integrated intensity (`99N0`)
* `morph_moran_i`: Moran's I index (`N365`)
* `morph_geary_c`: Geary's C measure (`NPT7`)
* `morph_vol_dens_ombb`: Volume density - oriented minimum bounding box (`ZH1A`; reference values absent)
* `morph_area_dens_ombb`: Area density - oriented minimum bounding box (`IQYR`; reference values absent)
* `morph_vol_dens_mvee`: Volume density - minimum volume enclosing ellipsoid (`SWZ1`; reference values absent)
* `morph_area_dens_mvee`: Area density - minimum volume enclosing ellipsoid (`BRI8`; reference values absent)

The final four features lack reference values in the IBSI standard. These are only computed if `ibsi_compliant=False`.

Local intensity features (`9ST6`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Local intensity features are listed below:

* `loc_peak_loc`: Local intensity peak (`VJGA`)
* `loc_peak_glob`: Global intensity peak (`0F91`)

Intensity-based statistical features (`UHIW`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Statistical features are listed below:

* `stat_mean`: Mean (`Q4LE`)
* `stat_var`: Variance (`ECT3`)
* `stat_skew`: Skewness (`KE2A`)
* `stat_kurt`: Kurtosis (`IPH6`)
* `stat_median`: Median (`Y12H`)
* `stat_min`: Minimum (`1GSF`)
* `stat_p10`: 10th percentile (`QG58`)
* `stat_p90`: 90th percentile (`8DWT`)
* `stat_max`: Maximum (`84IY`)
* `stat_iqr`: Interquartile range (`SALO`)
* `stat_range`: Range (`2OJQ`)
* `stat_mad`: Mean absolute deviation (`4FUA`)
* `stat_rmad`:Robust mean absolute deviation (`1128`)
* `stat_medad`: Median absolute deviation (`N72L`)
* `stat_cov`: Coefficient of variation (`7TET`)
* `stat_qcod`: Quartile coefficient of dispersion (`9S40`)
* `stat_energy`: Energy (`N8CA`)
* `stat_rms`: Root mean square (`5ZWQ`)


Examples
--------

References
----------
.. [Zwanenburg2016] Zwanenburg A, Leger S, Vallieres M, Loeck S. Image Biomarker Standardisation Initiative. arXiv
  [cs.CV] 2016. doi:`10.48550/arXiv.1612.07003 <https://doi.org/10.48550/arXiv.1612.07003>`_

.. [Depeursinge2020] Depeursinge A, Andrearczyk V, Whybra P, van Griethuysen J, Mueller H, Schaer R, et al.
  Standardised convolutional filtering for radiomics. arXiv [eess.IV]. 2020.
  doi:`10.48550/arXiv.2006.05470 <https://doi.org/10.48550/arXiv.2006.05470>`_