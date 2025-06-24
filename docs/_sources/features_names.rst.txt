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

Intensity histogram features (`ZVCW`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intensity histogram features are similar to the statistical features listed above, but are computed from discretised
images. MIRP can compute the following intensity histogram features:

* `ih_mean`: Intensity histogram mean: (`X6K6`)
* `ih_var`: Intensity histogram variance (`CH89`)
* `ih_skew`: Intensity histogram skewness (`88K1`)
* `ih_kurt`: Intensity histogram kurtosis (`C3I7`)
* `ih_median`: Intensity histogram median (`WIFQ`)
* `ih_min`: Intensity histogram minimum (`1PR8`)
* `ih_p10`: Intensity histogram 10th percentile (`GPMT`)
* `ih_p90`: Intensity histogram 90th percentile (`OZ0C`)
* `ih_max`: Intensity histogram maximum (`3NCY`)
* `ih_mode`: Intensity histogram mode (`AMMC`)
* `ih_iqr`: Intensity histogram interquartile range (`WR0O`)
* `ih_range`: Intensity histogram range (`5Z3W`)
* `ih_mad`: Intensity histogram mean absolute deviation (`D2ZX`)
* `ih_rmad`: Intensity histogram robust mean absolute deviation (`WRZB`)
* `ih_medad`: Intensity histogram median absolute deviation (`4RNL`)
* `ih_cov`: Intensity histogram coefficient of variation (`CWYJ`)
* `ih_qcod`: Intensity histogram quartile coefficient of dispersion (`SLWD`)
* `ih_entropy`: Intensity histogram entropy (`TLU2`)
* `ih_uniformity`: Intensity histogram uniformity (`BJ5W`)
* `ih_max_grad`: Maximum histogram gradient (`12CE`)
* `ih_max_grad_g`: Maximum histogram gradient grey level (`8E6O`)
* `ih_min_grad`: Minimum histogram gradient (`VQB3`)
* `ih_min_grad_g`: Minimum histogram gradient grey level (`RHQZ`)

Intensity-volume histogram features (`P88C`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intensity volume histogram features are listed below. Note that the IBSI reference standard provides a general
definition of these features, whereas MIRP computes these features for specific values:

* `ivh_v10`: Volume fraction at 10% intensity (`BC2M`; `NK6P`)
* `ivh_v25`: Volume fraction at 25% intensity (`BC2M`)
* `ivh_v50`: Volume fraction at 50% intensity (`BC2M`)
* `ivh_v75`: Volume fraction at 75% intensity (`BC2M`)
* `ivh_v90`: Volume fraction at 90% intensity (`BC2M`; `4279`)
* `ivh_i10`: Intensity at 10% volume (`GBPN`; `PWN1`)
* `ivh_i25`: Intensity at 25% volume (`GBPN`)
* `ivh_i50`: Intensity at 50% volume (`GBPN`)
* `ivh_i75`: Intensity at 75% volume (`GBPN`)
* `ivh_i90`: Intensity at 90% volume (`GBPN`; `BOHI`)
* `ivh_diff_v10_v90`: Difference in volume fraction between 10% and 90% intensity (`DDTU`; `WITY`)
* `ivh_diff_v25_v75`: Difference in volume fraction between 25% and 75% intensity (`DDTU`)
* `ivh_diff_i10_i90`: Difference in intensity between 10% and 90% volume (`CNV2`; `JXJA`)
* `ivh_diff_i25_i75`: Difference in intensity between 25% and 75% volume (`CNV2`)
* `ivh_auc`: Area under IVH curve (`9CMM`; reference values absent)

The `ivh_auc` feature lacks reference values in the IBSI standard. It is only computed if `ibsi_compliant=False`.

Grey level co-occurrence matrix features (`LFYI`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from grey level co-occurrence matrices (GLCM) are:

* `cm_joint_max`: Joint maximum (`GYBY`)
* `cm_joint_avg`: Joint average (`60VM`)
* `cm_joint_var`: Joint variance (`UR99`)
* `cm_joint_entr`: Joint entropy (`TU9B`)
* `cm_diff_avg`: Difference average (`TF7R`)
* `cm_diff_var`: Difference variance (`D3YU`)
* `cm_diff_entr`: Difference entropy (`NTRS`)
* `cm_sum_avg`: Sum average (`ZGXS`)
* `cm_sum_var`: Sum variance (`OEEB`)
* `cm_sum_entr`: Sum entropy (`P6QZ`)
* `cm_energy`: Angular second moment (`8ZQL`)
* `cm_contrast`: Contrast (`ACUI`)
* `cm_dissimilarity`: Dissimilarity (`8S9J`)
* `cm_inv_diff`: Inverse difference (`IB1Z`)
* `cm_inv_diff_norm`: Normalised inverse difference (`NDRX`)
* `cm_inv_diff_mom`: Inverse difference moment (`WF0Z`)
* `cm_inv_diff_mom_norm`: Normalised inverse difference moment (`1QCO`)
* `cm_inv_var`: Inverse variance (`E8JP`)
* `cm_corr`: Correlation (`NI2N`)
* `cm_auto_corr`: Autocorrelation (`QWB0`)
* `cm_clust_tend`: Cluster tendency (`DG8W`)
* `cm_clust_shade`: Cluster shade (`7NFM`)
* `cm_clust_prom`: Cluster prominence (`AE86`)
* `cm_info_corr1`: First measure of information correlation (`R8DG`)
* `cm_info_corr2`: Second measure of information correlation (`JN9H`)
* `cm_mcc`: Maximum correlation coefficient (no identifier).

The `cm_mcc` feature lacks reference values in the IBSI standard. It is only computed if `ibsi_compliant=False`.

GLCM-features are computed with the following parameters, in sequential order:

* distance:
    * `d#.#`: Chebyshev distance for considering the neighbourhood for determining co-occurrence (`PVMT`). Typically
      `d1.0`.
* feature and matrix aggregation:
    * `2d_avg`: features computed by averaging feature values of each 2D directional matrix across all directions and
      slices (`BTW3`)
    * `2d_s_mrg`: features computed by averaging feature values for each slice after merging 2D directional matrices
      within that slice (`SUJT`)
    * `2.5d_d_mrg`: features computed by averaging feature values for each direction after merging 2D directional
      matrices corresponding to that direction (`JJUI`)
    * `2.5d_v_mrg`: feature computed from a single matrix after merging all 2D directional matrices (`ZW7Z`)
    * `3d_avg`: features computed by averaging feature values of each 3D directional matrix (`ITBB`)
    * `3d_v_mrg`: features computed from a single matrix after merging all 3D directional matrices (`IAZD`)

Grey level run length matrix features (`TP0I`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from grey level run length matrices (GLRLM) are:

* `rlm_sre`: Short runs emphasis (`22OV`)
* `rlm_lre`: Long runs emphasis (`W4KF`)
* `rlm_lgre`: Low grey level run emphasis (`V3SW`)
* `rlm_hgre`: High grey level run emphasis (`G3QZ`)
* `rlm_srlge`: Short run low grey level emphasis (`HTZT`)
* `rlm_srhge`: Short run high grey level emphasis (`GD3A`)
* `rlm_lrlge`: Long run low grey level emphasis (`IVPO`)
* `rlm_lrhge`: Long run high grey level emphasis (`3KUM`)
* `rlm_glnu`: Grey level non-uniformity (`R5YN`)
* `rlm_glnu_norm`: Normalised grey level non-uniformity (`OVBL`)
* `rlm_rlnu`: Run length non-uniformity (`W92Y`)
* `rlm_rlnu_norm`: Normalised run length non-uniformity (`IC23`)
* `rlm_r_perc`: Run percentage (`9ZK5`)
* `rlm_gl_var`: Grey level variance (`8CE5`)
* `rlm_rl_var`: Run length variance (`SXLW`)
* `rlm_rl_entr`: Run entropy (`HJ9O`)

GLRLM features are computed with the following parameter:

* feature and matrix aggregation:
    * `2d_avg`: features computed by averaging feature values of each 2D directional matrix across all directions and
      slices (`BTW3`)
    * `2d_s_mrg`: features computed by averaging feature values for each slice after merging 2D directional matrices
      within that slice (`SUJT`)
    * `2.5d_d_mrg`: features computed by averaging feature values for each direction after merging 2D directional
      matrices corresponding to that direction (`JJUI`)
    * `2.5d_v_mrg`: feature computed from a single matrix after merging all 2D directional matrices (`ZW7Z`)
    * `3d_avg`: features computed by averaging feature values of each 3D directional matrix (`ITBB`)
    * `3d_v_mrg`: features computed from a single matrix after merging all 3D directional matrices (`IAZD`)

Grey level size zone matrix features (`9SAK`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from grey level size zone matrices (GLSZM) are:

* `szm_sze`: Small zone emphasis (`5QRC`)
* `szm_lze`: Large zone emphasis (`48P8`)
* `szm_lgze`: Low grey level zone emphasis (`XMSY`)
* `szm_hgze`: High grey level zone emphasis (`5GN9`)
* `szm_szlge`: Small zone low grey level emphasis (`5RAI`)
* `szm_szhge`: Small zone high grey level emphasis (`HW1V`)
* `szm_lzlge`: Large zone low grey level emphasis (`YH51`)
* `szm_lzhge`: Large zone high grey level emphasis (`J17V`)
* `szm_glnu`: Grey level non-uniformity (`JNSA`)
* `szm_glnu_norm`: Normalised grey level non-uniformity (`Y1RO`)
* `szm_zsnu`: Zone size non-uniformity (`4JP3`)
* `szm_zsnu_norm`: Normalised zone size non-uniformity (`VB3A`)
* `szm_z_perc`: Zone percentage (`P30P`)
* `szm_gl_var`: Grey level variance (`BYLV`)
* `szm_zs_var`: Zone size variance (`3NSA`)
* `szm_zs_entr`: Zone size entropy (`GU8N`)

GLSZM features are computed with the following parameter:

* feature and matrix aggregation:
    * `2d`: features computed by averaging feature values of each 2D matrix across all slices (`8QNN`)
    * `2.5d`: features computed from a single matrix after merging all 2D matrices (`62GR`)
    * `3d`: features computed from 3D matrix (`KOBO`)

Grey level distance zone matrix features (`VMDZ`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from grey level distance zone matrices (GLDZM) are:

* `dzm_sde`: Small distance emphasis (`0GBI`)
* `dzm_lde`: Large distance emphasis (`MB4I`)
* `dzm_lgze`: Low grey level zone emphasis (`S1RA`)
* `dzm_hgze`: High grey level zone emphasis (`K26C`)
* `dzm_sdlge`: Small distance low grey level emphasis (`RUVG`)
* `dzm_sdhge`: Small distance high grey level emphasis (`DKNJ`)
* `dzm_ldlge`: Large distance low grey level emphasis (`A7WM`)
* `dzm_ldhge`: Large distance high grey level emphasis (`KLTH`)
* `dzm_glnu`: Grey level non-uniformity (`VFT7`)
* `dzm_glnu_norm`: Normalised grey level non-uniformity (`7HP3`)
* `dzm_zdnu`: Zone distance non-uniformity (`V294`)
* `dzm_zdnu_norm`: Normalised zone distance non-uniformity (`IATH`)
* `dzm_z_perc`: Zone percentage (`VIWW`)
* `dzm_gl_var`: Grey level variance (`QK93`)
* `dzm_zd_var`: Zone distance variance (`7WT1`)
* `dzm_zd_entr`: Zone distance entropy (`GBDU`)

GLDZM features are computed with the following parameter:

* feature and matrix aggregation:
    * `2d`: features computed by averaging feature values of each 2D matrix across all slices (`8QNN`)
    * `2.5d`: features computed from a single matrix after merging all 2D matrices (`62GR`)
    * `3d`: features computed from 3D matrix (`KOBO`)

Neighbourhood grey tone difference matrix features (`IPET`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from neighbourhood grey tone difference matrix (NGTDM) features are:

* `ngt_coarseness`: Coarseness (`QCDE`)
* `ngt_contrast`: Contrast (`65HE`)
* `ngt_busyness`: Busyness (`NQ30`)
* `ngt_complexity`: Complexity (`HDEZ`)
* `ngt_strength`: Strength (`1X9X`)

NGTDM features are computed with the following parameter:

* feature and matrix aggregation:
    * `2d`: features computed by averaging feature values of each 2D matrix across all slices (`8QNN`)
    * `2.5d`: features computed from a single matrix after merging all 2D matrices (`62GR`)
    * `3d`: features computed from 3D matrix (`KOBO`)

Neighbouring grey level dependence matrix features (`REK0`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed neighbouring grey level dependence matrix (NGLDM) features are:

* `ngl_lde`: Low dependence emphasis (`SODN`)
* `ngl_hde`: High dependence emphasis (`IMOQ`)
* `ngl_lgce`: Low grey level count emphasis (`TL9H`)
* `ngl_hgce`: High grey level count emphasis (`OAE7`)
* `ngl_ldlge`: Low dependence low grey level emphasis (`EQ3F`)
* `ngl_ldhge`: Low dependence high grey level emphasis (`JA6D`)
* `ngl_hdlge`: High dependence low grey level emphasis (`NBZI`)
* `ngl_hdhge`: High dependence high grey level emphasis (`9QMG`)
* `ngl_glnu`: Grey level non-uniformity (`FP8K`)
* `ngl_glnu_norm`: Normalised grey level non-uniformity (`5SPA`)
* `ngl_dcnu`: Dependence count non-uniformity (`Z87G`)
* `ngl_dcnu_norm`: Normalised dependence count non-uniformity (`OKJI`)
* `ngl_dc_perc`: Dependence count percentage (`6XV8`)
* `ngl_gl_var`: Grey level variance (`1PFV`)
* `ngl_dc_var`: Dependence count variance (`DNX2`)
* `ngl_dc_entr`: Dependence count entropy (`FCBV`)
* `ngl_dc_energy`: Dependence count energy (`CAS9`)

NGLDM features are computed with the following parameters:

* distance:
    * `d#.#`: Chebyshev distance for considering the neighbourhood for determining co-occurrence (`PVMT`). Typically
      `d1.0`.
* dependence coarseness:
    * `a#`: Coarseness parameter for assessing dependence (`VXRR`). Typically `a0`.
* feature and matrix aggregation:
    * `2d`: features computed by averaging feature values of each 2D matrix across all slices (`8QNN`)
    * `2.5d`: features computed from a single matrix after merging all 2D matrices (`62GR`)
    * `3d`: features computed from 3D matrix (`KOBO`)

Discretisation (`4R0B`)
-----------------------

Features from several feature families are computed from discretised images, i.e. where image intensities are binned,
notably intensity histogram features and features computed from texture matrices. These are indicated as follows:

* `fbs`: Fixed bin size (`Q3RU`)
* `fbn`: Fixed bin number (`K15C`)
* `fbsp`: Fixed bin size, pyradiomics variant (not IBSI-compliant)

These are then followed by a parameter specifying the number of bins or bin size:

* `w#.#`: Width of each bin for fixed bin size discretisation methods.
* `n#`: Number of bins for the fixed bin number discretisation method.

Filters
-------

Features can not only be computed from the base image, but also from filtered images (*response maps*). Features
computed from filtered images are prefixed by different filter-specific items, which are detailed below.

Gabor transformation (`Q88H`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from images filtered using Gabor filters are prefixed by:

* `gabor`: Indicating Gabor filters.
* `s#.#`:  Scale parameter (`41LN`), in physical units.
* `g#,#`: Ellipticity parameter (`GDR5`).
* `l#,#`: Wavelength parameters (`S4N6`) in physical units.
* `t#,#`: Filter orientation parameter (`FQER`), only shown if Gabor-filtered images are not pooled.
* Filter application:
    * `2D`: Gabor filter is applied by slice.
    * `3D`: Gabor filters are applied along every orthogonal direction.
* `invar`: Pseudo-rotational invariance (`O1AQ`). Absent if not invariant.

Gaussian transformation
^^^^^^^^^^^^^^^^^^^^^^^

Features computed from images filtered using Gaussian filters are prefixed by:

* `gaussian`: Indicating Gaussian filters.
* `s#,#`: Scale parameter (`41LN`), in physical units.

Gaussian filters lack reference values in the IBSI standard. They are only computed if `ibsi_compliant=False`.

Laplacian-of-Gaussian transformation (`L6PA`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from images filtered using Laplacian-of-Gaussian filters are prefixed by:

* `log`: Indicating Laplacian-of-Gaussian filters.
* `s#,#`: Scale parameter (`41LN`), in physical units.

Laws kernels (`JTXT`)
^^^^^^^^^^^^^^^^^^^^^

Feature computed from images filtered using Laws kernels are prefixed by:

* `laws`: Indicating filter using Laws kernels.
* Set of filter kernels (`JVAD`).
* `energy`: Indicates that an energy map (`PQSD`) was computed, otherwise absent.
* `delta`: Energy map distance (`I176`). Absent if an energy map was not computed.
* `invar`: Pseudo-rotational invariance (`O1AQ`). Absent if not invariant.

Mean transformation (`S60F`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from images filtered using mean filters are prefixed by:

* `mean`: Indicating mean filters.
* `d#`: Filter support (`YNOF`)

Non-separable wavelets (`LODD`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from images filtered using non-separable wavelets are prefixed by:

* `wavelet`: Indicating wavelet filters.
* Non-separable wavelet family (`389V`),
* `level#`: Wavelet decomposition filter (`GCEK`)

Separable wavelets (`25BO`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from images filtered using separable wavelets are prefixed by:

* `wavelet`: Indicating wavelet filters.
* Separable wavelet family (`BPXS`).
* Wavelet filter combination (`UK1F`).
* `level#`: Wavelet decomposition filter (`GCEK`)
* `decimated`: Decimated image decomposition (`PH3R`). Absent if stationary,
* `invar`: Pseudo-rotational invariance (`O1AQ`). Absent if not invariant.

Square transformation
^^^^^^^^^^^^^^^^^^^^^

Features computed from images that underwent square transformation are prefixed by:

* `square`: Indicating square transformation.

Square transformations lack reference values in the IBSI standard. They are only computed if `ibsi_compliant=False`.

Square root transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from images that underwent square root transformation are prefixed by:

* `sqrt`: Indicating square root transformation.

Square root transformations lack reference values in the IBSI standard. They are only computed if
`ibsi_compliant=False`.

Logarithmic transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from images that underwent logarithmic transformation are prefixed by:

* `lgrthm`: Indicating logarithmic transformation. Note that `log` refers to Laplacian-of-Gaussian filters.

Exponential transformations lack reference values in the IBSI standard. They are only computed if
`ibsi_compliant=False`.

Exponential transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Features computed from images that underwent exponential transformation are prefixed by:

* `exp`: Indicating exponential transformation.

Exponential transformations lack reference values in the IBSI standard. They are only computed if
`ibsi_compliant=False`.

References
----------
.. [Zwanenburg2016] Zwanenburg A, Leger S, Vallieres M, Loeck S. Image Biomarker Standardisation Initiative. arXiv
  [cs.CV] 2016. doi:`10.48550/arXiv.1612.07003 <https://doi.org/10.48550/arXiv.1612.07003>`_

.. [Depeursinge2020] Depeursinge A, Andrearczyk V, Whybra P, van Griethuysen J, Mueller H, Schaer R, et al.
  Standardised convolutional filtering for radiomics. arXiv [eess.IV]. 2020.
  doi:`10.48550/arXiv.2006.05470 <https://doi.org/10.48550/arXiv.2006.05470>`_