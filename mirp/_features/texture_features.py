from mirp._features.histogram import HistogramDerivedFeature


class FeatureTexture(HistogramDerivedFeature):

    def __init__(self, spatial_method: str, **kwargs):
        super().__init__(**kwargs)
        self.spatial_method = spatial_method.lower()

    def _get_spatial_table_name_element(self) -> list[str | None]:
        if self.spatial_method == "2d_average":
            table_elements = ["2d_avg"]
        elif self.spatial_method == "2d_slice_merge":
            table_elements = ["2d_s_mrg"]
        elif self.spatial_method == "2.5d_direction_merge":
            table_elements = ["2.5d_d_mrg"]
        elif self.spatial_method == "2.5d_volume_merge":
            table_elements = ["2.5d_v_mrg"]
        elif self.spatial_method == "3d_average":
            table_elements = ["3d_avg"]
        elif self.spatial_method == "3d_volume_merge":
            table_elements = ["3d_v_mrg"]
        elif self.spatial_method is None:
            table_elements = []
        else:
            table_elements = [self.spatial_method]

        return table_elements
