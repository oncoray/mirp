import os
from mirp.extract_features_and_images import extract_features

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_adc_conversion_multi_frame():
    # Multi-frame ADC image - micrometers^2/s
    feature_data = extract_features(
        adc_conversion="um2/s",
        image=os.path.join(CURRENT_DIR, "data", "adc_images_pm_dicom4qi", "data_1", "image.dcm"),
        base_feature_families="statistics"
    )[0]
    assert len(feature_data) == 1
    assert 4000.0 < feature_data["stat_max"].values[0] < 4200.0
    assert feature_data["stat_min"].values[0] == 0.0

    # Multi-frame ADC image - millimeters^2/s
    feature_data = extract_features(
        adc_conversion="mm2/s",
        image=os.path.join(CURRENT_DIR, "data", "adc_images_pm_dicom4qi", "data_1", "image.dcm"),
        base_feature_families="statistics"
    )[0]
    assert len(feature_data) == 1
    assert 0.0040 < feature_data["stat_max"].values[0] < 0.0042
    assert feature_data["stat_min"].values[0] == 0.0

    # Multi-frame ADC image - centimeters^2/s
    feature_data = extract_features(
        adc_conversion="cm2/s",
        image=os.path.join(CURRENT_DIR, "data", "adc_images_pm_dicom4qi", "data_1", "image.dcm"),
        base_feature_families="statistics"
    )[0]
    assert len(feature_data) == 1
    assert 0.000040 < feature_data["stat_max"].values[0] < 0.000042
    assert feature_data["stat_min"].values[0] == 0.0

    # Multi-frame ADC image - meters^2/s
    feature_data = extract_features(
        adc_conversion="m2/s",
        image=os.path.join(CURRENT_DIR, "data", "adc_images_pm_dicom4qi", "data_1", "image.dcm"),
        base_feature_families="statistics"
    )[0]
    assert len(feature_data) == 1
    assert 0.0000000040 < feature_data["stat_max"].values[0] < 0.0000000042
    assert feature_data["stat_min"].values[0] == 0.0

    # Multi-frame ADC image - none
    feature_data = extract_features(
        adc_conversion="none",
        image=os.path.join(CURRENT_DIR, "data", "adc_images_pm_dicom4qi", "data_1", "image.dcm"),
        base_feature_families="statistics"
    )[0]
    assert len(feature_data) == 1
    assert 4000.0 < feature_data["stat_max"].values[0] < 4200.0
    assert feature_data["stat_min"].values[0] == 0.0


def test_adc_conversion_legacy():
    # micrometers^2/s
    feature_data = extract_features(
        adc_conversion="um2/s",
        image=os.path.join(CURRENT_DIR, "data", "adc_images_mr", "SCAN_001", "adc_image"),
        base_feature_families="statistics"
    )[0]
    assert len(feature_data) == 1
    assert 9100.0 < feature_data["stat_max"].values[0] < 9200.0
    assert feature_data["stat_min"].values[0] == 0.0

    # millimeters^2/s
    feature_data = extract_features(
        adc_conversion="mm2/s",
        image=os.path.join(CURRENT_DIR, "data", "adc_images_mr", "SCAN_001", "adc_image"),
        base_feature_families="statistics"
    )[0]
    assert len(feature_data) == 1
    assert 0.0091 < feature_data["stat_max"].values[0] < 0.0092
    assert feature_data["stat_min"].values[0] == 0.0

    # centimeters^2/s
    feature_data = extract_features(
        adc_conversion="cm2/s",
        image=os.path.join(CURRENT_DIR, "data", "adc_images_mr", "SCAN_001", "adc_image"),
        base_feature_families="statistics"
    )[0]
    assert len(feature_data) == 1
    assert 0.000091 < feature_data["stat_max"].values[0] < 0.000092
    assert feature_data["stat_min"].values[0] == 0.0

    # meters^2/s
    feature_data = extract_features(
        adc_conversion="m2/s",
        image=os.path.join(CURRENT_DIR, "data", "adc_images_mr", "SCAN_001", "adc_image"),
        base_feature_families="statistics"
    )[0]
    assert len(feature_data) == 1
    assert 0.0000000091 < feature_data["stat_max"].values[0] < 0.0000000092
    assert feature_data["stat_min"].values[0] == 0.0

    # none
    # centimeters^2/s
    feature_data = extract_features(
        adc_conversion="none",
        image=os.path.join(CURRENT_DIR, "data", "adc_images_mr", "SCAN_001", "adc_image"),
        base_feature_families="statistics"
    )[0]
    assert len(feature_data) == 1
    assert 9100.0 < feature_data["stat_max"].values[0] < 9200.0
    assert feature_data["stat_min"].values[0] == 0.0
