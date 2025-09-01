use burn::data::dataset::vision::ImageFolderDataset;
pub enum COCOVersion {
    V2017,
    V2018,
}

impl COCOVersion {
    pub fn to_string(&self) -> &str {
        match self {
            COCOVersion::V2017 => "2017",
            COCOVersion::V2018 => "2018",
        }
    }
}

pub trait COCODataSet {
    fn coco_ds_train(version: COCOVersion, root: String) -> Self;
    fn coco_ds_test(version: COCOVersion, root: String) -> Self;
}

impl COCODataSet for ImageFolderDataset {
    /// Creates a new COCO train dataset.
    fn coco_ds_train(version: COCOVersion, root: String) -> Self {
        let annotations =
            root.clone() + "/annotations/instances_train" + version.to_string() + ".json";
        let root = root + "train" + version.to_string() + "/";
        Self::new_coco_detection(&annotations, &root)
            .unwrap_or_else(|_| panic!("Error {} or {} not found.", &annotations, &root))
    }

    /// Creates a new COCO test dataset.
    fn coco_ds_test(version: COCOVersion, root: String) -> Self {
        let annotations =
            root.clone() + "/annotations/instances_val" + version.to_string() + ".json";
        let root = root + "val" + version.to_string() + "/";
        Self::new_coco_detection(&annotations, &root)
            .unwrap_or_else(|_| panic!("Error {} or {} not found.", &annotations, &root))
    }
}
