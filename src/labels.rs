use crate::coco_labels::COCO_LABELS;

/// Maps a subset of COCO category IDs to a compact, contiguous set of model IDs.
///
/// This remapping reduces memory usage during training and indexing by avoiding
/// large, sparse class representations. Cross-entropy loss benefits because it
/// uses integer class indices directly instead of one-hot encodings.
#[derive(Clone)]
pub struct SSDRemapCOCOID {
    det_classes: Vec<usize>,
}

impl SSDRemapCOCOID {
    /// Returns the total number of model classes including the background class.
    ///
    /// Background is always assigned to index `0`.
    ///
    /// # Returns
    ///
    /// The number of classes as `usize`.
    pub fn count(&self) -> usize {
        self.det_classes.len() + 1
    }

    /// Checks whether a given model class index exists in the mapping.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The class index in the model's ID space.
    ///
    /// # Returns
    ///
    /// `true` if the class index exists, otherwise `false`.
    pub fn contains(&self, model_id: usize) -> bool {
        self.det_classes.contains(&model_id)
    }

    // Converts COCO class names into their corresponding COCO class IDs
    // by looking them up in the `COCO_LABELS` array.
    //
    // The output IDs will later be remapped to the model's compact ID space.
    fn coco_names_to_model_ids(coco_names: Vec<&str>) -> Vec<usize> {
        let mut model_ids = vec![];

        for name in coco_names.iter() {
            for (i, label) in COCO_LABELS.iter().enumerate() {
                if *label == *name {
                    model_ids.push(i);
                    break;
                }
            }
        }

        model_ids
    }

    /// Creates a new mapping from a list of COCO class names.
    ///
    /// Internally, each name is mapped to its COCO ID and stored in the remapping table.
    ///
    /// # Arguments
    ///
    /// * `class` - Vector of COCO class names to include in the model.
    ///
    /// # Returns
    ///
    /// A new `SSDRemapCOCOIDs` instance with the given mapping.
    pub fn new(class: Vec<&str>) -> Self {
        let ids = Self::coco_names_to_model_ids(class);
        SSDRemapCOCOID { det_classes: ids }
    }

    /// Maps a model class index back to its COCO class name.
    ///
    /// Background is returned for index `0`.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model's class index.
    ///
    /// # Returns
    ///
    /// The COCO class name as a `&'static str`, or `"background"` if index is `0`.
    pub fn model_id_to_coco_name(&self, model_id: &usize) -> &'static str {
        let id = if *model_id > 0 {
            *model_id - 1
        } else {
            return "background";
        };
        COCO_LABELS[self.det_classes[id]]
    }

    /// Returns the list of COCO class names used in this model.
    ///
    /// Background is excluded, and the list is ordered by model class ID (starting at 1).
    ///
    /// # Returns
    ///
    /// A vector of COCO class names as `String`.
    pub fn names(&self) -> Vec<String> {
        self.det_classes
            .iter()
            .enumerate()
            .map(|(i, _)| self.model_id_to_coco_name(&(i + 1)).into())
            .collect()
    }

    /// Maps a COCO class ID to its model class index.
    ///
    /// # Arguments
    ///
    /// * `coco_id` - The original COCO dataset class ID.
    ///
    /// # Returns
    ///
    /// `Some(model_id)` if the class exists in the mapping, otherwise `None`.
    /// The returned `model_id` is 1-based (0 is background).
    pub fn coco_id_to_model_id(&self, coco_id: &usize) -> Option<usize> {
        self.det_classes
            .iter()
            .position(|x| x == coco_id)
            .map(|x| x + 1)
    }
}


mod tests { 
    
    #[test]
    fn coco_remap_test() {
        let objects = vec!["person", "cat", "dog"];
        let cr = crate::labels::SSDRemapCOCOID::new(objects.clone()); // 1, 17, 18

        assert_eq!(cr.coco_id_to_model_id(&1).unwrap(), 1);
        assert_eq!(cr.coco_id_to_model_id(&17).unwrap(), 2);
        assert_eq!(cr.coco_id_to_model_id(&18).unwrap(), 3);
        assert_eq!(cr.coco_id_to_model_id(&19), Option::None);
        assert_eq!(cr.count(), 4);

        // check array index value
        assert_eq!(cr.model_id_to_coco_name(&0), "background");
        assert_eq!(cr.model_id_to_coco_name(&1), "person");
        assert_eq!(cr.model_id_to_coco_name(&2), "cat");
        assert_eq!(cr.model_id_to_coco_name(&3), "dog");

        assert_eq!(objects, cr.names());
    }
}
