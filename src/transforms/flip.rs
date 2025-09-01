use crate::{boxes::boxes_to_components, transforms::pipeline::Transform};
use burn::{prelude::Backend, tensor::Tensor};

impl<B: Backend> Transform<B> {
    /// Flips a 3-channel image tensor and its associated bounding boxes vertically,
    /// applied stochastically with probability `p`.
    ///
    /// This augmentation mirrors the input image along the horizontal axis (top ↔ bottom),
    /// and adjusts bounding boxes accordingly to preserve alignment with objects in
    /// the transformed image.
    ///
    /// # Arguments
    ///
    /// * `p` – Probability (`0.0 ≤ p ≤ 1.0`) that the vertical flip is applied.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the image after optional vertical flip
    /// and associated metadata.
    ///
    /// # Notes
    ///
    /// - Bounding boxes are transformed vertically to remain consistent with the flipped image.  
    /// - If the transformation is skipped (based on probability `p`), the image and
    ///   bounding boxes remain unchanged.  
    /// - Vertical flips are less common than horizontal flips in object detection
    ///   pipelines, but may improve robustness depending on the dataset.
    ///
    pub fn random_vertical_flip(&mut self, p: f32) -> Self {
        if !self.should_apply(p) {
            return self.clone();
        }

        self.vertical_flip()
    }

    /// Vertically flips a 3-channel image tensor and its associated bounding boxes.
    ///
    /// This transformation mirrors the input image along the horizontal axis (top ↔ bottom)
    /// and adjusts bounding boxes to maintain alignment with objects in the flipped image.
    ///
    /// # Type Parameters
    ///
    /// * `B` – The backend used by the tensor, implementing the `Backend` trait.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the vertically flipped image and
    /// associated metadata.
    ///
    /// # Notes
    ///
    /// - Bounding boxes are updated to reflect the vertical flip.  
    /// - If the pipeline contains no bounding boxes, only the image is affected.  
    /// - This operation does not modify the original image in-place; it returns a new pipeline
    ///   state ready for further transformations or finalization.
    pub fn vertical_flip(&mut self) -> Self {
        let [_ch, height, _width] = self.image.dims();

        // Flip image vertically
        self.image = self.image.clone().flip([1]);

        // Flip bounding boxes vertically
        if let Some(bboxes) = self.bboxes.as_mut() {
            let (x1, y1, x2, y2) = boxes_to_components(bboxes.clone());

            let temp_y1 = y2.clone() + (height as f32 / 2.0 - y2) * 2.0;
            let temp_y2 = y1.clone() + (height as f32 / 2.0 - y1) * 2.0;

            self.bboxes = Some(Tensor::cat(vec![x1, temp_y1, x2, temp_y2], 1));
        }

        self.clone()
    }

    /// Horizontally flips a 3-channel image tensor and its associated bounding boxes.
    ///
    /// This transformation mirrors the input image along the vertical axis (left ↔ right)
    /// and adjusts bounding boxes to maintain alignment with objects in the flipped image.
    ///
    /// # Type Parameters
    ///
    /// * `B` – The backend used by the tensor, implementing the `Backend` trait.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the horizontally flipped image and
    /// associated metadata.
    ///
    /// # Notes
    ///
    /// - The flip is applied along the width dimension (`W`), reversing the left–right order of
    ///   pixels.  
    /// - Bounding box `x` coordinates are updated to reflect the horizontal flip; `y` coordinates
    ///   remain unchanged.  
    /// - If the pipeline contains no bounding boxes, only the image is affected.  
    /// - This operation does not modify the original image in-place; it returns a new pipeline
    ///   state ready for further transformations or finalization.
    pub fn horizontal_flip(&mut self) -> Self {
        let [_ch, _height, width] = self.image.dims();

        // Flip image horizontally
        self.image = self.image.clone().flip([2]);

        // Flip bounding boxes horizontally
        if let Some(bboxes) = self.bboxes.as_mut() {
            let (x1, y1, x2, y2) = boxes_to_components(bboxes.clone());

            let temp_x1 = x2.clone() + (width as f32 / 2.0 - x2) * 2.0;
            let temp_x2 = x1.clone() + (width as f32 / 2.0 - x1) * 2.0;

            self.bboxes = Some(Tensor::cat(vec![temp_x1, y1, temp_x2, y2], 1));
        }

        self.clone()
    }

    /// Flips a 3-channel image tensor and its associated bounding boxes horizontally,
    /// applied stochastically with probability `p`.
    ///
    /// This augmentation mirrors the input image along the vertical axis (left ↔ right),
    /// and adjusts bounding boxes accordingly to preserve alignment with objects in
    /// the transformed image.
    ///
    /// # Arguments
    ///
    /// * `p` – Probability (`0.0 ≤ p ≤ 1.0`) that the horizontal flip is applied.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the image after optional horizontal flip
    /// and associated metadata.
    ///
    /// # Notes
    ///
    /// - Bounding boxes are transformed horizontally to remain consistent with the flipped image.  
    /// - If the transformation is skipped (based on probability `p`), the image and
    ///   bounding boxes remain unchanged.
    /// - Horizontal flips are one of the most common augmentations in object detection,
    ///   often used to increase dataset diversity with minimal semantic change.
    ///
    pub fn random_horizontal_flip(&mut self, p: f32) -> Self {
        if !self.should_apply(p) {
            return self.clone();
        }
        self.horizontal_flip()
    }
}

#[cfg(test)]
mod tests {

    use std::hash::{DefaultHasher, Hash, Hasher};

    use burn::data::dataset::vision::BoundingBox;
    use burn::{
        backend::{NdArray, ndarray::NdArrayDevice},
        tensor::Tensor,
    };

    use crate::transforms::pipeline::{Transform, bbox_as_tensor, create_test_image};

    #[test]
    fn vertical_flip_test() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        let image = create_test_image(12, 12, [127, 128, 255]);
        let mut bboxes = Vec::<Tensor<B, 2>>::new();

        let bb = BoundingBox {
            coords: [1.0, 1.0, 6.0, 6.0],
            label: 0,
        };

        bboxes.push(bbox_as_tensor::<B>(bb, &device));

        let bb = BoundingBox {
            coords: [1.0, 2.0, 3.0, 4.0],
            label: 1,
        };
        bboxes.push(bbox_as_tensor::<B>(bb, &device));
        let bboxes = Tensor::cat(bboxes, 0);

        let t = Transform::new(image, Some(bboxes), None, device);

        let (image, bboxes, _) = t.clone().vertical_flip().finish().unwrap();

        // Test hash of flipped image
        let test_success_hash: u64 = 10732386221966926898;
        let mut h = DefaultHasher::new();
        image.to_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());

        // Check bounding box translations
        let bboxes = bboxes.unwrap();

        Tensor::<B, 2>::from_data([[1.0, 6.0, 6.0, 11.0], [1.0, 8.0, 3.0, 10.0]], &device)
            .to_data()
            .assert_eq(&bboxes.clone().to_data(), true);
    }

    #[test]
    fn horizontal_flip_test() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;
        let image = create_test_image(12, 12, [127, 128, 255]);
        let mut bboxes = Vec::<Tensor<B, 2>>::new();

        let bb = BoundingBox {
            coords: [1.0, 1.0, 6.0, 6.0],
            label: 0,
        };

        bboxes.push(bbox_as_tensor::<B>(bb, device));

        let bb = BoundingBox {
            coords: [1.0, 2.0, 3.0, 4.0],
            label: 1,
        };
        bboxes.push(bbox_as_tensor::<B>(bb, device));

        let bboxes = Tensor::cat(bboxes, 0);
        let t = Transform::new(image, Some(bboxes), None, device);

        let (image, bboxes, _) = t.clone().horizontal_flip().finish().unwrap();

        // Test hash of flipped image
        let test_success_hash: u64 = 10732386221966926898;
        let mut h = DefaultHasher::new();
        image.to_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());

        // Check bounding box translations
        let bboxes = bboxes.unwrap();

        Tensor::<B, 2>::from_data([[6.0, 1.0, 11.0, 6.0], [9.0, 2.0, 11.0, 4.0]], device)
            .to_data()
            .assert_eq(&bboxes.clone().to_data(), true);
    }
}
