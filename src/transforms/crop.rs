use crate::{
    boxes::{boxes_to_components, get_iou},
    broadcast,
};

use super::pipeline::Transform;
use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, cast::ToElement},
};
use rand::Rng;

impl<B: Backend> Transform<B> {
    pub fn crop_area(&self, c_x1: usize, c_y1: usize, c_x2: usize, c_y2: usize) -> Tensor<B, 3> {
        let [ch, image_height, image_width] = self.image.dims();

        if c_x2 <= c_x1 || c_y2 <= c_y1 {
            panic!(
                "Box size error x2 and y2 cannot be smaller or equal to x1 or y1: {c_x1},{c_y1},{c_x2},{c_y2}"
            );
        };

        if c_x2 >= image_width || c_y2 >= image_height {
            panic!(
                "Box size error x2 or y2 can not be larger than image dimensions: {c_x2},{c_y2},{image_width},{image_height}"
            );
        }

        let crop_width = c_x2 - c_x1;
        let crop_height = c_y2 - c_y1;

        // Create mask with the crop area and location
        let mask = Tensor::<B, 3, Int>::ones([3, crop_height, crop_width], &self.device);
        let mask = mask.pad((c_x1, image_width - c_x2, c_y1, image_height - c_y2), 0);
        let mask = Tensor::cat(mask.flatten::<1>(0, 2).bool().nonzero(), 0);

        // Crop the image using the index of unmasked pixels
        let image = self.clone().image.flatten::<1>(0, 2);
        image.select(0, mask).reshape([ch, crop_height, crop_width])
    }

    /// Crops a 3-channel image tensor to the specified rectangular region and adjusts
    /// associated bounding boxes accordingly.
    ///
    /// This transformation extracts the region defined by `(c_x1, c_y1)` as the top-left
    /// corner and `(c_x2, c_y2)` as the bottom-right corner. Bounding boxes that are
    /// partially or fully outside the crop are filtered or truncated to remain within
    /// the new image region.
    ///
    /// # Parameters
    ///
    /// * `c_x1` – X-coordinate of the top-left corner of the crop.  
    /// * `c_y1` – Y-coordinate of the top-left corner of the crop.  
    /// * `c_x2` – X-coordinate of the bottom-right corner of the crop.  
    /// * `c_y2` – Y-coordinate of the bottom-right corner of the crop.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the cropped image tensor and adjusted
    /// bounding boxes (if present).
    ///
    /// # Notes
    ///
    /// - Bounding boxes entirely outside the crop are removed.  
    /// - Bounding boxes partially overlapping the crop are clipped to the crop boundaries.  
    /// - Labels are updated to match the remaining valid bounding boxes.  
    /// - The operation does not modify the original image in-place; it returns a new pipeline
    ///   state ready for further augmentations or finalization.  
    /// - Panics if the crop coordinates are invalid (e.g., zero or negative width/height, or
    ///   outside the image dimensions).
    ///

    pub fn crop(&mut self, c_x1: usize, c_y1: usize, c_x2: usize, c_y2: usize) -> Self {
        let cropped = self.crop_area(c_x1, c_y1, c_x2, c_y2);

        // Crop and remove bounding boxes if necessary
        if let Some(bboxes) = self.bboxes.clone() {
            let (x1, y1, x2, y2) = boxes_to_components(bboxes.clone());
            let (c_x1, c_x2, c_y1, c_y2) = (c_x1 as i32, c_x2 as i32, c_y1 as i32, c_y2 as i32);

            // Remove all the bboxes and labels outside of the crop
            //
            // Boxes overlap if the following is true:
            //
            // a_x1 < b_x2 &&
            // a_x2 < b_x1 &&
            // a_y1 < b_y2 &&
            // a_y2 > b_y1
            //
            // Cool demo of this
            //
            // https://silentmatt.com/rectangle-intersection/

            let mask = x1
                .clone()
                .lower_elem(c_x2)
                .bool_and(x2.clone().greater_elem(c_x1))
                .bool_and(y1.clone().lower_elem(c_y2))
                .bool_and(y2.clone().greater_elem(c_y1))
                .flatten::<1>(0, 1);

            let valid_index = Tensor::cat(mask.clone().nonzero(), 0);
            let valid_boxes = bboxes.clone().select(0, valid_index.clone());

            // Remove corresponding labels

            if let Some(labels) = self.labels.clone() {
                self.labels = Some(labels.clone().select(0, valid_index.clone()));
            }

            let (x1, y1, x2, y2) = boxes_to_components(valid_boxes);

            // Find where the remaining boxes intersect the new crop and crop their
            // coordinates to inside the new bounding box

            let c_x1_t = Tensor::from_data([c_x1], &self.device);
            let (a, b) = broadcast!(x1: Tensor<B, 2>, c_x1_t: Tensor<1>);
            let i_x1 = a.max_pair(b);

            let c_y1_t = Tensor::from_data([c_y1], &self.device);
            let (a, b) = broadcast!(y1: Tensor<B, 2>, c_y1_t: Tensor<1>);
            let i_y1 = a.max_pair(b);

            let c_x2_t = Tensor::from_data([c_x2], &self.device);
            let (a, b) = broadcast!(x2: Tensor<B, 2>, c_x2_t: Tensor<1>);
            let i_x2 = a.min_pair(b);

            let c_y2_t = Tensor::from_data([c_y2], &self.device);
            let (a, b) = broadcast!(y2: Tensor<B, 2>, c_y2_t: Tensor<1>);
            let i_y2 = a.min_pair(b);

            let x1 = i_x1 - c_x1;
            let y1 = i_y1 - c_y1;
            let x2 = i_x2 - c_x1;
            let y2 = i_y2 - c_y1;

            self.bboxes = Some(Tensor::cat(vec![x1, y1, x2, y2], 1));
        }

        self.image = cropped;

        self.clone()
    }

    /// Performs a random crop on the image tensor based on Intersection-over-Union (IoU) with
    /// existing bounding boxes.
    ///
    /// The method attempts to randomly sample a crop window that satisfies a minimum IoU threshold
    /// with at least one of the bounding boxes. It retries multiple times with thresholds randomly
    /// chosen from `[0.3, 0.5, 0.7, 0.9, 1.0]`. If no crop satisfies the condition, the original
    /// image is returned unchanged.
    ///
    /// # Parameters
    ///
    /// * `p` – Probability of applying the random IoU crop. Values should be between `0.0` and
    /// `1.0`.
    ///
    /// # Returns
    ///
    /// A new `Transform<B>` instance with the cropped image and adjusted bounding boxes if a crop
    /// was applied, or a clone of the original transform if the crop was skipped (based on `p`)
    /// or no valid crop was found.
    ///
    /// # Behavior
    ///
    /// - Checks probabilistically whether to apply the crop using `should_apply(p)`.  
    /// - Samples random crop windows and computes IoU with existing bounding boxes.  
    /// - Accepts the first crop that satisfies the random threshold with at least one bounding
    ///   box.  
    /// - Adjusts the image and bounding boxes via the `crop` method.  
    /// - Returns a new transform pipeline ready for further augmentations.
    ///
    /// # Notes
    ///
    /// - The image tensor is assumed to have 3 dimensions: `[channels, height, width]`.  
    /// - Panics if the crop coordinates generated are invalid (should not occur with proper
    ///   bounds).  
    ///
    pub fn random_iou_crop(&mut self, p: f32) -> Self {
        if !self.should_apply(p) {
            return self.clone();
        }

        let [_ch, height, width] = self.image.dims();

        for _ in 0..40 {
            // Pick a threshold randomly from list
            let thresholds = [0.3, 0.5, 0.7, 0.9, 1.0];
            let rdm_threshold = thresholds[self.rng.random_range(0..thresholds.len())];

            // Randomly sample a crop window
            let x1 = self.rng.random_range(0..(width));
            let y1 = self.rng.random_range(0..(height));
            let x2 = self.rng.random_range((x1)..width);
            let y2 = self.rng.random_range((y1)..height);
            let rdm_box = Tensor::from_data(
                [[x1 as f32, y1 as f32, x2 as f32, y2 as f32]],
                &self.image.device(),
            );

            // Compute IoU with all bounding boxes.
            let iou = get_iou(rdm_box.clone(), self.bboxes.clone().unwrap());
            let threshold_mask = iou.clone().max_dim(0).greater_elem(rdm_threshold);

            // If any box has IoU >= threshold accept the crop.
            if threshold_mask.int().sum().into_scalar().to_i32() > 0 {
                self.crop(x1, y1, x2, y2);
                break;
            }
        }

        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash, Hasher};

    use burn::{
        backend::{NdArray, ndarray::NdArrayDevice},
        tensor::{Tolerance, ops::FloatElem},
    };
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::{
        debug::set_tensor_dbg_precision,
        transforms::pipeline::{bbox_as_tensor, create_test_image},
    };
    use burn::data::dataset::vision::BoundingBox;

    #[test]
    fn crop_test() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        set_tensor_dbg_precision(2);

        let image = create_test_image(20, 20, [3, 3, 3]);

        let bb1 = BoundingBox {
            coords: [1.0, 1.0, 6.0, 6.0],
            label: 0,
        };

        let bb2 = BoundingBox {
            coords: [1.0, 1.0, 2.0, 2.0],
            label: 0,
        };

        let bboxes = Tensor::cat(
            vec![
                bbox_as_tensor::<B>(bb1, device),
                bbox_as_tensor::<B>(bb2, device),
            ],
            0,
        );

        let labels = Tensor::from_data([1, 2], device);

        let t = Transform::new(image, Some(bboxes), Some(labels), device);

        let (image, bboxes, _labels) = t.clone().crop(5, 5, 10, 10).finish().unwrap();

        assert!(image.shape().dims() == [3, 5, 5]);

        Tensor::<B, 2>::from_data([[0.00, 0.00, 1.00, 1.00]], device)
            .into_data()
            .assert_approx_eq::<FloatElem<B>>(&bboxes.unwrap().to_data(), Tolerance::default());
    }

    #[test]
    fn test_iou_crop() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;
        let test_success_hash: u64 = 10393011717360222344;
        let image = create_test_image(128, 128, [128, 128, 255]);
        let image_t: Tensor<B, 3> = Transform::rgb_img_as_tensor(image, device);

        let bboxes = Tensor::<B, 2>::from_data(
            [
                [12.8000, 12.8000, 38.4000, 38.4000],
                [25.6000, 32.0000, 51.2000, 57.6000],
                [11.8000, 11.0000, 102.4000, 89.6000],
                [44.8000, 19.2000, 70.4000, 44.8000],
                [64.0000, 76.8000, 89.6000, 102.4000],
                [32.0000, 51.2000, 57.6000, 76.8000],
            ],
            device,
        );

        let aug = Transform::new_seeded(image_t, Some(bboxes), None, StdRng::seed_from_u64(3));

        let (image, bboxes, _) = aug.clone().random_iou_crop(1.0).finish().unwrap();
        // Test hash of image
        let mut h = DefaultHasher::new();
        image.to_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());

        Tensor::<B, 2>::from_data(
            [
                [0.00, 0.00, 14.40, 10.40],
                [1.60, 4.00, 27.20, 29.60],
                [0.00, 0.00, 76.00, 61.60],
                [20.80, 0.00, 46.40, 16.80],
                [40.00, 48.80, 65.60, 66.00],
                [8.00, 23.20, 33.60, 48.80],
            ],
            device,
        )
        .into_data()
        .assert_approx_eq::<FloatElem<B>>(&bboxes.unwrap().to_data(), Tolerance::default());
    }
}
