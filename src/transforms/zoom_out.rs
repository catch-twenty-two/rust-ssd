use crate::broadcast;

use super::pipeline::Transform;
use burn::{
    prelude::Backend,
    tensor::{ElementConversion, Tensor},
};
use rand::Rng;

pub trait RandomZoomOut<B: Backend> {
    fn random_zoom_out(&mut self, fill: u8, side_range: (f32, f32), p: f32) -> Self;
}

impl<B: Backend> RandomZoomOut<B> for Transform<B> {
    /// Applies a "zoom out" transformation by randomly padding the image and adjusting
    /// bounding boxes, as described in the *SSD: Single Shot MultiBox Detector* paper.
    ///
    /// This augmentation simulates zooming out by increasing the canvas size and filling
    /// the surrounding space with a constant value. The new canvas size is chosen randomly
    /// within the specified side range, and the transformation is applied with probability `p`.
    ///
    /// # Arguments
    ///
    /// * `fill` – Pixel value used to fill the padded regions (e.g., `0` for black).  
    /// * `side_range` – `(min, max)` range of scaling factors for the output canvas size
    ///   relative to the original image dimensions.  
    /// * `p` – Probability (`0.0 ≤ p ≤ 1.0`) that the zoom-out transformation is applied.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the image after optional zoom-out
    /// and associated metadata.
    ///
    /// # Notes
    ///
    /// - If bounding boxes are present, their coordinates are shifted to match the padded image.  
    /// - If the transformation is skipped (based on probability `p`), the image and
    ///   bounding boxes remain unchanged.  
    /// - This operation increases the spatial context around objects, simulating the effect
    ///   of smaller objects in a larger scene.
    ///
    fn random_zoom_out(&mut self, fill: u8, side_range: (f32, f32), p: f32) -> Self {
        if !self.should_apply(p) {
            return self.clone();
        }

        let image = self.image.clone();

        let [_ch, height, width] = image.dims();

        if side_range.0 < 1.0 || side_range.0 > side_range.1 {
            panic!("Invalid side range provided {:#?}.", side_range);
        }

        let r = self.rng.random_range(side_range.0..=side_range.1);

        let canvas_width = (width as f32 * r) as usize;
        let canvas_height = (height as f32 * r) as usize;

        let r = (self.rng.random::<f32>(), self.rng.random::<f32>());

        let left = ((canvas_width - width) as f32 * r.0) as usize;
        let top = ((canvas_height - height) as f32 * r.1) as usize;
        let right = canvas_width - (left + width);
        let bottom = canvas_height - (top + height);

        // Pad image
        self.image = image.pad(
            (left, right, top, bottom),
            ElementConversion::elem::<f32>(fill as f32),
        );

        // Translate bounding box
        if let Some(bboxes) = self.bboxes.as_mut() {
            let trans = Tensor::<B, 2>::from_data(
                [[left as f32, top as f32, left as f32, top as f32]],
                &self.device,
            );
            let (a, b) = broadcast!(trans: Tensor<B,2>, bboxes: Tensor<2>);

            self.bboxes = Some(a + b);
        }

        self.clone()
    }
}
#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash, Hasher};

    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::transforms::pipeline::bbox_as_tensor;
    use burn::data::dataset::vision::BoundingBox;

    #[test]
    fn random_zoom_test() {
        let device = &NdArrayDevice::default();
        type B = NdArray;

        let image = image::ImageReader::open("./tests/test.png")
            .expect("Error opening image file")
            .decode()
            .unwrap()
            .to_rgb8();
        let mut bboxes = Vec::<Tensor<B, 2>>::new();

        let bb = BoundingBox {
            coords: [29.0, 21.0, 53.0, 46.0],
            label: 0,
        };

        bboxes.push(bbox_as_tensor::<B>(bb, device));

        let bb = BoundingBox {
            coords: [1.0, 2.0, 3.0, 4.0],
            label: 0,
        };

        bboxes.push(bbox_as_tensor::<B>(bb, device));

        let image_t = Transform::rgb_img_as_tensor(image, device);

        let bb_list = Tensor::cat(bboxes, 0);

        let (img_tensor, bboxes, _) =
            Transform::new_seeded(image_t, Some(bb_list), None, StdRng::seed_from_u64(3))
                .random_zoom_out(128, (1.0, 4.0), 1.0)
                .finish()
                .unwrap();
        // Check image test result
        let test_success_hash: u64 = 7417100515516139536;
        let mut h = DefaultHasher::new();
        img_tensor.clone().into_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());

        // Check bounding box translations
        let bboxes = bboxes.unwrap();

        Tensor::<B, 2>::from_data(
            [[44.00, 26.00, 68.00, 51.00], [16.00, 7.00, 18.00, 9.00]],
            device,
        )
        .to_data()
        .assert_eq(&bboxes.clone().to_data(), true);
    }
}
