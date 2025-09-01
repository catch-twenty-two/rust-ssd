use crate::{transforms::pipeline::MAX_PIXEL_VAL};

use super::pipeline::Transform;
use burn::{
    prelude::Backend,
    tensor::{Tensor, cast::ToElement, s},
};

use rand::Rng;

impl<B: Backend> Transform<B> {
    /// Adjusts the brightness of a 3-channel RGB image tensor by adding a scalar value to all
    /// pixels.
    ///
    /// This transformation increases or decreases the overall brightness of the image
    /// by adding the specified `value` to each pixel.
    ///
    /// # Parameters
    ///
    /// * `value` – Brightness adjustment value:  
    ///     - Positive values brighten the image.  
    ///     - Negative values darken the image.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the image tensor with adjusted brightness.
    ///
    /// # Notes
    ///
    /// - The operation modifies only pixel intensities; spatial layout and bounding boxes remain
    ///   unchanged.  
    /// - The transformation does not modify the original image in-place; it returns a new pipeline
    ///   state ready for further augmentations or finalization.
    ///
    pub fn brightness(&mut self, value: i32) -> Self {
        self.image = self
            .image
            .clone()
            .add_scalar(value as f32)
            .clamp(0.0, super::pipeline::MAX_PIXEL_VAL);

        self.clone()
    }

    /// Computes the average complementary color for a given section of the image. This mainly can
    /// be used for better contrast on overlays.
    /// 
    /// # Returns
    /// A `([u8; 3] array representing the sections average complementary value
    /// 
    pub fn complementary(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> [u8; 3] {
        let image_section = self.crop_area(x1 as usize, y1 as usize, x2 as usize, y2 as usize);
        let r = image_section.clone().slice(s![0]);
        let g = image_section.clone().slice(s![1]);
        let b = image_section.clone().slice(s![2]);

        let r_ave =
            (r.clone().sum().into_scalar().to_f64() / r.shape().num_elements().to_f64()).to_f32();
        let g_ave =
            (g.clone().sum().into_scalar().to_f64() / g.shape().num_elements().to_f64()).to_f32();
        let b_ave =
            (b.clone().sum().into_scalar().to_f64() / b.shape().num_elements().to_f64()).to_f32();

        [255 - r_ave as u8, 255 - b_ave as u8, 255 - g_ave as u8]
    }

    /// Adjusts the contrast of a 3-channel image tensor by a specified percentage.
    ///
    /// This transformation scales pixel intensity differences to increase or decrease
    /// the contrast of the image.
    ///
    /// # Type Parameters
    ///
    /// * `B` – The backend used by the tensor, implementing the `Backend` trait.
    ///
    /// # Parameters
    ///
    /// * `contrast` – A `f32` value representing the percentage of contrast adjustment:  
    ///     - `0.0` leaves the image unchanged.  
    ///     - Positive values increase contrast (e.g., `20.0` increases by 20%).  
    ///     - Negative values decrease contrast (e.g., `-20.0` decreases by 20%).  
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the image tensor with adjusted contrast.
    ///
    /// # Notes
    ///
    /// - This operation modifies only pixel intensities; spatial layout and bounding boxes remain
    ///   unchanged.  
    /// - The transformation does not modify the original image in-place; it returns a new
    ///   pipeline state
    ///   ready for further augmentations or finalization.
    ///
    pub fn contrast(&mut self, contrast: f32) -> Self {
        let percent = ((100.0 + contrast) / 100.0).powi(2);
        self.image = self
            .image
            .clone()
            .div_scalar(MAX_PIXEL_VAL)
            .sub_scalar(0.5)
            .mul_scalar(percent)
            .add_scalar(0.5)
            .mul_scalar(MAX_PIXEL_VAL)
            .clamp(0.0, MAX_PIXEL_VAL);

        self.clone()
    }

    /// Rotates the hue of a 3-channel RGB image tensor.
    ///
    /// This transformation applies a hue shift using a rotation matrix derived from the
    /// specified angle. The operation is performed directly in RGB space without converting
    /// to HSV.
    ///
    /// # Parameters
    ///
    /// * `angle` – Hue rotation angle in degrees:  
    ///     - Positive values rotate clockwise.  
    ///     - Negative values rotate counter-clockwise.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the image tensor with adjusted hue.
    ///
    /// # Notes
    ///
    /// - This transformation modifies only color values; spatial layout and bounding boxes remain
    ///   unchanged.  
    /// - The operation does not modify the original image in-place; it returns a new pipeline state
    ///   ready for further augmentations or finalization.
    ///
    pub fn hue_rotate(&mut self, angle: f32) -> Self {
        let cosv = angle.to_radians().cos();
        let sinv = angle.to_radians().sin();

        let coeffs: [f32; 9] = [
            // Reds
            0.213 + cosv * 0.787 - sinv * 0.213,
            0.715 - cosv * 0.715 - sinv * 0.715,
            0.072 - cosv * 0.072 + sinv * 0.928,
            // Greens
            0.213 - cosv * 0.213 + sinv * 0.143,
            0.715 + cosv * 0.285 + sinv * 0.140,
            0.072 - cosv * 0.072 - sinv * 0.283,
            // Blues
            0.213 - cosv * 0.213 - sinv * 0.787,
            0.715 - cosv * 0.715 + sinv * 0.715,
            0.072 + cosv * 0.928 + sinv * 0.072,
        ];

        let chunks = self.image.clone().split(1, 0);

        let red = chunks[0]
            .clone()
            .mul_scalar(coeffs[0])
            .add(chunks[1].clone().mul_scalar(coeffs[1]))
            .add(chunks[2].clone().mul_scalar(coeffs[2]));

        let green = chunks[0]
            .clone()
            .mul_scalar(coeffs[3])
            .add(chunks[1].clone().mul_scalar(coeffs[4]))
            .add(chunks[2].clone().mul_scalar(coeffs[5]));

        let blue = chunks[0]
            .clone()
            .mul_scalar(coeffs[6])
            .add(chunks[1].clone().mul_scalar(coeffs[7]))
            .add(chunks[2].clone().mul_scalar(coeffs[8]));

        self.image = Tensor::cat(vec![red, green, blue], 0).clamp(0.0, MAX_PIXEL_VAL);

        self.clone()
    }

    /// Applies random photometric distortions to a 3-channel image tensor, following
    /// the augmentation strategy described in *SSD: Single Shot MultiBox Detector*.
    ///
    /// Each distortion type (brightness, contrast, hue):
    /// - Is applied independently with probability `p`.
    /// - Uses a factor sampled uniformly from the specified range.
    /// - Leaves the image unchanged for that attribute if not applied.
    ///
    /// # Arguments
    ///
    /// * `brightness` – `(min, max)` adjustment range for brightness.  
    ///   The factor is sampled uniformly from this range and added to pixel values.  
    ///   Range must satisfy `-1.0 ≤ min ≤ max ≤ 1.0`.
    ///
    /// * `contrast` – `(min, max)` adjustment range for contrast scaling.  
    ///   The factor is sampled uniformly from this range and used to scale pixel values.  
    ///   Range must satisfy `-1.0 ≤ min ≤ max ≤ 1.0`.
    ///
    /// * `hue` – `(min, max)` adjustment range for hue rotation in degrees.  
    ///   The factor is sampled uniformly from this range and applied as a shift in hue space.  
    ///   Range must satisfy `-180.0 ≤ min ≤ max ≤ 180.0`.
    ///
    /// * `p` – Probability (`0.0 ≤ p ≤ 1.0`) that each transformation is applied independently.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the image with distortions applied
    /// and associated metadata.
    ///
    /// # Notes
    ///
    /// - Bounding boxes and labels are unaffected by photometric distortions.  
    /// - Transformations are stochastic; some may be applied while others are skipped.  
    /// - This randomness increases data variability for training object detection models.
    ///
    /// # Example
    ///
    /// ```rust
    /// let t = img.random_photometric_distort(
    ///     (-0.2, 0.2),   // brightness adjustment
    ///     (-0.3, 0.3),   // contrast adjustment
    ///     (-70.0, 70.0), // hue adjustment
    ///     0.5,           // probability per transform
    /// );
    ///
    /// let augmented_img = t.image;
    /// ```
    pub fn random_photometric_distort(
        &mut self,
        brightness: (f32, f32),
        contrast: (f32, f32),
        hue: (f32, f32),
        p: f32,
    ) -> Self {
        if self.should_apply(p) {
            let r_bright = (self
                .rng
                .random_range(brightness.0.clamp(-1.0, 1.0)..brightness.1.clamp(-1.0, 1.0))
                * super::pipeline::MAX_PIXEL_VAL) as i32;
            self.brightness(r_bright);
        }

        if self.should_apply(p) {
            let r_contrast = self
                .rng
                .random_range(contrast.0.clamp(-1.0, 1.0)..contrast.1.clamp(-1.0, 1.0))
                * 100.0;
            self.contrast(r_contrast);
        }

        if self.should_apply(p) {
            let r_hue_rot = self
                .rng
                .random_range(hue.0.clamp(-1.0, 1.0)..hue.1.clamp(-1.0, 1.0) * 180.0);

            self.hue_rotate(r_hue_rot);
        }

        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::transforms::pipeline::create_test_image;
    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    use rand::{SeedableRng, rngs::StdRng};
    use std::hash::{DefaultHasher, Hash, Hasher};

    use super::*;

    #[test]
    fn brightness_test() {
        let image = create_test_image(12, 12, [127, 128, 255]);
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;
        let image = Transform::rgb_img_as_tensor(image, device);
        let mut t = Transform::<B>::new_seeded(image, None, None, StdRng::seed_from_u64(3));

        let (image, _, _) = t.brightness(4).finish().unwrap();

        // Test hash of image
        let test_success_hash: u64 = 2115067528597659219;
        let mut h = DefaultHasher::new();
        image.to_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());
    }

    #[test]
    fn hue_rotate_test() {
        let image = create_test_image(12, 12, [127, 128, 100]);
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;
        let image = Transform::rgb_img_as_tensor(image, device);
        let mut t = Transform::<B>::new_seeded(image, None, None, StdRng::seed_from_u64(3));

        let (image, _, _) = t.hue_rotate(180.0).finish().unwrap();

        // Test hash of image
        let test_success_hash: u64 = 8485462149801660459;
        let mut h = DefaultHasher::new();
        image.to_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());
    }
}
