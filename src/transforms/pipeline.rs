use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use burn::data::dataset::vision::BoundingBox;
use image::RgbImage;

use rand::{Rng, SeedableRng, rngs::StdRng};

/// Maximum pixel value for a RGB8 pixel
pub const MAX_PIXEL_VAL: f32 = 255.0;

/// A pipeline for performing chained image transformations and augmentations.
///
/// `Transform` wraps an image tensor along with optional bounding boxes and labels,
/// providing a unified interface for common augmentation operations such as:
/// - Photometric adjustments (brightness, contrast, hue)  
/// - Geometric transformations (horizontal/vertical flip, crop, zoom-out, resize)  
/// - Normalization and sanitization of bounding boxes  
/// - Saving images with visualized bounding boxes and labels
///
///
/// # Type Parameters
///
/// * `B` – The backend used for tensor operations, implementing the `Backend` trait.  
/// * `R` – Random number generator used for stochastic augmentations (default: `StdRng`).
///
/// # Fields
///
/// * `image` – The 3-channel image tensor in `[C, H, W]` format.  
/// * `bboxes` – Optional tensor of bounding boxes in `[N, 4]` format (`[x, y, w, h]`).  
/// * `labels` – Optional tensor of class labels for the bounding boxes.  
/// * `device` – Backend device where tensors reside (CPU or GPU).  
/// * `rng` – Random number generator for stochastic transformations.
///
#[derive(Clone, Debug)]
pub struct Transform<B, R = StdRng>
where
    B: Backend,
    R: rand::Rng,
{
    pub image: Tensor<B, 3>,
    pub bboxes: Option<Tensor<B, 2>>,
    pub labels: Option<Tensor<B, 1, Int>>,
    pub device: <B as Backend>::Device,
    pub rng: R,
}

/// Creates a new [`Transform`] with an explicitly provided random number generator (`rng`).
///
/// # Parameters
/// - `image`: A 3D tensor representing the input image. The device of this tensor
///   is captured and stored in the transform.
/// - `bboxes`: An optional 2D tensor of bounding boxes, typically shaped
///   `[num_boxes, 4]` where each row encodes the coordinates of a box.
/// - `labels`: An optional 1D tensor of integer labels corresponding to the bounding boxes.
///   Its length usually matches the number of boxes in `bboxes`.
/// - `rng`: A user-supplied random number generator implementing [`rand::Rng`].
///   This enables deterministic or customized random behavior.
///
/// # Returns
/// A new [`Transform`] instance containing the provided image, bounding boxes,
/// labels, and RNG, with the device inferred from `image`.
///
/// # Examples
/// ```
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let rng = StdRng::seed_from_u64(42);
/// let transform = Transform::new_seeded(image, Some(bboxes), Some(labels), rng);
/// ```
///
/// This ensures reproducible transformations by controlling the RNG seed.
impl<B: Backend, R: rand::Rng> Transform<B, R> {
    pub fn new_seeded(
        image: Tensor<B, 3>,
        bboxes: Option<Tensor<B, 2>>,
        labels: Option<Tensor<B, 1, Int>>,
        rng: R,
    ) -> Self {
        let device = image.device().clone();
        Self {
            image,
            bboxes,
            rng,
            device,
            labels,
        }
    }
}

impl<B: Backend> Transform<B> {
    /// Creates a new instance of A [`Transform`] instance directly from tensors
    /// representing the image, optional bounding boxes, and optional labels.
    ///
    /// # Arguments
    /// * `image` - A 3D tensor representing the image, typically shaped as `[channels, height width]`.
    /// * `bboxes` - An optional 2D tensor of bounding boxes, usually shaped as `[num_boxes, 4]`.
    /// * `labels` - An optional 1D tensor of integer type containing class labels for each
    ///   bounding box.
    ///
    /// # Returns
    ///
    /// A [`Transform`] instance containing the converted tensor image, optional bounding
    /// boxes and labels, the cloned device, and a seeded RNG.
    ///
    /// # Example
    /// ```rust
    /// let image: Tensor<B, 3> = ...;
    /// let bboxes: Option<Tensor<B, 2>> = Some(...);
    /// let labels: Option<Tensor<B, 1, Int>> = Some(...);
    ///
    /// let instance = Transform::from_tensors(image, bboxes, labels);
    /// ```
    ///
    pub fn from_tensors(
        image: Tensor<B, 3>,
        bboxes: Option<Tensor<B, 2>>,
        labels: Option<Tensor<B, 1, Int>>,
    ) -> Self {
        let rng = StdRng::from_os_rng();
        let device = image.device().clone();
        Self {
            rng,
            image,
            bboxes,
            device,
            labels,
        }
    }

    /// Creates a new instance of the struct from an `RgbImage` and optional bounding boxes and
    /// labels.
    ///
    /// # Arguments
    /// * `image` - An `RgbImage` from the `image` crate, representing the raw RGB input image.
    /// * `bboxes` - An optional 2D tensor of bounding boxes, typically shaped as `[num_boxes, 4]`.
    /// * `labels` - An optional 1D tensor of integer class labels corresponding to the bounding
    ///   boxes.
    /// * `device` - A reference to the backend device (e.g., CPU or GPU) where the tensors should
    ///   be allocated.
    ///
    /// # Returns
    /// A [`Transform`] instance containing the converted tensor image, optional bounding
    /// boxes and labels.
    ///
    /// # Example
    /// ```rust
    /// use rand::SeedableRng;
    /// use image::RgbImage;
    ///
    /// let rgb_image: RgbImage = ...;
    /// let bboxes: Option<Tensor<B, 2>> = Some(...);
    /// let labels: Option<Tensor<B, 1, Int>> = Some(...);
    /// let device = Device::default();
    ///
    /// let instance = Transform::new(rgb_image, bboxes, labels, &device);
    /// ```
    ///
    pub fn new(
        image: image::RgbImage,
        bboxes: Option<Tensor<B, 2>>,
        labels: Option<Tensor<B, 1, Int>>,
        device: &<B as Backend>::Device,
    ) -> Self {
        let rng = StdRng::from_os_rng();
        let image = Self::rgb_img_as_tensor(image, device);

        Self {
            rng,
            image,
            bboxes,
            device: device.clone(),
            labels,
        }
    }

    /// Returns a boolean result based on a uniform random probability.
    ///
    /// This function generates a random boolean value, where the probability of returning
    /// `true` is determined by `p`.
    ///
    /// # Arguments
    ///
    /// * `p` – The probability (between 0.0 and 1.0) that the function will return `true`.
    ///
    /// # Returns
    ///
    /// A boolean value: `true` with probability `p`.
    ///
    /// # Remarks
    ///
    /// - If `p` is 0.0, the function will always return `false`.
    /// - If `p` is 1.0, the function will always return `true`.
    /// - Values outside the range [0.0, 1.0] are clamped to this range.
    pub fn should_apply(&mut self, p: f32) -> bool {
        self.rng.random::<f32>() < p.clamp(0.0, 1.0)
    }

    /// Finalizes a chained transformation pipeline and returns the processed data
    /// in its raw form.
    ///
    /// This method consumes the transformation pipeline and produces the final
    /// outputs:
    /// - The transformed image tensor of shape `[3, H, W]`.  
    /// - Optional bounding boxes tensor of shape `[N, 4]`, in `[x1, y1, x2, y2]` format.  
    /// - Optional labels tensor of shape `[N]`, containing integer class identifiers.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing:
    /// - `Ok((image, bboxes, labels))` if the pipeline completes successfully.  
    /// - `Err(String)` if an error occurs during finalization.
    ///
    /// # Notes
    ///
    /// - This is typically called at the end of a chain of augmentations.  
    /// - The returned values are detached from the pipeline, leaving no additional state.  
    /// - Bounding boxes and labels may be `None` if the pipeline was constructed without them.
    ///
    /// # Example
    ///
    /// ```rust
    /// let result = img
    ///     .random_horizontal_flip(0.5)
    ///     .random_zoom_out(0, (1.0, 2.0), 0.5)
    ///     .random_photometric_distort((-0.2, 0.2), (-0.3, 0.3), (-15.0, 15.0), 0.5)
    ///     .finish();
    ///
    /// match result {
    ///     Ok((image, bboxes, labels)) => {
    ///         // use augmented image and metadata
    ///     }
    ///     Err(e) => eprintln!("Pipeline failed: {}", e),
    /// }
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn finish(
        self,
    ) -> Result<
        (
            Tensor<B, 3>,
            Option<Tensor<B, 2>>,
            Option<Tensor<B, 1, Int>>,
        ),
        String,
    > {
        Ok((self.image, self.bboxes, self.labels))
    }

    /// Converts an `image::RgbImage` into a 3-channel tensor in `[C, H, W]` format.
    ///
    /// This function Transform a standard RGB image into a tensor suitable for
    /// GPU or CPU processing in the augmentation pipeline. The pixel values are
    /// converted to `f32` and the channel dimension is moved to the first axis.
    ///
    /// # Parameters
    ///
    /// * `image` – An `image::RgbImage` to convert.  
    /// * `device` – The device where the resulting tensor will be allocated (CPU or GPU).
    ///
    /// # Returns
    ///
    /// A [`Tensor<B, 3>`] representing the image in `[3, H, W]` format, with pixel values as `f32`.
    ///
    /// # Notes
    ///
    /// - The tensor is ready for use in further augmentation or model input pipelines.  
    /// - No normalization or scaling is applied; pixel values remain in the `0–255` range.  
    /// - The channel-first format `[C, H, W]` is standard for deep learning frameworks.
    ///
    pub fn rgb_img_as_tensor(image: image::RgbImage, device: &B::Device) -> Tensor<B, 3> {
        let img_vec = image.clone().into_raw().iter().map(|&p| p as f32).collect();
        Tensor::<B, 3>::from_data(
            TensorData::new(
                img_vec,
                [image.height() as usize, image.width() as usize, 3],
            )
            .convert::<B::FloatElem>(),
            device,
        )
        .permute([2, 0, 1])
    }
}

/// Converts a bounding box into a tensor.
///
/// Takes a `BoundingBox` and converts it into a 1D tensor.
///
/// # Parameters
/// - `bbox`: The bounding box to be converted.
///
/// # Returns
/// An `Option` containing the `[x, y, w, h]` tensor representing the bounding box,
/// or `None` if the bounding box is invalid.
pub fn bbox_as_tensor<B: Backend>(bbox: BoundingBox, device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data([bbox.coords], device)
}

/// Creates an RGB test image with a specified pattern.
///
/// Generates a new image of the given width and height, filling it with the specified
/// RGB pattern.
///
/// # Arguments
///
/// * `width` – The width of the image in pixels.
/// * `height` – The height of the image in pixels.
/// * `pattern` – A 3-element array representing the RGB pattern to fill the image with.
///
/// # Returns
///
/// An `RgbImage` with the specified width, height, and pattern applied to all pixels.
pub fn create_test_image(width: u32, height: u32, pattern: [u8; 3]) -> RgbImage {
    let mut img = RgbImage::new(width, height);
    let img_pattern: image::Rgb<u8> = image::Rgb(pattern);

    for px in img.pixels_mut() {
        *px = img_pattern;
    }

    img
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn test_seeded_random_number_generation() {
        let mut mse = StdRng::seed_from_u64(3);
        let mut test_vec = Vec::<i32>::new();
        let expected_vec = vec![-1513825812, 408920382, -83330236, 1513922966, 612228279];

        for _ in 0..5 {
            test_vec.push(mse.random::<i32>());
        }

        assert_eq!(expected_vec, test_vec);
    }
}
