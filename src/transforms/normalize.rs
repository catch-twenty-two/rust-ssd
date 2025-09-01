use crate::boxes::boxes_to_components;

use super::pipeline::Transform;
use burn::{
    prelude::Backend,
    tensor::{Device, Tensor},
};

// ImageNet mean and std values

const MEAN: [f64; 3] = [0.485, 0.456, 0.406];
const STD: [f64; 3] = [0.229, 0.224, 0.225];

#[derive(Clone)]
pub struct ImageNormalizer<B: Backend> {
    pub mean: Tensor<B, 3>,
    pub std: Tensor<B, 3>,
}

impl<B: Backend> ImageNormalizer<B> {
    /// Creates a new normalizer.
    pub fn new(device: &Device<B>) -> Self {
        let mean = Tensor::<B, 1>::from_floats(MEAN, device).reshape([3, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(STD, device).reshape([3, 1, 1]);
        Self { mean, std }
    }

    /// Normalizes the input image according to the ImageNet dataset.
    ///
    /// The input image should be in the range [0, 1].
    /// The output image will be in the range [-1, 1].
    ///
    /// The normalization is done according to the following formula:
    /// `input = (input - mean) / std`
    pub fn normalize(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        (input - self.mean.clone()) / self.std.clone()
    }

    /// Returns a new normalizer on the given device.
    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            mean: self.mean.clone().to_device(device),
            std: self.std.clone().to_device(device),
        }
    }
}

impl<B: Backend> Transform<B> {
    /// Normalizes a 3-channel image tensor and its associated bounding boxes for SSD training.
    ///
    /// This transformation performs two operations:
    /// 1. **Bounding boxes** – Converts absolute coordinates to normalized coordinates
    ///    relative to the image dimensions (`[0.0, 1.0]` range).  
    /// 2. **Image tensor** – Scales pixel values to `[0.0, 1.0]` and applies channel-wise
    ///    normalization on the GPU using the configured device.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the normalized image tensor and
    /// bounding boxes (if present).
    ///
    /// # Notes
    ///
    /// - Bounding boxes are normalized to match the format expected by SSD default boxes.  
    /// - Image normalization is performed on the GPU to improve performance.  
    /// - The operation does not modify the original image in-place; it returns a new pipeline
    ///   state ready for further augmentations or finalization.
    ///
    pub fn normalize(&mut self) -> Self {
        // Normalize the bounding boxes for this image to match the normalized default bounding
        // boxes from SSD
        let [_ch, height, width] = self.image.dims();

        if let Some(bboxes) = self.bboxes.as_mut() {
            let (x1, y1, x2, y2) = boxes_to_components(bboxes.clone());
            let x1 = x1 / width as f32;
            let y1 = y1 / height as f32;
            let x2 = x2 / width as f32;
            let y2 = y2 / height as f32;

            let normalized_bboxes = Tensor::cat(vec![x1, y1, x2, y2], 1);

            self.bboxes = Some(normalized_bboxes);
        }

        self.image = self.image.clone().div_scalar(255.0);

        self.image = ImageNormalizer::new(&self.device)
            .to_device(&self.device)
            .normalize(self.image.clone());

        self.clone()
    }
}
