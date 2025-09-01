use crate::boxes::boxes_to_components;

use super::pipeline::Transform;
use burn::{prelude::Backend, tensor::Tensor};

impl<B: Backend> Transform<B> {
    /// Validates and filters bounding boxes in the image to ensure they are well-formed
    /// and within image boundaries.
    ///
    /// This transformation performs the following checks on each bounding box:
    /// - `x2 > x1` and `y2 > y1` (box has positive width and height).  
    /// - `x2` and `y2` do not exceed the image width and height.  
    /// - Box area is at least 1 pixel.
    ///
    /// - Boxes failing any of these criteria are removed, and associated labels are filtered
    ///   accordingly.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing:
    /// - `Ok(Self)` with only valid bounding boxes remaining.  
    /// - `Err(String)` if no valid bounding boxes remain after filtering.
    ///
    /// # Notes
    ///
    /// - Labels are updated to match the remaining valid bounding boxes.  
    /// - If bounding boxes are already valid, the pipeline is returned unchanged.  
    /// - The operation does not modify the image tensor itself; it only updates bounding boxes and
    ///   labels.  
    /// - This function is useful to clean data before training or further augmentation.
    ///
    pub fn clean_boxes(&mut self) -> Result<Self, String> {
        if let Some(bboxes) = self.bboxes.as_mut() {
            let (x1, y1, x2, y2) = boxes_to_components(bboxes.clone());
            let [_, image_height, image_width] = self.image.dims();

            // (x2 =< x1) || (y2 =< y1) || (x2 => image_width) || (y2 => image_height)

            let mask = x2
                .clone()
                .lower_equal(x1.clone())
                .bool_or(y2.clone().lower_equal(y1.clone()))
                .bool_or(x2.clone().greater_elem(image_width as f32))
                .bool_or(y2.clone().greater_elem(image_height as f32));

            let areas = (x2.clone() - x1.clone()) * (y2.clone() - y1.clone());
            let mask = mask.bool_or(areas.lower_elem(1.0));
            let valid_boxes = mask.bool_not().flatten::<1>(0, 1).nonzero();

            if valid_boxes.is_empty() {
                return Err("List contains no bounding boxes".into());
            }
            // min height/width

            let valid_boxes = Tensor::cat(valid_boxes, 0);

            self.bboxes = Some(bboxes.clone().select(0, valid_boxes.clone()));
            self.labels = Some(self.labels.clone().unwrap().select(0, valid_boxes));
        }

        Ok(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::pipeline::{Transform, create_test_image};
    use burn::{
        backend::{NdArray, ndarray::NdArrayDevice},
        tensor::Int,
    };

    #[test]
    fn sanitize_bounding_boxes_test() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        let image = create_test_image(12, 12, [127, 128, 255]);

        let bboxes = Tensor::<B, 2>::from_floats(
            [
                [1.0, 1.0, 6.0, 6.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 2.0, 1.0, 4.0],
                [2.0, 2.0, 15.0, 4.0],
                [1.0, 1.0, 1.0, 1.0],
                [5.0, 5.0, 8.0, 8.0],
            ],
            device,
        );

        let labels = Tensor::<B, 1, Int>::from_ints([1, 2, 3, 4, 5, 6], device);

        let t = Transform::new(image, Some(bboxes), Some(labels), device);

        let (_, bboxes, labels) = t.clone().clean_boxes().unwrap().finish().unwrap();

        Tensor::<B, 2>::from_data(
            [
                [1.0, 1.0, 6.0, 6.0],
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 5.0, 8.0, 8.0],
            ],
            device,
        )
        .into_data()
        .assert_eq(&bboxes.unwrap().to_data(), false);

        Tensor::<B, 1>::from_data([1, 2, 6], device)
            .into_data()
            .assert_eq(&labels.unwrap().to_data(), false);
    }
}
