use crate::{
    boxes::boxes_to_components,
    broadcast,
    torch_ops::{linspace, meshgrid},
};

use super::pipeline::Transform;
use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, s},
};
use image::DynamicImage;

fn resize_bboxes<B: Backend>(
    t: &mut Transform<B>,
    new_w: usize,
    new_h: usize,
    image_h: usize,
    image_w: usize,
) {
    if let Some(bboxes) = t.bboxes.as_mut() {
        let (x1, y1, x2, y2) = boxes_to_components(bboxes.clone());
        let h_ratio = new_h as f32 / image_h as f32;
        let w_ratio = new_w as f32 / image_w as f32;

        t.bboxes = Some(
            Tensor::cat(
                vec![x1 * w_ratio, y1 * h_ratio, x2 * w_ratio, y2 * h_ratio],
                1,
            )
            .floor(),
        );
    }
}

impl<B: Backend> Transform<B> {
    /// Resizes a 3-channel image tensor to the specified width and height using triangular
    /// interpolation.
    ///
    /// This transformation changes the spatial dimensions of the image using triangular
    /// interpolation, which provides smooth resizing without GPU acceleration.  
    /// Bounding boxes are scaled proportionally to maintain correct object locations.
    ///
    /// # Parameters
    ///
    /// * `new_w` – Target width of the output image.  
    /// * `new_h` – Target height of the output image.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the resized image tensor and adjusted bounding
    /// boxes (if present).
    ///
    /// # Notes
    ///
    /// - Labels remain unchanged.  
    /// - This method performs resizing on the CPU rather than the GPU.  
    /// - The operation does not modify the original image in-place; it returns a new pipeline
    ///   state ready for further augmentations or finalization.
    ///
    pub fn resize_triangular(&mut self, new_w: usize, new_h: usize) -> Self {
        let [_ch, height, width] = self.image.dims();
        let image = self.image.clone().permute([1, 2, 0]);

        let buf: Vec<u8> = image
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&p| p as u8)
            .collect();

        let image = DynamicImage::from(
            image::RgbImage::from_vec(width as u32, height as u32, buf).unwrap(),
        );

        let image = image
            .resize_exact(
                new_w as u32,
                new_h as u32,
                image::imageops::FilterType::Triangle,
            )
            .to_rgb8();

        self.image = Self::rgb_img_as_tensor(image, &self.device);

        resize_bboxes(self, new_w, new_h, height, width);

        self.clone()
    }

    /// Resizes a 3-channel image tensor to the specified width and height using bilinear
    /// interpolation, and adjusts associated bounding boxes to match the new dimensions.
    ///
    /// This transformation changes the spatial dimensions of the image while preserving
    /// the overall appearance using smooth interpolation between pixel values.  
    /// Bounding boxes are scaled proportionally to maintain correct object locations.
    ///
    /// # Parameters
    ///
    /// * `new_w` – Target width of the output image.  
    /// * `new_h` – Target height of the output image.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance containing the resized image tensor and adjusted bounding
    /// boxes (if present).
    ///
    /// # Notes
    ///
    /// - Labels remain unchanged.  
    /// - The operation does not modify the original image in-place; it returns a new pipeline
    ///   state ready for further augmentations or finalization.
    ///
    pub fn resize_bilinear(&mut self, new_w: usize, new_h: usize) -> Self {
        fn get_quadrant_values<B: Backend>(
            image: &Tensor<B, 3>,
            x_idx: Tensor<B, 2, Int>,
            y_idx: Tensor<B, 2, Int>,
        ) -> Tensor<B, 3> {
            let [new_h, new_w] = y_idx.dims();

            let x_idx = x_idx.float();
            let y_idx = y_idx.float();

            let (image, x_idx) = broadcast!(image: Tensor<B, 3>, x_idx: Tensor<2>);
            let (image, y_idx) = broadcast!(image: Tensor<B, 3>, y_idx: Tensor<2>);

            let idx_slice = y_idx.int().slice(s![0, ..]).flatten(0, 2);

            let mut ch_vals = vec![];

            for i in 0..3 {
                let vals_y = image.clone().slice(s![i, ..]).select(1, idx_slice.clone());

                let idx_slice = x_idx
                    .clone()
                    .int()
                    .slice(s![0, ..])
                    .flatten::<1>(0, 2)
                    .reshape([1, -1, 1]);

                let vals =
                    vals_y
                        .gather(2, idx_slice.clone())
                        .reshape([1, new_h as i32, new_w as i32]);
                let vals = vals.permute([0, 2, 1]).clamp(0.0, 255.0);

                ch_vals.push(vals);
            }

            Tensor::cat(ch_vals, 0)
        }

        let [_ch, image_h, image_w] = self.image.dims();

        if new_h == 0 && new_w == 0 {
            return self.clone();
        }

        if new_h == image_h && new_w == image_w {
            return self.clone();
        }

        let grid_y = linspace::<B>(0.0, (image_h - 1) as f32, new_h, &self.device);
        let grid_x = linspace::<B>(0.0, (image_w - 1) as f32, new_w, &self.device);

        let (grid_x, grid_y) = meshgrid(grid_x, grid_y);

        let x1 = grid_x.clone().floor().int();
        let y1 = grid_y.clone().floor().int();

        let x2 = (x1.clone() + 1).clamp_max(image_w as i32 - 1);
        let y2 = (y1.clone() + 1).clamp_max(image_h as i32 - 1);

        let dx = grid_x - x1.clone().float();
        let dy = grid_y - y1.clone().float();

        let mut resized_image;
        {
            let q11 = get_quadrant_values(&self.image, x1.clone(), y1.clone());
            let w11 = (1.0 - dx.clone()) * (1.0 - dy.clone());
            let (q11, w11) = broadcast!(q11: Tensor<B, 3>, w11: Tensor<2>);

            resized_image = q11 * w11.permute([0, 2, 1]);
        }

        {
            let q12 = get_quadrant_values(&self.image, x1.clone(), y2.clone());
            let w12 = (1.0 - dx.clone()) * dy.clone();
            let (q12, w12) = broadcast!(q12: Tensor<B, 3>, w12: Tensor<2>);

            resized_image = resized_image + q12 * w12.permute([0, 2, 1]);
        }

        {
            let q21 = get_quadrant_values(&self.image, x2.clone(), y1.clone());
            let w21 = dx.clone() * (1.0 - dy.clone());
            let (q21, w21) = broadcast!(q21: Tensor<B, 3>, w21: Tensor<2>);

            resized_image = resized_image + q21 * w21.permute([0, 2, 1]);
        }

        {
            let q22 = get_quadrant_values(&self.image, x2, y2);
            let w22 = dx * dy;
            let (q22, w22) = broadcast!(q22: Tensor<B, 3>, w22: Tensor<2>);

            resized_image = resized_image + q22 * w22.permute([0, 2, 1]);
        }

        resize_bboxes(self, new_w, new_h, image_h, image_w);

        self.image = resized_image;

        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::pipeline::bbox_as_tensor;
    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    use burn::data::dataset::vision::BoundingBox;
    use burn::tensor::{Tolerance, ops::FloatElem};

    #[test]
    fn bilinear_resize_test() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;
        type FT = FloatElem<B>;
        // reshape the image into 2x2 pixel parts

        #[rustfmt::skip]
        let image: Tensor<B, 3> = Tensor::from_data(
            [
                [[10,  20],
                 [30,  40]],

                [[50,  60],
                 [70,  80]],

                [[90,  100],
                 [110, 120]],
            ],
            device,
        );

        let bb = BoundingBox {
            coords: [0.0, 0.0, 1.0, 1.0],
            label: 1,
        };

        let bboxes = bbox_as_tensor::<B>(bb, device);
        let (image, bboxes, _labels) = Transform::from_tensors(image, Some(bboxes), None)
            .resize_bilinear(4, 4)
            .finish()
            .unwrap();

        Tensor::<B, 3>::from_data(
            [
                [
                    [10.00, 13.33, 16.67, 20.00],
                    [16.67, 20.00, 23.33, 26.67],
                    [23.33, 26.67, 30.00, 33.33],
                    [30.00, 33.33, 36.67, 40.00],
                ],
                [
                    [50.00, 53.33, 56.67, 60.00],
                    [56.67, 60.00, 63.33, 66.67],
                    [63.33, 66.67, 70.00, 73.33],
                    [70.00, 73.33, 76.67, 80.00],
                ],
                [
                    [90.00, 93.33, 96.67, 100.00],
                    [96.67, 100.00, 103.33, 106.67],
                    [103.33, 106.67, 110.00, 113.33],
                    [110.00, 113.33, 116.67, 120.00],
                ],
            ],
            device,
        )
        .into_data()
        .assert_approx_eq::<FT>(&image.to_data(), Tolerance::default());

        Tensor::<B, 2>::from_data([[0.00, 0.00, 2.00, 2.00]], device)
            .into_data()
            .assert_approx_eq::<FT>(&bboxes.unwrap().to_data(), Tolerance::default());
    }
}
