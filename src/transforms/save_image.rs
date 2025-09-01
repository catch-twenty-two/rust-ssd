use crate::labels::SSDRemapCOCOID;

use super::pipeline::Transform;
use ab_glyph::{FontRef, PxScale};
use burn::{
    prelude::Backend,
    tensor::{cast::ToElement, s},
};
use imageproc::drawing::{draw_hollow_rect, draw_text_mut};

impl<B: Backend> Transform<B> {
    /// Saves the current image tensor to a file and optionally overlays bounding boxes and labels
    /// if present.
    ///
    /// # Parameters
    ///
    /// * `path` – File path where the image will be saved.  
    /// * `lbl_remap` – A remapping utility (`SSDRemapCOCOID`) used to convert label IDs to
    ///   human-readable names for overlaying text.
    ///
    /// # Returns
    ///
    /// A [`Transforms<B, R>`] instance, unchanged, allowing further chaining if desired.
    ///
    /// # Notes
    ///
    /// - The tensor is assumed to be in `[C, H, W]` format
    /// - Pixel values are converted from `f32` to `u8`.
    /// - This operation writes to disk and does not modify the image tensor in memory.  
    /// - The font used for labels is `DejaVuSans.ttf` included in the project.
    ///
    pub fn save_as(&mut self, path: String, lbl_remap: &SSDRemapCOCOID) -> Self {
        let [_ch, height, width] = self.image.dims();

        let t = self.image.clone().permute([1, 2, 0]);

        let buf: Vec<u8> = t
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&p| p as u8)
            .collect();

        let mut image = image::RgbImage::from_vec(width as u32, height as u32, buf).unwrap();

        let font =
            FontRef::try_from_slice(include_bytes!("../../assets/fonts/DejaVuSans.ttf")).unwrap();

        if let Some(bboxes) = self.bboxes.clone() {
            let bboxes = bboxes.split(1, 0);

            for (i, bbox) in bboxes.iter().enumerate() {
                let (x1, y1, x2, y2) = (
                    bbox.clone().slice(s![0, 0]).into_scalar().to_f32(),
                    bbox.clone().slice(s![0, 1]).into_scalar().to_f32(),
                    bbox.clone().slice(s![0, 2]).into_scalar().to_f32(),
                    bbox.clone().slice(s![0, 3]).into_scalar().to_f32(),
                );
                let box_width = x2 - x1;
                let box_height = y2 - y1;

                let rect = imageproc::rect::Rect::at(x1 as i32, y1 as i32)
                    .of_size((box_width) as u32, (box_height) as u32);

                let color = image::Rgb(self.complementary(x1, y1, x2, y2));

                image = draw_hollow_rect(&image, rect, color);

                let label = self
                    .labels
                    .clone()
                    .unwrap()
                    .slice(s![i])
                    .into_scalar()
                    .to_usize();
                let text = lbl_remap.model_id_to_coco_name(&label);

                draw_text_mut(
                    &mut image,
                    color,
                    x1 as i32 + 5,
                    y1 as i32 + 5,
                    PxScale {
                        x: 12.4 * 1.5,
                        y: 12.4,
                    },
                    &font,
                    text,
                );
            }
        }

        image.save(path).unwrap();

        self.clone()
    }
}
