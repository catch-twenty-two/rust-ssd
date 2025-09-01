#![allow(unused)]
use crate::config::VGG_WEIGHTS_FILE;
use burn::nn::PaddingConfig2d;
use burn::nn::conv;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2dConfig;

use crate::boxes::generate_all_default_boxes;
use crate::check_nan_1;
use crate::layers::SSDConvLayers;
use crate::models;
use crate::models::ssd_prediction_head::SSDPredictionHead;

use burn::{
    module::Module,
    prelude::*,
    tensor::{Tensor, backend::Backend},
};

use models::vgg::VGG16;

#[derive(Module, Debug)]
pub struct SSD<B: Backend> {
    pub vgg16: VGG16<B>,
    conv_6: Conv2d<B>,
    conv_7: Conv2d<B>,
    conv8_1: Conv2d<B>,
    conv8_2: Conv2d<B>,
    conv9_1: Conv2d<B>,
    conv9_2: Conv2d<B>,
    conv10_1: Conv2d<B>,
    conv10_2: Conv2d<B>,
    conv11_1: Conv2d<B>,
    conv11_2: Conv2d<B>,
    pub ssd_pred_heads: Vec<SSDPredictionHead<B>>,
    cls_cnt: usize,
}

#[allow(clippy::field_reassign_with_default)]
impl<B: Backend> SSD<B> {
    pub fn new(device: &B::Device, record: Option<SSDRecord<B>>, cls_cnt: usize) -> Self {
        let mut vgg_mod: VGG16<B> = match record {
            Some(_) => VGG16::new_vgg_ssd(device),
            None => {
                println!("Training a new model, loading pre-trained VGG16 model weights from {}", VGG_WEIGHTS_FILE);
                VGG16::from_file(VGG_WEIGHTS_FILE, device)
            }
        };

        // Create new layers on top of vgg network

        vgg_mod.maxpool2d5 = MaxPool2dConfig::new([3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([2, 2])
            .init();

        // 19x19 - Conv6: 3x3x1024 (Fig 2, Pg 4) replaces FC6
        let conv_6 = Conv2dConfig::new([512, 1024], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // 19x19 - Conv7: 1x1x1024 (Fig 2, Pg 4) replaces FC7
        let conv_7: Conv2d<B> = Conv2dConfig::new([1024, 1024], [1, 1]).init(device);

        // 10x10 => 5x5 - Conv8_2: 1x1x256/Conv: 3x3x512-s2 (Fig 2, Pg 4)
        let conv8_1: Conv2d<B> = Conv2dConfig::new([1024, 256], [1, 1]).init(device);
        let conv8_2: Conv2d<B> = Conv2dConfig::new([256, 512], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_stride([2, 2])
            .init(device);

        // 5x5 => 3x3 - Conv9_2: 1x1x128/Conv: 3x3x256-s2 (Fig 2, Pg 4)
        let conv9_1: Conv2d<B> = Conv2dConfig::new([512, 128], [1, 1]).init(device);
        let conv9_2: Conv2d<B> = Conv2dConfig::new([128, 256], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_stride([2, 2])
            .init(device);

        // 3x3 => 1x1 - Conv10_2: 1x1x128/Conv: 3x3x256-s1 (Fig 2, Pg 4)
        let conv10_1: Conv2d<B> = Conv2dConfig::new([256, 128], [1, 1]).init(device);
        let conv10_2: Conv2d<B> = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([1, 1])
            .init(device);

        // 1x1- Conv11_2:  1x1x128/Conv: 3x3x256-s1 (Fig 2, Pg 4)
        let conv11_1: Conv2d<B> = Conv2dConfig::new([256, 128], [1, 1]).init(device);
        let conv11_2: Conv2d<B> = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([1, 1])
            .init(device);

        // Create prediction heads for:
        // conv4_3, conv7 (was FC7), conv8_2, conv9_2, conv10_2, conv11_2

        let mut ssd_pred_heads = Vec::new();

        for conv_layer in SSDConvLayers::as_list().iter() {
            ssd_pred_heads.push(SSDPredictionHead::new(device, conv_layer, cls_cnt));
        }

        let ssd = SSD {
            vgg16: vgg_mod,
            conv_6,
            conv_7,
            conv8_1,
            conv8_2,
            conv9_1,
            conv9_2,
            conv10_1,
            conv10_2,
            conv11_1,
            conv11_2,
            ssd_pred_heads,
            cls_cnt,
        };

        match record {
            Some(record) => {
                println!("Loading pretrained SSD model weights...");
                ssd.load_record(record)
            }
            None => ssd,
        }
    }

    /// Performs a forward pass through the SSD backbone, feature layers, and prediction heads.
    ///
    /// # Parameters
    /// - `input`: A 4D tensor of shape `(B, C, H, W)` representing the input batch of images:
    ///   - `B`: Batch size
    ///   - `C`: Number of input channels (e.g., 3 for RGB)
    ///   - `H`, `W`: Image height and width
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. **Class predictions** — `Tensor<B, 3>` with shape `(B, num_boxes, num_classes)`  
    ///    Concatenated classification logits from all prediction heads, one set per default box.
    /// 2. **Box predictions** — `Tensor<B, 3>` with shape `(B, num_boxes, 4)`  
    ///    Concatenated bounding-box regression outputs (4 coordinates per default box).
    /// 3. **Feature maps** — `[Tensor<B, 4>; 6]`  
    ///    The six intermediate feature map outputs from the backbone and extra SSD layers:
    ///    `Conv4_3`, `Conv7`, `Conv8_2`, `Conv9_2`, `Conv10_2`, `Conv11_2`.
    ///
    /// # Description
    /// This method implements the SSD feature extraction and prediction pipeline:
    /// - Runs the input through the VGG16 backbone (partial forward) to obtain early feature maps.
    /// - Passes intermediate outputs through additional convolutional layers (`Conv6`–`Conv11`).
    /// - Collects six feature maps at different spatial resolutions for multi-scale detection.
    /// - For each feature map:
    ///   - Applies the **classification head** to produce per-class logits for each default box.
    ///   - Applies the **bounding-box head** to predict `(cx, cy, w, h)` offsets for each default
    ///     box.
    ///   - Reshapes and permutes outputs into `(B, num_boxes_per_map, num_classes)` for
    ///     classification
    ///     and `(B, num_boxes_per_map, 4)` for regression.
    /// - Concatenates predictions from all feature maps along the `num_boxes` dimension.
    ///
    /// # Notes
    /// - The number of default boxes per feature map is determined by aspect ratio configuration.
    /// - The output `num_boxes` is slightly different from the original SSD paper (`9040` vs
    ///   `8732`)
    ///   due to differences in Burn's pooling layer (39×39 feature map instead of 38×38 for
    ///   `Conv4_3`).
    ///
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>, [Tensor<B, 4>; 6]) {
        let (conv_4_3_38x38_out, conv_5_3_out) = self.vgg16.partial_forward(input);

        // Convolutional layers 6,7 - 19x19
        let conv_6_out = self.conv_6.forward(conv_5_3_out);
        let relu_6_out = burn::tensor::activation::relu(conv_6_out);
        let conv_7_out = self.conv_7.forward(relu_6_out);

        // relu_7_19x19_out 512 -> prediction head
        let conv_7_19x19_out = burn::tensor::activation::relu(conv_7_out);

        // Convolutional layer 8 - 10x10
        let conv_8_1_out = self.conv8_1.forward(conv_7_19x19_out.clone());
        let relu_8_1_out = burn::tensor::activation::relu(conv_8_1_out);
        let conv_8_2_out = self.conv8_2.forward(relu_8_1_out);

        // relu_8_2_10x10_out -> prediction head
        let conv_8_2_10x10_out = burn::tensor::activation::relu(conv_8_2_out);

        // Convolutional layer 9 - 5x5
        let conv_9_1_out = self.conv9_1.forward(conv_8_2_10x10_out.clone());
        let relu_9_1_out = burn::tensor::activation::relu(conv_9_1_out);
        let conv_9_2_out = self.conv9_2.forward(relu_9_1_out);

        // relu_9_2_5x5_out -> prediction head
        let conv_9_2_5x5_out = burn::tensor::activation::relu(conv_9_2_out);

        // Convolutional layer 10 - 3x3
        let conv_10_1_out = self.conv10_1.forward(conv_9_2_5x5_out.clone());
        let relu_10_1_out = burn::tensor::activation::relu(conv_10_1_out);
        let conv_10_2_out = self.conv10_2.forward(relu_10_1_out);

        // relu_10_2_3x3_out -> prediction head
        let conv_10_2_3x3_out = burn::tensor::activation::relu(conv_10_2_out);

        // Convolutional layer 11 - 1x1
        let conv_11_1_out = self.conv11_1.forward(conv_10_2_3x3_out.clone());
        let relu_11_1_out = burn::tensor::activation::relu(conv_11_1_out);
        let conv_11_2_out = self.conv11_2.forward(relu_11_1_out);

        // relu_11_2_1x1_out -> prediction head
        let conv_11_2_1x1_out = burn::tensor::activation::relu(conv_11_2_out);

        // pass each conv layer through a prediction head
        // conv4_3 (was FC6), conv7 (was FC7), conv8_2, conv9_2, conv10_2, conv11_2

        let outputs = [
            conv_4_3_38x38_out,
            conv_7_19x19_out,
            conv_8_2_10x10_out,
            conv_9_2_5x5_out,
            conv_10_2_3x3_out,
            conv_11_2_1x1_out,
        ];

        let mut class_predictors = vec![];
        let mut box_predictors = vec![];
        let mut total = 0;

        for (i, conv_out) in outputs.iter().enumerate() {
            // Get each prediction head
            let box_pred = self.ssd_pred_heads[i].conv_bbox.forward(conv_out.clone());
            let class_pred = self.ssd_pred_heads[i]
                .conv_classifier
                .forward(conv_out.clone());
            total += class_pred.shape().num_elements();

            // Get the shape of the class prediction head
            let [batch_size, _, height, width] = class_pred.shape().dims();

            // Reshape to be the class prediction Tensor to be:
            //
            // Batch Size, (Inferred size), Number Of Classes (21), Prediction Head Height, Prediction Head Width
            //
            // (512, 1024 or 256)
            //
            // This reshape groups a set number of feature outputs with a specific class
            // output and will train this small group of logits from the conv network to identify
            // with the image type classification for this prediction head output

            let class_pred = class_pred.reshape([
                batch_size as i32,
                -1,
                self.cls_cnt as i32,
                height as i32,
                width as i32,
            ]);

            // B = Batch Size
            // A = Anchor Boxes
            // H = Feature Height
            // W = Feature Width
            // N = Number Of Classes
            //
            // Reshape the CLASS prediction output to the following:
            //
            // (B, A * N, H, W) -> (B, H, W, A, N)

            let class_pred = class_pred.permute([0, 3, 4, 1, 2]);

            let class_pred = class_pred.reshape([batch_size as i32, -1, self.cls_cnt as i32]);

            class_predictors.push(class_pred);

            // B = Batch Size
            // A = Anchor Boxes
            // H = Feature Height
            // W = Feature Width
            // 4 = 4 points per anchor box
            //
            // Reshape the BOX prediction output to the following:
            //
            // (B, A * 4, H, W) -> (B, H, W, A, 4)

            let box_pred = box_pred.reshape([
                batch_size as i32,
                -1,
                4, // 4 cooridnates
                height as i32,
                width as i32,
            ]);

            let box_pred = box_pred.permute([0, 3, 4, 1, 2]);
            let box_pred = box_pred.reshape([batch_size as i32, -1, 4]);

            // ready to calculate default box loss from feature maps
            box_predictors.push(box_pred);
        }

        // create default batched tensor shape for prediction heads

        // Shape { dims: [(1), 9040, 21] }
        let class_predictors = Tensor::cat(class_predictors, 1);

        // Shape { dims: [(1), 9040, 4] }
        let box_predictions = Tensor::cat(box_predictors, 1);

        // This is slightly different than the paper (8732) because Burn does not
        // calculate it's maxpooling using ceiling so a 300x300 image ends up as a
        // 39x39 feature map in the first layer, rather than 38x38

        (class_predictors, box_predictions, outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::backend::NdArray;

    #[test]
    fn ssd_model_test() {
        let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
        type B = burn::backend::LibTorch;
        let ssd_model: SSD<B> = SSD::new(&device, None, 21);
        println!("{}", ssd_model);
    }

    #[test]
    fn ssd_model_forward_test() {
        let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
        type B = burn::backend::LibTorch;
        let ssd_model: SSD<B> = SSD::new(&device, None, 21);
        let t = Tensor::<B, 4>::ones([1, 3, 300, 300], &device);
        ssd_model.forward(t);
    }

    #[test]
    fn ssd_model_backwards_test_w_weights() {
        type B = burn::backend::LibTorch;
        type ADB = Autodiff<B>;
        let device = &burn::backend::libtorch::LibTorchDevice::Cuda(0);
        let ssd_model: models::ssd::SSD<ADB> = SSD::new(device, None, 21);
        let t = Tensor::ones([1, 3, 300, 300], device);
        let (a, b, c) = ssd_model.forward(t);
        println!("model = {}", ssd_model);
        println!("{:#?}", a);
        let gradients = a.backward();
    }
}
