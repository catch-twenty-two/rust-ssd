use crate::config::VGG_WEIGHTS_FILE;
use burn::nn::Dropout;
use burn::nn::DropoutConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;
use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};

/// SSD uses VGG-16 Type D is used for partial training
///
/// “Very Deep Convolutional Networks for Large-Scale Image Recognition”
/// Authors: Karen Simonyan, Andrew Zisserman
/// Link (official): https://arxiv.org/abs/1409.1556
///
/// Pg. 7
///
/// Base network Our experiments are all based on VGG16 [15], which is pre-trained on the ILSVRC
/// CLS-LOC dataset [16]. Similar to DeepLab-LargeFOV [17], we convert fc6 and fc7 to convolutional
/// layers, subsample parameters from fc6 and fc7, change pool5 from 2 × 2 − s2 to 3 × 3 − s1,
/// and use the à trous algorithm [18] to fill the ”holes”. We remove all the dropout layers
/// and the fc8 layer. We fine-tune the resulting model using SGD with initial learning rate 10−3
/// , 0.9 momentum, 0.0005 weight decay, and batch size 32. The learning rate decay policy is
/// slightly different for each dataset, and we will describe details later. The full training and
/// testing code is built on Caffe [19] and is open source at:
/// https://github.com/weiliu89/caffe/tree/ssd .
///
/// Pg. 3
///
/// Table 1: ConvNet configurations (shown in columns). The depth of the configurations increases
/// from the left (A) to the right (E), as more layers are added (the added layers are shown in
/// bold). The convolutional layer parameters are denoted as “conv(receptive field size)-(number
/// of channels)”. The ReLU activation function is not shown for brevity.
///
///     D
/// -----------
/// 16 weight
///  layers
/// -----------
///  conv3-64
///  conv3-64
///
///  conv3-128
///  conv3-128
///
///  conv3-256
///  conv3-256
///  conv3-256
///
///  conv3-512
///  conv3-512
///  conv3-512
///
///  conv3-512
///  conv3-512
///  conv3-512
///
///   maxpool
///   FC-4096
///   FC-4096
///   FC-1000
///   softmax
///
/// Imported with burn onnx import from:
/// https://github.com/onnx/models/tree/main/validated/vision/classification/vgg
///
#[derive(Module, Debug)]
pub struct VGG16<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    maxpool2d1: MaxPool2d,

    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    maxpool2d2: MaxPool2d,

    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    maxpool2d3: MaxPool2d,

    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    maxpool2d4: MaxPool2d,

    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    pub maxpool2d5: MaxPool2d,

    gemm1: Option<Linear<B>>,
    dropout1: Option<Dropout>,
    gemm2: Option<Linear<B>>,
    dropout2: Option<Dropout>,
    gemm3: Option<Linear<B>>,
}

impl<B: Backend> Default for VGG16<B> {
    fn default() -> Self {
        Self::from_file(VGG_WEIGHTS_FILE, &Default::default())
    }
}

impl<B: Backend> VGG16<B> {
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let record = burn::record::NamedMpkFileRecorder::<FullPrecisionSettings>::new()
            .load(file.into(), device)
            .expect("Couldn't find the VGG 16-12 weights file, please execute the ./scripts/get_models.sh script and try again.");
        Self::new_vgg_ssd(device).load_record(record)
    }
}

impl<B: Backend> VGG16<B> {
    #[allow(unused_variables)]
    pub fn new_vgg16(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d2 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d5 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d3 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d8 = Conv2dConfig::new([256, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d4 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d11 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d5 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let gemm1 = Some(LinearConfig::new(25088, 4096).with_bias(true).init(device));
        let dropout1 = Some(DropoutConfig::new(0.5).init());
        let gemm2 = Some(LinearConfig::new(4096, 4096).with_bias(true).init(device));
        let dropout2 = Some(DropoutConfig::new(0.5).init());
        let gemm3 = Some(LinearConfig::new(4096, 1000).with_bias(true).init(device));
        Self {
            conv2d1,
            conv2d2,
            maxpool2d1,
            conv2d3,
            conv2d4,
            maxpool2d2,
            conv2d5,
            conv2d6,
            conv2d7,
            maxpool2d3,
            conv2d8,
            conv2d9,
            conv2d10,
            maxpool2d4,
            conv2d11,
            conv2d12,
            conv2d13,
            maxpool2d5,
            gemm1,
            dropout1,
            gemm2,
            dropout2,
            gemm3,
        }
    }

    #[allow(unused_variables)]
    pub fn new_vgg_ssd(device: &B::Device) -> Self {
        // 64 out 3x3 (x2) conv1_3 layer
        let conv1_1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv1_2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Max pool 1
        let maxpool2d1 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        // // out -> 128 3x3 (x2) conv2_3 layer
        let conv2_1 = Conv2dConfig::new([64, 128], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv2_2 = Conv2dConfig::new([128, 128], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Max pool 2
        let maxpool2d2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        // out -> 256 3x3 (x3) conv3_3
        let conv3_1 = Conv2dConfig::new([128, 256], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv3_2 = Conv2dConfig::new([256, 256], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv3_3 = Conv2dConfig::new([256, 256], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Maxpool 3
        let maxpool2d3 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        // out -> 512 3x3 (x3) conv4_3
        let conv4_1 = Conv2dConfig::new([256, 512], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv4_2 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        // changed padding here since burn has no ceiling function
        let conv4_3 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .init(device);

        // Maxpool 4
        let maxpool2d4 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        // out -> 512 3x3 (x3) Conv5_3 layer
        let conv5_1 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv5_2 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv5_3 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Maxpool 5
        let maxpool2d5 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        // VGG - Dropout regularization for the first two fully-connected layers (dropout ratio set to 0.5).

        // FC6 - For SSD convert to Conv2d
        // FC-4096
        let fc6 = None;

        // SSD - Remove
        let dropout1 = None;

        // FC7 - For SSD convert to Conv2d
        // FC-4096
        let fc7 = None;

        // // SSD - Remove
        let dropout2 = None;

        // FC8 - We remove all the dropout layers and the fc8 layer.
        // FC-1000 - Classification layer
        let fc8 = None;

        Self {
            conv2d1: conv1_1,
            conv2d2: conv1_2,
            maxpool2d1,
            conv2d3: conv2_1,
            conv2d4: conv2_2,
            maxpool2d2,
            conv2d5: conv3_1,
            conv2d6: conv3_2,
            conv2d7: conv3_3,
            maxpool2d3,
            conv2d8: conv4_1,
            conv2d9: conv4_2,
            conv2d10: conv4_3,
            maxpool2d4,
            conv2d11: conv5_1,
            conv2d12: conv5_2,
            conv2d13: conv5_3,
            maxpool2d5,
            gemm1: fc6,
            dropout1,
            gemm2: fc7,
            dropout2,
            gemm3: fc8,
        }
    }

    pub fn partial_forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        // Convolutional layer 1
        let conv1_1_out = self.conv2d1.forward(input);
        let relu1_1_out = burn::tensor::activation::relu(conv1_1_out);
        let conv1_2_out = self.conv2d2.forward(relu1_1_out);
        let relu1_2_out = burn::tensor::activation::relu(conv1_2_out);

        let maxpool_1_out = self.maxpool2d1.forward(relu1_2_out); // 150x150

        // Convolutional layer 2
        let conv2_1_out = self.conv2d3.forward(maxpool_1_out);
        let relu2_1_out = burn::tensor::activation::relu(conv2_1_out);
        let conv2_2_out = self.conv2d4.forward(relu2_1_out);
        let relu2_2_out = burn::tensor::activation::relu(conv2_2_out);

        let maxpool_2_out = self.maxpool2d2.forward(relu2_2_out); // 75x75

        //  Convolutional layer 3
        let conv3_1_out = self.conv2d5.forward(maxpool_2_out);
        let relu3_1_out = burn::tensor::activation::relu(conv3_1_out);
        let conv3_2_out = self.conv2d6.forward(relu3_1_out);
        let relu3_2_out = burn::tensor::activation::relu(conv3_2_out);
        let conv3_3_out = self.conv2d7.forward(relu3_2_out);
        let relu3_3_out = burn::tensor::activation::relu(conv3_3_out);

        let maxpool_3_out = self.maxpool2d3.forward(relu3_3_out); // 38x38

        // Convolutional layer 4
        let conv4_1_out = self.conv2d8.forward(maxpool_3_out);
        let relu4_1_out = burn::tensor::activation::relu(conv4_1_out);
        let conv4_2_out = self.conv2d9.forward(relu4_1_out);
        let relu4_2_out = burn::tensor::activation::relu(conv4_2_out);
        let conv4_3_out = self.conv2d10.forward(relu4_2_out);
        let conv4_3_out = burn::tensor::activation::relu(conv4_3_out);

        let maxpool_4_out = self.maxpool2d4.forward(conv4_3_out.clone()); // 19x19

        //  Convolutional layer 5
        let conv5_1_out = self.conv2d11.forward(maxpool_4_out.clone());
        let relu5_1_out = burn::tensor::activation::relu(conv5_1_out);
        let conv5_2_out = self.conv2d12.forward(relu5_1_out);
        let relu5_2_out = burn::tensor::activation::relu(conv5_2_out);
        let conv5_3_out = self.conv2d13.forward(relu5_2_out);
        let conv5_3_out = burn::tensor::activation::relu(conv5_3_out);

        (conv4_3_out, conv5_3_out)
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let (_conv_4_3_out, conv_5_3_out) = self.partial_forward(input);

        // Original VGG ouput here wrapped in Option, since this part of the network uses a large
        // amount of memory, and is not needed

        let maxpool_4_out = self.maxpool2d5.forward(conv_5_3_out); // 9x9

        let flatten1_out = {
            let leading_dim = maxpool_4_out.shape().dims[..1].iter().product::<usize>() as i32;
            maxpool_4_out.reshape::<2, _>([leading_dim, -1])
        };

        // fully connected layer 6
        // value: Torch("mat1 and mat2 shapes cannot be multiplied (1x41472 and 25088x4096)
        let fc6_out = self.gemm1.unwrap().forward(flatten1_out);

        let relu14_out = burn::tensor::activation::relu(fc6_out);

        let dropout1_out = self.dropout1.unwrap().forward(relu14_out);

        let flatten2_out = {
            let leading_dim = dropout1_out.shape().dims[..1].iter().product::<usize>() as i32;
            dropout1_out.reshape::<2, _>([leading_dim, -1])
        };

        // fully connected layer 7

        let fc7_out = self.gemm2.unwrap().forward(flatten2_out);
        let relu15_out = burn::tensor::activation::relu(fc7_out);

        let dropout2_out = self.dropout2.unwrap().forward(relu15_out);

        let flatten3_out = {
            let leading_dim = dropout2_out.shape().dims[..1].iter().product::<usize>() as i32;
            dropout2_out.reshape::<2, _>([leading_dim, -1])
        };

        // fully connected layer 8 (for softmax calculation)

        self.gemm3.clone().unwrap().forward(flatten3_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vgg_model_test() {
        let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
        type B = burn::backend::LibTorch;

        B::seed(42);

        let vgg_model: VGG16<B> = VGG16::new_vgg_ssd(&device);

        println!("{}", vgg_model);

        println!("weight = {:#?}", vgg_model.conv2d1.weight.to_data());
    }

    #[test]
    fn vgg_model_forward_test() {
        let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
        type B = burn::backend::LibTorch;

        B::seed(42);

        let vgg_model: VGG16<B> = VGG16::new_vgg16(&device);

        // Image input needs to be 224x224 for original VGG architecture

        let t = Tensor::<B, 4>::ones([1, 3, 224, 224], &device);

        vgg_model.forward(t);
    }

    #[test]
    fn vgg_model_inference_test() {
        let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
        type B = burn::backend::LibTorch;

        B::seed(42);

        let vgg_model: VGG16<B> = VGG16::new_vgg16(&device);

        println!("{}", vgg_model);

        // Image input needs to be 224x224 for original VGG architecture
        let t = Tensor::<B, 4>::ones([1, 3, 224, 224], &device);

        vgg_model.forward(t);
    }
}
