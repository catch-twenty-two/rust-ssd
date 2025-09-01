use burn::{config::Config, optim::SgdConfig};
use {argh::FromArgs, std::fmt::Debug};

pub const VGG_WEIGHTS_FILE: &str = "./assets/pretrained_models/vgg16-12";
pub const CHECKPOINTS_DIR: &str = "./artifacts/checkpoints/";
pub const LOG_PATH: &str = "./artifacts/log.txt";
pub const WIDTH: usize = 300;
pub const HEIGHT: usize = 300;

/// “SSD: Single Shot MultiBox Detector”
/// Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
///          Scott Reed, Cheng-Yang Fu, Alexander C. Berg
/// Link (official): https://arxiv.org/abs/1512.02325
///
/// Default boxes and aspect ratios - Section 3.4
///
/// To further validate the SSD framework, we trained our SSD300 and SSD512 architec-
/// tures on the COCO dataset. Since objects in COCO tend to be smaller than PASCAL
/// VOC, we use smaller default boxes for all layers. We follow the strategy mentioned in
/// Sec. 2.2, but now our smallest default box has a scale of 0.15 instead of 0.2, and the
/// scale of the default box on conv4 3 is 0.07 (e.g. 21 pixels for a 300 × 300 image)5 .
/// We use the trainval35k [24] for training. We first train the model with 10−3
/// learning rate for 160k iterations, and then continue training for 40k iterations with
/// 10−4 and 40k iterations with 10−5.
///
#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: SgdConfig,
    #[config(default = 100)]
    pub num_epochs: usize,
    #[config(default = 2)]
    pub batch_size: usize,
    #[config(default = 7)]
    pub num_workers: usize,
    #[config(default = 1)]
    pub seed: u64,
    #[config(default = 0.001)]
    pub learning_rate: f64,
}

#[derive(FromArgs, PartialEq, Debug)]
/// Top-level command.
pub struct SSDCmd {
    #[argh(subcommand)]
    pub commands: Commands,
    #[argh(option)]
    /// object names to learn from e.g 'dog,cat,person'
    pub o: String,
}

#[derive(FromArgs, PartialEq, Debug)]
#[argh(subcommand)]
pub enum Commands {
    Infer(SubCommandInfer),
    Train(SubCommandTrain),
}

#[derive(FromArgs, PartialEq, Debug)]
/// Train an SSD model using the COCO dataset
#[argh(subcommand, name = "infer")]
pub struct SubCommandInfer {
    #[argh(option)]
    /// image path to run inference on
    pub p: String,
    #[argh(option)]
    /// model file path to use
    pub m: String,
    #[argh(option)]
    /// iou overlap - how much do predicted boxes needs to overlap to be considered a single box
    pub i: Option<f32>,
    #[argh(option)]
    /// classification confidence level score
    pub c: Option<f32>,
}

#[derive(FromArgs, PartialEq, Debug)]
/// Train an SSD model using the COCO dataset
#[argh(subcommand, name = "train")]
pub struct SubCommandTrain {
    #[argh(option)]
    /// COCO dataset root location (Ex: ./root/annotations/instances_train2017.json)
    pub r: String,
    #[argh(option)]
    /// checkpoint number to start training from default None
    pub c: Option<usize>,
}
