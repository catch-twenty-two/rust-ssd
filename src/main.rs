use burn::{backend::Autodiff, backend::libtorch::LibTorch, config::Config};
use rust_ssd::{
    config::{Commands, SSDCmd, TrainingConfig},
    inference,
    labels::SSDRemapCOCOID,
    training,
};

fn main() {
    type AutoDiffBackend = Autodiff<LibTorch>;
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);

    let cli_cmd: SSDCmd = argh::from_env();
    let coco_remap = SSDRemapCOCOID::new(cli_cmd.o.split(',').collect());

    match cli_cmd.commands {
        Commands::Infer(sub_command_infer) => {
            let image_path = &sub_command_infer.p;
            let model_path = &sub_command_infer.m;
            let iou_overlap_thresh = &sub_command_infer.i.unwrap_or(0.7);
            let cls_conf_thresh = &sub_command_infer.c.unwrap_or(0.5);

            inference::infer::<LibTorch>(
                image_path,
                model_path,
                &coco_remap,
                &device,
                iou_overlap_thresh,
                cls_conf_thresh,
            );
        }
        Commands::Train(sub_command_train) => {
            let checkpoint = sub_command_train.c.unwrap_or(0);
            let coco_root = sub_command_train.r;
            let config = TrainingConfig::load("./config/training_config.json").unwrap();
            training::train::<AutoDiffBackend>(config, &device, &coco_remap, checkpoint, coco_root);
        }
    };
}
