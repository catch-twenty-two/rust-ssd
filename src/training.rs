use std::io;

use crate::config::CHECKPOINTS_DIR;
use crate::data::BatchType;
use crate::dataset::COCOVersion;
use crate::stats::Stats;
use crate::{config::TrainingConfig, labels::SSDRemapCOCOID};
use crate::{data::SSDBatcher, dataset::COCODataSet, loss::calculate_loss, models::ssd::SSD};
use burn::data::dataset::vision::ImageFolderDataset;
use burn::record::Recorder;
use burn::{
    data::dataloader::DataLoaderBuilder,
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, cast::ToElement},
};

fn create_dir(dir: &str) {
    if std::fs::exists(dir).unwrap() {
        println!("Directory {} exists, remove? (y)", dir);
        let mut response = String::new();

        io::stdin()
            .read_line(&mut response)
            .expect("Failed to read line");

        if response.contains("y") {
            std::fs::remove_dir_all(dir).ok();
        }
    } else {
        std::fs::create_dir_all(dir).ok();
    }
}

pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    device: &B::Device,
    lbl_remap: &SSDRemapCOCOID,
    mut checkpoint_start: usize,
    coco_root: String,
) {
    // Load a model checkpoint if the user has specified to start at a checkpoint other than 0

    let mut model = if checkpoint_start == 0 {
        checkpoint_start = 1;
        create_dir(CHECKPOINTS_DIR);
        SSD::<B>::new(device, None, lbl_remap.count())
    } else {
        let cp_name = format!(
            "{}ssd-checkpoint-{}-{}.mpk",
            CHECKPOINTS_DIR,
            lbl_remap.names().join("-"),
            checkpoint_start
        );

        let record = CompactRecorder::new()
            .load(cp_name.clone().into(), device)
            .unwrap_or_else(|_| panic!("Couldn't find trained model at {}", cp_name));
        println!("Found weights file at {}", cp_name);
        SSD::<B>::new(device, Some(record), lbl_remap.count())
    };

    B::seed(config.seed);

    let mut optim = config.optimizer.init();

    let batcher_train = SSDBatcher::<B>::new(lbl_remap, BatchType::Train);
    let batcher_valid = SSDBatcher::<B::InnerBackend>::new(lbl_remap, BatchType::Test);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .set_device(device.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::coco_ds_train(
            COCOVersion::V2017,
            coco_root.clone(),
        ));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .set_device(device.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::coco_ds_test(
            COCOVersion::V2017,
            coco_root,
        ));

    let mut stats = Stats::new(config.batch_size);

    // Iterate over our training and validation loop for X epochs.
    for epoch in checkpoint_start..config.num_epochs + 1 {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let (class_predictors, box_predictions, outputs) = model.forward(batch.images.clone());

            let (loss, _) = calculate_loss(
                class_predictors.clone(),
                box_predictions.clone(),
                outputs.clone(),
                &batch,
            );

            // loss is an accumulation relative to batch size so divide by this
            let loss = loss / config.batch_size.to_f32();

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);

            stats.update(loss, iteration, "Train".into(), epoch);
        }

        stats.flush();

        model
            .clone()
            .save_file(
                format!(
                    "{CHECKPOINTS_DIR}/ssd-checkpoint-{}-{}",
                    lbl_remap.names().join("-"),
                    epoch
                ),
                &CompactRecorder::new(),
            )
            .expect("Trained model should be saved successfully");

        let m_valid = model.valid();
        let val_cnt = dataloader_test.num_items();

        for (iteration, batch) in dataloader_test.iter().enumerate() {
            let (class_predictors, box_predictions, outputs) =
                m_valid.forward(batch.images.clone());

            let (loss, _targets) = calculate_loss(
                class_predictors.clone(),
                box_predictions.clone(),
                outputs.clone(),
                &batch,
            );

            let loss = loss.div_scalar(config.batch_size.to_f32());

            stats.update(loss, iteration, "Valid".into(), epoch);

            // Validate 20% of the set

            if iteration * config.batch_size > (val_cnt as f32 * 0.2) as usize {
                break;
            }
        }

        stats.flush();
    }
}
