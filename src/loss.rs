use burn::nn::loss::HuberLossConfig;
use burn::tensor::Tensor;
use burn::tensor::cast::ToElement;

use crate::boxes::{
    box_regression, generate_all_default_boxes, get_overlapping_dflt_boxes,
    target_lbls_to_default_boxes,
};
use crate::data::{SSDBatch, strip_padding};

use burn::prelude::*;

/// Computes the cross-entropy loss for multi-class classification without requiring one-hot
/// encoding similar to how torch cross entropy works.
///
/// # Arguments
/// * `logits` - A 2D tensor of shape `[num_boxes, num_classes]` representing the predicted raw
///   scores
///   (logits) for each class.
/// * `targets` - A 1D tensor of shape `[num_boxes]` containing the integer class labels for each
///   prediction. Each label should be in the range `[0, num_classes - 1]`.
///
/// # Returns
/// A 1D tensor of shape `[num_boxes]` containing the cross-entropy loss for each prediction.
///
/// # Details
/// This function applies the log-softmax to the predictions and then selects the predicted
/// probabilities corresponding to the true class labels using indexing (similar to PyTorch's
/// approach). The negative log-likelihood is computed for each prediction, producing the per-box
/// loss without requiring one-hot encoded labels.
///
/// Further reading:
///
/// Lau, R. (2025, March 5). Cross-Entropy, negative Log-Likelihood, and all that jazz.
///  Towards Data Science.
/// https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81/
///
/// Negative Log-Likelihood. (n.d.).
/// Notes by Lex.
/// https://notesbylex.com/negative-log-likelihood.html
///
fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 2>,       // (predictions) [# of boxes, # Classes]
    targets: Tensor<B, 1, Int>, // (labels, horse, dog, cat, ect..) [# of boxes]
) -> Tensor<B, 1> {
    let [box_count] = targets.dims();

    // Apply log_softmax along the class dimension and convert logits into log-probabilities,
    // which is required by negative log-likelihood.
    //
    //                        ⎛  exp ⎛x ⎞  ⎞
    //                        ⎜      ⎝ i⎠  ⎟
    //  log(softmax(x)) = log ⎜────────────⎟
    //               i        ⎜___         ⎟
    //                        ⎜╲   exp ⎛x ⎞⎟
    //                        ⎜╱       ⎝ i⎠⎟
    //                        ⎝‾‾‾         ⎠

    let log_probabilities = burn::tensor::activation::log_softmax(logits, 1);
    let targets = targets.clone().reshape([box_count, 1]);

    // Calculate loss or negative log likelyhood (nll) from the probabilities
    //
    // loss  = -log ⎛P ⎛y , x ⎞⎞
    //     i        ⎝  ⎝ i   i⎠⎠
    //
    let nll = log_probabilities.gather(1, targets) * -1;

    nll.reshape([box_count])
}

/// Computes the SSD loss for a batch of predictions.
///
/// # Arguments
/// - `class_logits`: Predicted class scores for all default boxes.
/// - `bbox_logits`: Predicted bounding box offsets for all default boxes.
/// - `layer_outputs`: Feature map outputs from each SSD detection layer.
/// - `ssd_batch`: Batched images, ground-truth boxes, labels, and padding info.
///
/// # Returns
/// A tuple containing:
/// 1. `Tensor<B, 2>`: Localization (bounding regression box) + confidence loss for each batch
///    element.
/// 2. `Tensor<B, 2, Int>`: Target class assignments for each default box in the batch.
///
pub fn calculate_loss<B: Backend>(
    class_logits: Tensor<B, 3>, // Shape { dims: [1, 9040, 21] }
    bbox_logits: Tensor<B, 3>,  // Shape { dims: [1, 9040, 4] }
    layer_outputs: [Tensor<B, 4>; 6],
    ssd_batch: &SSDBatch<B>,
) -> (Tensor<B, 2>, Tensor<B, 2, Int>) {
    let device = &class_logits.device();

    // Create all the necessary default boxes, this is governed by the shape of the layer outputs
    let dflt_bxs = generate_all_default_boxes(layer_outputs);

    let gt_boxes = ssd_batch.gt_boxes.clone();

    let [batch_size, _, _] = gt_boxes.shape().dims();

    let mut loss_b = vec![];
    let mut class_targets_b = vec![];

    for i in 0..batch_size {
        let gt_boxes: Tensor<B, 2> = gt_boxes.clone().slice(i).squeeze(0);
        let target_labels = ssd_batch.target_labels.clone().slice(i).squeeze(0);
        let target_padding: Tensor<B, 1, Int> =
            ssd_batch.target_padding.clone().slice(i).squeeze(0);

        let dflt_bxs: Tensor<B, 2> = dflt_bxs.clone().slice(i).squeeze(0);
        let bbox_logits: Tensor<B, 2> = bbox_logits.clone().slice(i).squeeze(0);
        let class_logits: Tensor<B, 2> = class_logits.clone().slice(i).squeeze(0);

        // strip the padding and process each tensor separately
        let (target_labels, gt_boxes) = strip_padding(gt_boxes, target_labels, target_padding);

        // Find the default boxes that overlap with the ground truth boxes
        let matching_dflt_boxes =
            get_overlapping_dflt_boxes(gt_boxes.clone(), dflt_bxs.clone(), 0.5);

        // Get the index of the elements that are matches by filtering out the -1 (non-matching) element values

        let dflt_indexes = matching_dflt_boxes
            .clone()
            .add_scalar(1)
            .bool()
            .clone()
            .nonzero()[0]
            .clone();

        // Select the ground truth boxes (x1,y1,x2,y2) and default boxes that overlap with each
        // other as an index of values
        let gt_bx_matches = gt_boxes.clone().select(
            0,
            matching_dflt_boxes.clone().select(0, dflt_indexes.clone()),
        );

        let dflt_bx_matches = dflt_bxs.clone().select(0, dflt_indexes.clone());

        // Get the transformation that the box predictor should use in order to translate it's
        // proposal to a ground truth box
        let box_trans = box_regression(gt_bx_matches, dflt_bx_matches, (10.0, 10.0, 5.0, 5.0));

        // Select predicted box matches that are  regressing in order to be a match with a default
        // box when the object is detected in the image
        let pred_boxes_trans: Tensor<B, 2> = bbox_logits.select(0, dflt_indexes.clone());

        // The predicted boxes contain the translations needed to make the default box align with
        // the actual detected object

        // Get bounding box loss using L1 smoothing
        //
        // The loss calculated from the prediction and where the prediction should translated
        // the existing ground truth box
        let pred_box_loss = HuberLossConfig::new(0.5)
            .init()
            .forward_no_reduction(pred_boxes_trans.clone(), box_trans.clone());

        // Creates a tensor of all the labels for all default bounding boxes (8732 in the case of
        // the original SSD paper) and assigns each one a 0 for a background label, or a class
        // type (0-21 for the original paper). These will be used to  create a loss value using
        // cross entropy loss. Since each default box can only contain 1 type of classification
        // they are assigned a classification id copied from the ground truth box it was matched
        // up with. Like the default boxes, each ground truth box can only contain 1
        // classification label. For both types, boxes considered to be in the background (no
        // match) are assigned a 0 and boxes considered to be in the foreground are assigned an
        // index from 1 to the number of classes that are being trained on (21 in the case of the
        // original paper) which represents a mapping to an actual object name.
        //
        // See here: file://./coco_labels.rs

        let cls_boxes =
            target_lbls_to_default_boxes(&matching_dflt_boxes, target_labels.clone(), dflt_bxs);

        let conf_loss = cross_entropy_loss(class_logits.clone(), cls_boxes.clone());

        // Hard negative mining - Pg. 6 Liu et al.
        //
        // After the matching step, most of the default boxes are negatives, especially when the
        // number of possible default boxes is large. This introduces a significant imbalance
        // between the positive and negative training examples. Instead of using all the negative
        // examples, we sort them using the highest confidence loss for each default box and pick
        // the top ones so that the ratio between the negatives and positives is at most 3:1. We
        // found that this leads to faster optimization and a more stable training.

        let fg_label_mask = cls_boxes.clone().greater_elem(0);

        let neg_pos_ratio = 3;

        // Get the number of background predictions to hard negative mine from

        let hard_neg_cnt_max =
            fg_label_mask.clone().int().sum().into_scalar().to_i32() * neg_pos_ratio;

        // Get the index of all labeled classes and which default box they ended up in
        // this function requires cat because nonzero returns a vector of tensor booleans
        let fg_index_map = Tensor::cat(fg_label_mask.nonzero(), 0);

        // Set the labeled box positions from the loss to NEG INIFINTY so during the sort
        // the only indexes that show up at the top of the sort are background values
        let hrd_neg_loss = conf_loss.clone().select_assign(
            0,
            fg_index_map.clone(),
            Tensor::full(fg_index_map.shape(), f32::NEG_INFINITY, device),
        );

        let (_vals, bg_index_map) = hrd_neg_loss.sort_descending_with_indices(0);
        let bg_index_map = bg_index_map.slice(0..hard_neg_cnt_max).clone();

        // “SSD: Single Shot MultiBox Detector”
        // Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
        //          Scott Reed, Cheng-Yang Fu, Alexander C. Berg
        // Link (official): https://arxiv.org/abs/1512.02325
        //
        // Calculating the total Loss
        //
        // L1 (Huber Loss) is utilized for regression of the boxes, Cross Entropy Loss for the
        // Multi-Label Classification.
        //
        // Training objective - Pg. 5 Liu et al.
        //
        // The SSD training objective is derived from the MultiBox objective [7,8] but is extended
        // to handle multiple object categories. The overall objective loss function is a weighted
        // sum of the localization loss (loc) and the confidence loss (conf):
        //
        //                 ⎛1⎞
        // L(x, c, l, g) = ⎜─⎟ ⋅ (Lconf(x, c)) + α ⋅ Lloc(x, l, g))
        //                 ⎝N⎠
        //
        // x: input (e.g., default/prior box assignments or matching indicator)
        // c: predicted class scores
        // l: predicted box locations
        // g: ground-truth box locations
        // L(oss)conf​: classification (confidence) loss
        // L(oss)loc​: localization loss (e.g., smooth L1​ between l and g)
        // α: weighting factor balancing localization vs. confidence
        // N: number of matched (positive) examples, used to normalize the loss

        // where N is the number of matched default boxes. If N = 0, we set the loss to 0. The
        // localization loss is a Smooth L1 loss between the predicted box (l) and the ground
        // truth box (g) parameters. Similar to Faster R-CNN [2], we regress to offsets for the
        // center (cx, cy) of the default bounding box (d) and for its width (w) and height (h).

        let n = dflt_indexes.shape().num_elements().to_i32();

        let loss = if n >= 1 {
            let alpha: f32 = 1.0;

            let lconf = conf_loss.clone().select(0, fg_index_map).sum()
                + conf_loss.select(0, bg_index_map.clone()).sum();

            let lloc = pred_box_loss.clone().sum();

            (lconf + alpha * lloc) / n
        } else {
            Tensor::zeros([1], device)
        };

        loss_b.push(loss);
        class_targets_b.push(cls_boxes);
    }

    (Tensor::stack(loss_b, 0), Tensor::stack(class_targets_b, 0))
}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::boxes::box_regression;

    use super::*;
    use burn::{
        backend::Autodiff,
        tensor::{Distribution, Int, Shape, Tolerance, ops::FloatElem},
    };

    #[test]
    fn test_loss() {
        let batches = 2;
        type B = Autodiff<burn::backend::LibTorch>;
        let device = &burn::backend::libtorch::LibTorchDevice::Cuda(0);

        B::seed(42);

        let outputs = [
            Tensor::<B, 4>::ones(Shape::new([batches, 512, 38, 38]), device), // VGG-16 through Conv5_3 layer
            Tensor::<B, 4>::ones(Shape::new([batches, 1024, 19, 19]), device), // Conv6 (FC7)
            Tensor::<B, 4>::ones(Shape::new([batches, 512, 10, 10]), device), // Conv8_2
            Tensor::<B, 4>::ones(Shape::new([batches, 256, 5, 5]), device),   // Conv9_2
            Tensor::<B, 4>::ones(Shape::new([batches, 256, 3, 3]), device),   // Conv10_2
            Tensor::<B, 4>::ones(Shape::new([batches, 256, 1, 1]), device),   // Conv11_2
        ];

        let target_image = Tensor::<B, 4>::ones([batches, 3, 300, 300], device);

        // labels are indexes into the name, (11 = cat, 12 = dog, 23 = horse)
        let target_labels = Tensor::<B, 1, Int>::from_data([11, 12, 20], device);
        let target_padding = Tensor::<B, 1, Int>::from_data([0], device);
        let gt_boxes = Tensor::<B, 2>::from_data(
            [
                [0.12, 0.15, 0.30, 0.40], // 0
                [0.05, 0.05, 0.25, 0.20], // 1
                [0.33, 0.20, 0.50, 0.45], // 2
            ],
            device,
        );

        let class_predictors: Tensor<B, 3> = Tensor::random(
            [batches, 8732, 21],
            Distribution::Uniform(-1.0, 1.0),
            device,
        );
        let box_predictors: Tensor<B, 3> =
            Tensor::random([batches, 8732, 4], Distribution::Uniform(-1.0, 1.0), device);

        let ssd_batch = SSDBatch {
            images: target_image,
            gt_boxes: Tensor::stack(vec![gt_boxes.clone(), gt_boxes], 0),
            target_labels: Tensor::stack(vec![target_labels.clone(), target_labels], 0),
            target_padding: Tensor::stack(vec![target_padding.clone(), target_padding], 0),
            batch_ids: vec![1, 2],
        };

        let (loss, _targets) = calculate_loss(
            class_predictors.clone(),
            box_predictors,
            outputs,
            &ssd_batch,
        );

        loss
            .into_data()
            .assert_within_range(17.5..18.5);
    }

    #[test]
    fn test_loss_single() {
        type B = Autodiff<burn::backend::LibTorch>;
        let device = &burn::backend::libtorch::LibTorchDevice::Cuda(0);

        let batches = 2;

        B::seed(42);

        let outputs = [
            Tensor::<B, 4>::ones(Shape::new([batches, 512, 38, 38]), device), // VGG-16 through Conv5_3 layer
            Tensor::<B, 4>::ones(Shape::new([batches, 1024, 19, 19]), device), // Conv6 (FC7)
            Tensor::<B, 4>::ones(Shape::new([batches, 512, 10, 10]), device), // Conv8_2
            Tensor::<B, 4>::ones(Shape::new([batches, 256, 5, 5]), device),   // Conv9_2
            Tensor::<B, 4>::ones(Shape::new([batches, 256, 3, 3]), device),   // Conv10_2
            Tensor::<B, 4>::ones(Shape::new([batches, 256, 1, 1]), device),   // Conv11_2
        ];

        let target_image = Tensor::<B, 4>::ones([batches, 3, 300, 300], device);

        // labels are indexes into the name, (11 = cat, 12 = dog, 23 = horse)
        let target_labels = Tensor::<B, 1, Int>::from_data([11], device);
        let target_padding = Tensor::<B, 1, Int>::from_data([0], device);
        let gt_boxes = Tensor::<B, 2>::from_data(
            [
                [0.35725, 0.51429164, 0.61651564, 0.7677916], // 0
            ],
            device,
        );

        let class_predictors: Tensor<B, 3> = Tensor::random(
            [batches, 8732, 21],
            Distribution::Uniform(-1.0, 1.0),
            device,
        );
        let box_predictors: Tensor<B, 3> =
            Tensor::random([batches, 8732, 4], Distribution::Uniform(-1.0, 1.0), device);

        let ssd_batch = SSDBatch {
            images: target_image,
            gt_boxes: Tensor::stack(vec![gt_boxes.clone()], 0),
            target_labels: Tensor::stack(vec![target_labels.clone()], 0),
            target_padding: Tensor::stack(vec![target_padding.clone(), target_padding], 0),
            batch_ids: vec![1, 2],
        };

        let (loss, _targets) =
            calculate_loss(class_predictors, box_predictors, outputs, &ssd_batch);

        loss.into_data().assert_within_range(17.5..18.5);
    }

    #[test]
    fn test_box_regression() {
        type B = Autodiff<burn::backend::LibTorch>;
        let device = &burn::backend::libtorch::LibTorchDevice::Cuda(0);
        type FT = FloatElem<B>;

        let gt_boxes = Tensor::<B, 2>::from_data(
            [
                [0.35725, 0.51429164, 0.61651564, 0.7677916],
                [0.35725, 0.51429164, 0.61651564, 0.7677916],
            ],
            device,
        );

        let dflt_bxs = Tensor::<B, 2>::from_data(
            [
                [0.4080761, 0.42141542, 0.5919239, 0.7891109],
                [0.3687838, 0.5133393, 0.7364793, 0.69718707], // x1,y1, x2, y2
            ],
            device,
        );

        let output = box_regression(gt_boxes, dflt_bxs, (10.0, 10.0, 5.0, 5.0));

        Tensor::<B, 2>::from_data(
            [[-0.71, 0.97, 1.72, -1.86], [-1.79, 1.95, -1.75, 1.61]],
            device,
        )
        .into_data()
        .assert_approx_eq::<FT>(&output.to_data(), Tolerance::default());
    }
}
