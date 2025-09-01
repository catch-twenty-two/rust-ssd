use core::panic;

use crate::{broadcast, check_nan, layers::SSDConvLayers};

use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, s},
};

/// Returns the number of default anchor boxes generated for a given convolutional
/// layer in an SSD (Single Shot Multibox Detector) model.
///
/// # Arguments
///
/// * `conv` - A reference to an `SSDConvLayers` instance, representing a specific
///   convolutional feature map layer in the SSD architecture.
///
/// # Behavior
///
/// Each layer is associated with a set of aspect ratios for anchor boxes.  
/// The function computes the total number of anchor boxes for the given layer by:
///
/// 1. Calling `ar(conv)` to retrieve the list of aspect ratios defined for the layer.  
/// 2. Taking the length of this aspect ratio list (`ar(conv).len()`).  
/// 3. Adding **1** to include the default anchor box with aspect ratio = 1 (which
///    SSD always uses in addition to the specified aspect ratios).
///
/// # Returns
///
/// The total number of anchor boxes (`usize`) assigned to the given layer,
/// equal to the number of configured aspect ratios plus one default box.
///
pub fn get_box_count_for_layer(conv: &SSDConvLayers) -> usize {
    // Add default box for aspect ratio 1
    ar(conv).len() + 1
}

/// Returns the widths and heights of all default boxes for a specific SSD300 layer.
///
/// In the SSD300 model, each convolutional feature map layer predicts bounding boxes
/// for every spatial location on that map. For example, a layer with a size of `38x38`
/// means there are 38 cells across the width and height of the feature map, and each
/// cell will have multiple default boxes at different aspect ratios and scales.
/// This function computes the `(width, height)` for each default box of the given layer.
///
/// # Arguments
///
/// * `conv` - Reference to the SSD convolutional layer metadata (`SSDConvLayers`).
///
/// # Returns
///
/// A vector of `(width, height)` tuples representing default box sizes in
/// relative scale (fractions of the input image).
pub fn get_default_boxes(conv: &SSDConvLayers) -> Vec<(f32, f32)> {
    let mut box_sizes = Vec::<(f32, f32)>::default();

    // Layer index or identifier used to determine scale
    let k = conv.get_id();

    // Add default box for aspect ratio 1
    box_sizes.push(ar1(k));

    // Add the rest of the boxes, stored as (width, height) for each aspect ratio
    for ar in ar(conv).iter() {
        box_sizes.push((wk(k, *ar), hk(k, *ar)));
    }

    box_sizes
}

/// Maps target boxes and labels to the SSD default boxes.
///
/// In SSD300, there are N default boxes across all layers. Each default box
/// can be assigned a ground truth label or marked as background (0) if no
/// object overlaps it.
///
/// # Notes
///
/// If a default box is overlapping a ground truth box, it is marked with that ground truth
/// box’s label class id, if not it is marked with a 0 as a box containing a background
/// image. This allows the tensor to act as an Associative Array
/// (https://en.wikipedia.org/wiki/Associative_array), where in the index of the id equals
/// the default box number, the value at that index equals label number
///
///                                       0, 1, 2, 3, 4 <- Index (Default Box #)
/// Matching boxes are in the form of ([[-1,-1, 1, 0,-1]])
///
/// Above default box 2 overlaps with ground truth box 1, while default box 3 overlaps with
/// ground truth box 0.
///
/// # Arguments
///
/// * `matching_boxes` - Tensor of shape `[num_default_boxes]` containing the indices
///   of ground truth boxes that each default box overlaps (-1 if no match).
/// * `target_labels` - Tensor of shape `[num_ground_truth_boxes]` containing
///   the class labels of the ground truth boxes.
/// * `dflt_boxes` - Tensor of shape `[num_default_boxes, 4]` containing all default box
///   coordinates.
///
/// # Returns
///
/// * `Tensor<B, 1, Int>` - Tensor of shape `[num_default_boxes]` containing the
///   assigned class labels for each default box (0 for background).
pub fn target_lbls_to_default_boxes<B: Backend>(
    matching_boxes: &Tensor<B, 1, Int>,
    target_labels: Tensor<B, 1, Int>,
    dflt_boxes: Tensor<B, 2>,
) -> Tensor<B, 1, Int> {
    let device = &matching_boxes.device();

    // Get the indexes of all the default boxes that match with a corresponding ground truth box
    let mb_mask = matching_boxes.clone().greater_elem(-1);
    let labels_index = Tensor::cat(mb_mask.clone().nonzero(), 0);

    // Save this  map to an index and the save the label value at this index
    let index_map: Tensor<B, 1, Int> = Tensor::from_data(
        (0..mb_mask.shape().num_elements())
            .map(|x| x as i32)
            .collect::<Vec<i32>>()
            .as_slice(),
        device,
    );

    let boxes_index = matching_boxes.clone().select(0, labels_index.clone());

    // Create a new associative array tensor where the labels correspond to an index in the
    // default box tensor. (Most of these will be 0, or background)
    Tensor::zeros([dflt_boxes.shape().num_elements() / 4], device).scatter(
        0,
        index_map.select(0, labels_index),
        target_labels.select(0, boxes_index),
    )
}

/// Computes the Intersection over Union (IoU) between two sets of bounding boxes in `xyxy` format.
///
/// This function takes two tensors:
/// - `gt_boxes` containing ground truth boxes
/// - `dflt_boxes` containing default/prior boxes
///
/// Each box is represented by its `(x1, y1, x2, y2)` coordinates, where `(x1, y1)`
/// is the top-left corner and `(x2, y2)` is the bottom-right corner.  
/// IoU is calculated as:
///
/// `IoU = intersection_area / union_area`
///
/// The function returns a matrix of shape `[num_gt_boxes, num_default_boxes]` where
/// each entry `(i, j)` is the IoU between ground truth box `i` and default box `j`.
///
/// # Type Parameters
/// * `B` - Backend implementing the tensor operations.
///
/// # Arguments
/// * `gt_boxes` - Tensor of shape `[N, 4]` containing ground truth boxes in `xyxy` format.
/// * `dflt_boxes` - Tensor of shape `[M, 4]` containing default boxes in `xyxy` format.
///
/// # Returns
/// * `Tensor<B, 2>` - IoU matrix of shape `[N, M]`.
pub fn get_iou<B: Backend>(gt_boxes: Tensor<B, 2>, dflt_boxes: Tensor<B, 2>) -> Tensor<B, 2> {
    // Split ground truth and default boxes into component coordinates (x1, y1, x2, y2)
    let (gtx1, gty1, gtx2, gty2) = boxes_to_components(gt_boxes.clone());
    let (dfx1, dfy1, dfx2, dfy2) = boxes_to_components(dflt_boxes.clone());

    // --- Intersection top-left corner ---
    // max left: select the larger x1 between each ground truth and default box
    let dfx1_b = dfx1.clone().reshape([1, -1]);
    let (a, b) = broadcast!(gtx1: Tensor<B, 2>, dfx1_b: Tensor<2>);
    let x1_max = a.max_pair(b);

    // max top: select the larger y1 between each ground truth and default box
    let dfy1_b = dfy1.clone().reshape([1, -1]);
    let (a, b) = broadcast!(gty1: Tensor<B, 2>, dfy1_b: Tensor<2>);
    let y1_max = a.max_pair(b);

    // --- Intersection bottom-right corner ---
    // min right: select the smaller x2 between each ground truth and default box
    let dfx2_b = dfx2.clone().reshape([1, -1]);
    let (a, b) = broadcast!(gtx2: Tensor<B, 2>, dfx2_b: Tensor<2>);
    let x2_min = a.min_pair(b);

    // min bottom: select the smaller y2 between each ground truth and default box
    let dfy2_b = dfy2.clone().reshape([1, -1]);
    let (a, b) = broadcast!(gty2: Tensor<B, 2>, dfy2_b: Tensor<2>);
    let y2_min = a.min_pair(b);

    // --- Areas ---
    // Area of each ground truth box
    let area_dflt = (gtx2 - gtx1) * (gty2 - gty1);

    // Area of each default box
    let area_gt = (dfx2 - dfx1) * (dfy2 - dfy1);

    // --- Intersection area ---
    // Width = x2_min - x1_max, height = y2_min - y1_max, clamp at 0 to avoid negatives
    let intersection_area = (x2_min - x1_max).clamp_min(0) * (y2_min - y1_max).clamp_min(0);

    // --- Union area ---
    let area_gt_b = area_gt.reshape([1, -1]);
    let (a, b) = broadcast!(area_dflt: Tensor<B, 2>, area_gt_b: Tensor<2>);
    let union = (a + b) - intersection_area.clone();

    // IoU = intersection / union
    intersection_area / union
}

/// Determines which default boxes overlap with ground truth boxes based on IoU.
///
/// During SSD training, each ground truth box is matched to default boxes across different
/// locations, aspect ratios, and scales. This function:
///
/// 1. Matches each ground truth box to the default box with the highest Jaccard (IoU) overlap.
/// 2. Additionally matches any default box whose IoU with a ground truth box exceeds `threshold`.
///
/// This allows the network to learn from multiple overlapping default boxes,
/// not just the single best match, simplifying training.
///
/// # Notes
///
/// Relevant excerpt from
///
/// “SSD: Single Shot MultiBox Detector”
/// Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
///          Scott Reed, Cheng-Yang Fu, Alexander C. Berg
/// Link (official): https://arxiv.org/abs/1512.02325
///
/// Matching strategy - Liu et al. - Pg 6
///
/// During training we need to determine which default boxes correspond to a ground truth
/// detection and train the network accordingly. For each ground truth box we are selecting from
/// default boxes that vary over location, aspect ratio, and scale. We begin by matching each ground
/// truth box to the default box with the best Jaccard overlap (as in MultiBox [7]). Unlike
/// MultiBox, we then match default boxes to any ground truth with Jaccard overlap higher than a
/// threshold (0.5). This simplifies the learning problem, allowing the network to predict high
/// scores for multiple overlapping default boxes rather than requiring it to pick only the one with
/// maximum overlap.
///
/// # Arguments
///
/// * `gt_boxes` - Tensor of shape `[num_gt_boxes, 4]` containing ground truth boxes in xyxy format.
/// * `dflt_bxs` - Tensor of shape `[num_default_boxes, 4]` containing all SSD default boxes in
///   xyxy format.
/// * `threshold` - IoU threshold to consider a default box as overlapping with a ground truth box.
///
/// # Returns
///
/// * `Tensor<B, 1, Int>` - Tensor of shape `[num_default_boxes]` containing the index of the
///   matching ground truth box for each default box, or `-1` if no overlap exceeds the threshold.
pub fn get_overlapping_dflt_boxes<B: Backend>(
    gt_boxes: Tensor<B, 2>,
    dflt_bxs: Tensor<B, 2>,
    threshold: f32,
) -> Tensor<B, 1, Int> {
    let device = &gt_boxes.device();

    let [gt_bx_cnt, _] = gt_boxes.shape().dims();

    // Get IOU matrix
    //
    //                    Dim 1
    //             Default Anchor Boxes
    //             ┌──────────────►
    //             │┌──────────────────┐
    //             ││                  │
    //             ││                  │
    //    Dim 0    ││       IOU        │
    // Image Target││                  │
    //    Boxes    ││                  │
    //             ││                  │
    //             ▼└──────────────────┘

    let iou = get_iou(gt_boxes, dflt_bxs);

    // Get ALL overlapping box values and only keep ones that are higher than the threshold
    // all others get replaced with a -1
    //
    // ┌───────────────────────────► IOU Dim 0
    // │     ┌─────┬───────┬─────┬──── Get maximum value in each of these columns
    // │     ▼     ▼       ▼     ▼
    // │ [0.10, 0.10, ..., 0.30, 0.30], <-- dim0 , index 1
    // │ [0.20, 0.25, ..., 0.40, 0.45], <-- ... , index  2
    // │ [0.60, 0.50, ..., 0.80, 0.70], <-- ... , index  3
    // │ [...,  ...,  ..., ...,   ...], <-- ...
    // │ [0.35, 0.15, ..., 0.55, 0.35], <-- ... , index  8730
    // │ [0.50, 0.60, ..., 0.70, 0.80], <-- ... , index  8731
    // │ [0.25, 0.40, ..., 0.45, 0.60], <-- ... , index  8732
    // ▼
    // IOU Dim 1
    //

    let (val, index) = iou.clone().max_dim_with_indices(0);
    let threshold_mask = val.lower_elem(threshold);
    let min_iou_thresh = index.mask_fill(threshold_mask, -1);

    // Get the maximum values in the rows along dimension 0 save these
    // as the target boxes that have the maximum overlap with the
    // anchor boxes (see below for implementation)
    //
    // ┌───────────────────────────► IOU Dim 0
    // │ ┌►[0.10, 0.10, ..., 0.30, 0.30], <-- ..., index 1
    // │ ├►[0.20, 0.25, ..., 0.40, 0.45], <-- ..., index 2
    // │ ├►[0.60, 0.50, ..., 0.80, 0.70], <-- ..., index 3
    // │ ├►[...,  ...,  ..., ...,  ... ], <-- ...
    // │ ├►[0.35, 0.15, ..., 0.55, 0.35], <-- ..., index 8730
    // │ ├►[0.50, 0.60, ..., 0.70, 0.80], <-- ..., index 8731
    // │ ├►[0.25, 0.40, ..., 0.45, 0.60], <-- ..., index 8732
    // │ │
    // │ Maximum value in each of these rows
    // ▼
    // IOU Dim 1

    let (_, max_iou_boxes) = iou.max_dim_with_indices(1);
    let max_iou_boxes = max_iou_boxes.reshape([1, -1]);

    // Find which Default Boxes (Anchor Boxes) have maximum overlap with the
    // Ground Truth Boxes (Labeled Bounding Boxes)
    //
    // Example output is in this form:
    //                        ┌──┬────── Labeled Bounding Boxes #1 and #0
    //                        ▼  ▼
    // (Tensor)-> ([[-1,-1, 1, 0,-1]])
    //                0, 1, 2, 3, 4 ◄──────── Index is Anchor Box #0,#1,#2,#3 and #4
    //                ▲  ▲  ▲  ▲
    //                │  │  │  │
    //                │  │  │  │
    //                │  │  │  |
    //                │  │  |  Anchor Box #3 has maximum overlap with Labeled Bounding Box #0
    //                │  |  Anchor Box #2 has maximum overlap with Labeled Bounding Box #1
    //                |  Anchor Box #1 does not overlap with anything (Or threshold <5)
    //                Anchor Box #0 does not overlap with anything (Or threshold < .5)
    //
    // Summary: Labeled Bounding Boxes #1 and #0 have maximum overlap with Anchor Boxes
    // #2 and #3 respectfully

    let mut best_matches = min_iou_thresh.to_data().to_vec::<i64>().unwrap();

    for (i, max_iou_box) in max_iou_boxes.iter_dim(1).enumerate() {
        if i == gt_bx_cnt {
            break;
        }

        let max_iou_box = max_iou_box.to_data().as_mut_slice::<i64>().unwrap()[0];

        if max_iou_box < 0 {
            continue;
        }

        best_matches[max_iou_box as usize] = i as i64;
    }

    Tensor::<B, 1, Int>::from_data(best_matches.as_slice(), device)
}

/// Generates all default boxes for multiple SSD feature maps.
///
/// Each feature map cell is associated with a set of default boxes of different aspect ratios
/// and scales. This function tiles default boxes over each feature map in a convolutional
/// manner, centering them on the "grid points" of the feature map, so that each box is
/// aligned relative to the corresponding cell. For each default box:
///
/// - The network predicts `c` class scores (per-class probabilities)
/// - The network predicts 4 offsets relative to the box's original shape (cx, cy, w, h)
///
/// Relevant excerpt from
///
/// “SSD: Single Shot MultiBox Detector”
/// Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
///          Scott Reed, Cheng-Yang Fu, Alexander C. Berg
/// Link (official): https://arxiv.org/abs/1512.02325
///
/// Default boxes and aspect ratios - Pg. 4 Liu et al.
///
/// We associate a set of default bounding boxes with each feature map cell, for multiple feature
/// maps at the top of the network. The default boxes tile the feature map in a convolutional
/// manner, so that the position of each box relative to its corresponding cell is fixed. At each
/// feature map cell, we predict the offsets relative to the default box shapes in the cell, as well
/// as the per-class scores that indicate the presence of a class instance in each of those boxes.
/// Specifically, for each box out of k at a given location, we compute c class scores and the 4
/// offsets relative to the original default box shape. This results in a total of (c + 4)k filters
/// that are applied around each location in the feature map, yielding (c + 4)kmn outputs for a m ×
/// n feature map. For an illustration of default boxes, please refer to Fig. 1. Our default boxes
/// are similar to the anchor boxes used in Faster R-CNN [2], however we apply them to several
/// feature maps of different resolutions. Allowing different default box shapes in several feature
/// maps let us efficiently discretize the space of possible output box shapes.
///
/// # Arguments
///
/// * `feature_maps` - Array of 6 feature map tensors from different layers of SSD, each
///   of shape `[batch, channels, width, height]`.
///
/// # Returns
///
/// * `Tensor<B, 3>` - Tensor containing all default boxes in xyxy format. The boxes are
///   grouped per feature map layer and stacked into a single tensor.
pub fn generate_all_default_boxes<B: Backend>(feature_maps: [Tensor<B, 4>; 6]) -> Tensor<B, 3> {
    let mut default_box_vec = vec![];

    let conv_list = SSDConvLayers::as_list();
    let device = feature_maps[0].device();

    // match the feature map outputs with the correct box group

    for (i, conv) in conv_list.iter().enumerate() {
        // Get the height and width of the feature and spread the boxes evenly across the
        // feature map centering each box at the middle of each map "grid point" which is
        // a grid superimposed on the original image. Since each map's spatial information
        // contains image information at this point it contains information for a box on
        // the image at the center of this point

        let [_batch, _depth, width, _] = feature_maps[i].shape().dims();

        // 'cxcywh': boxes are represented via center, width and height,cx,cy being center of
        // box, w, h being width and height

        // We set the center of each default box to (i+0.5)/|fk|, (j+0.5)/|fk| where |fk|
        // is the size of the k-th square feature map, i, j ∈ [0, |fk|).

        let (cx_vec, cy_vec) = get_default_box_centers(width);

        // create a list of boxes for as many grid points as required
        // e.g. for conv 4_3 38x38 this is 5776 boxes (38*38*4)
        //      for conv Conv7 this is 2166
        //      ect...
        // TODO: convert to tensor for faster speed
        let mut grid_pos: Vec<f32> = vec![];

        for cy in &cy_vec {
            for cx in &cx_vec {
                // get all default boxes for a single grid point for this layer
                // stored as (width, height)
                let default_boxes = get_default_boxes(conv);
                for (w, h) in default_boxes {
                    // convert from cx,cy,w,h to x1,y1,x2,y2
                    let xyxy = cxcywh_to_x1y1x2y2_f32(cx, cy, &w, &h);
                    grid_pos.extend(xyxy);
                }
            }
        }

        // Create a tensor from the current list of boxes and group the boxes coordinates
        // to groups of 4 from [1,...num of boxes] to [1, 4]
        //
        // e.g. [1,2,3,4,5,6,7,8,9,10,11,12] -> [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

        let boxes = Tensor::<B, 1>::from_floats(grid_pos.as_slice(), &device);
        let boxes = boxes.reshape([grid_pos.len() / 4, 4]);

        default_box_vec.push(boxes);
    }

    // Cat all the boxes together and Stack as many default boxes
    // as images that are in this batch. One for each image

    let [batch_size, _, _, _] = feature_maps[0].shape().dims();

    // pg 4, fig. 2, Boxes: 8732 per image

    let default_box_vec = [Tensor::cat(default_box_vec, 0)];

    let default_box_vec = default_box_vec
        .iter()
        .cycle() // infinite iterator over the original
        .take(default_box_vec.len() * batch_size)
        .cloned()
        .collect::<Vec<Tensor<B, 2>>>();

    Tensor::stack(default_box_vec, 0)
}

/// Computes the box regression targets for training
///
/// This function converts ground truth boxes (`g`) and default boxes (`d`) from `x1y1x2y2`
/// format to `cx,cy,w,h` format and computes the offsets (tx, ty, tw, th) for each box.
/// These offsets are used to train the network to predict corrections to the default boxes.
///
/// The regression formula:
/// - tx = (Gx - Px) / Pw * w1
/// - ty = (Gy - Py) / Ph * w2
/// - tw = log(Gw / Pw) * w3
/// - th = log(Gh / Ph) * w4
///
/// Where:
/// - (Gx, Gy, Gw, Gh) are ground truth box center coordinates, width, and height
/// - (Px, Py, Pw, Ph) are default box center coordinates, width, and height
/// - w1, w2, w3, w4 are scaling weights for each regression component
///
/// Logarithmic scaling for width and height allows stable learning across different box sizes.
///
/// Relevant excerpt from
///
/// “Rich feature hierarchies for accurate object detection and semantic segmentation”
/// Authors: Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik
/// Link (official): https://arxiv.org/abs/1311.2524
///
/// Appendix Section C. Bounding-box regression
///
/// Our goal is to learn a transformation that maps a proposed box P to a ground-truth box G. We
/// parameterize the transformation in terms of four functions dx(P), dy(P), dw(P), and dh(P). The
/// first two specify a scale-invariant translation of the center of P’s bounding box, while the
/// second two specify log-space translations of the width and height of P’s bounding box. After
/// learning these functions, we can transform an input proposal P into a predicted ground-truth box
/// by applying the transformation.
///
/// "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
/// Authors: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
/// Paper: https://arxiv.org/abs/1506.01497
///
/// 3.1.2 Loss Function
///
/// This can be thought of as bounding-box regression from an anchor box to a nearby ground-truth
/// box.
///
/// There is no mention in any academic paper as far as I can tell where they have added w1,w2,w3,
/// w4 or how to tune them effectively. They seem to have shown up in a facebook research
/// implementation of the bounding box regression weight implementation without
/// any explanation. (10.0, 10.0, 5.0, 5.0)
///
/// (https://github.com/facebookresearch/detectron2/blob/18f69583391e5040043ca4f4bebd2c60f0ebfde0/detectron2/config/defaults.py#L302)
///
/// In essence they act as hyperparameters controlling the relative importance of center
/// vs. size in the box regression loss. Tuning them effects how strongly the network
/// tries to correct offsets for x/y positions versus width/height, which can influence
/// convergence speed and final detection accuracy.
///
/// # Arguments
///
/// * `g` - Ground truth boxes tensor of shape [num_boxes, 4] in xyxy format.
/// * `d` - Default boxes tensor of shape [num_boxes, 4] in xyxy format.
/// * `(w1, w2, w3, w4)` - Scale factors for the four regression components.
///
/// # Returns
///
/// * `Tensor<B, 2>` - Tensor of shape [num_boxes, 4] containing the box regression targets
///   `(tx, ty, tw, th)` for each box.
pub fn box_regression<B: Backend>(
    g: Tensor<B, 2>,
    d: Tensor<B, 2>,
    (w1, w2, w3, w4): (f32, f32, f32, f32),
) -> Tensor<B, 2> {
    let (gx, gy, gw, gh) = boxes_to_components(x1y1x2y2_to_cxcywh(g));
    let (px, py, pw, ph) = boxes_to_components(x1y1x2y2_to_cxcywh(d));

    //      Gx - Px
    // tx = ───────
    //        Pw

    let tx = (gx - px) / pw.clone() * w1;

    //      Gy - Py
    // ty = ───────
    //        Ph

    let ty = (gy - py) / ph.clone() * w2;

    // Logarithmic scale factors measure how much a box’s size changes relative to a
    // default box. Using the log of the size ratio helps the model handle both small
    // and large objects better. Which makes training more stable and improves box
    // predictions.

    //          ⎛Gw⎞
    // tw = log ⎜──⎟
    //          ⎝Pw⎠

    let tw = (gw.clone() / pw.clone()).log() * w3;
    check_nan!(tw, gw, pw);

    //          ⎛Gh⎞
    // th = log ⎜──⎟
    //          ⎝Ph⎠

    let th = (gh / ph).log() * w4;

    Tensor::cat(vec![tx, ty, tw, th], 1)
}

/// Generates predicted bounding boxes from regression outputs and default boxes.
///
/// This function converts predicted offsets (`p`) and default boxes (`d`) into actual
/// bounding box coordinates in `x1y1x2y2` format. It essentially performs the inverse
/// of `box_regression`, applying the predicted adjustments to the default boxes to
/// reconstruct predicted box locations and sizes.
///
/// The formulas used:
/// - pcx = dx * Pw + Px
/// - pcy = dy * Ph + Py
/// - pw = exp(dw) * Pw
/// - ph = exp(dh) * Ph
///
/// Where:
/// - (dx, dy, dw, dh) are the normalized predictions from the network
/// - (Px, Py, Pw, Ph) are default box center coordinates, width, and height
/// - w1, w2, w3, w4 are scaling factors used during training
///
/// # Arguments
///
/// * `p` - Predicted regression offsets tensor of shape [num_boxes, 4].
/// * `d` - Default boxes tensor of shape [num_boxes, 4] in xyxy format.
/// * `(w1, w2, w3, w4)` - Scale factors that normalize predictions.
///
/// # Returns
///
/// * `Tensor<B, 2>` - Tensor of shape [num_boxes, 4] containing predicted boxes in
///   `x1y1x2y2` format.
pub fn box_generation<B: Backend>(
    p: Tensor<B, 2>,
    d: Tensor<B, 2>,
    (w1, w2, w3, w4): (f32, f32, f32, f32),
) -> Tensor<B, 2> {
    // Get cx,cy,w,h from x1,y1,x2,y2 for ground truth boxes

    let (center_x, center_y, w, h) = boxes_to_components(x1y1x2y2_to_cxcywh(d));

    // Get center x,center y, width and height from for prediction boxes

    let (mut dx, mut dy, mut dw, mut dh) = boxes_to_components(p);

    dx = dx / w1;
    dy = dy / w2;
    dw = dw / w3;
    dh = dh / w4;

    // Get the newly generated bounding box predictions

    let pcx = dx * w.clone() + center_x;
    let pcy = dy * h.clone() + center_y;
    let pw = dw.exp() * w;
    let ph = dh.exp() * h;

    let cxcywh = Tensor::cat(vec![pcx, pcy, pw, ph], 1);

    cxcywh_to_x1y1x2y2(cxcywh)
}

/// Each layer has a specified scale and each feature map for the layer has 6 aspect ratios for
/// each box, the following code has been worked directly into the paper:
///
/// “SSD: Single Shot MultiBox Detector”
/// Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu,
/// Alexander C. Berg
/// Link (official): https://arxiv.org/abs/1512.02325
///
/// Liu et al. Pg 6
///
/// Feature maps from different levels within a network are known to have different (empirical)
/// receptive field sizes. Fortunately, within the SSD framework, the default boxes do not
/// necessary need to correspond to the actual receptive fields of each layer. We design the tiling
/// of default boxes so that specific feature maps learn to be responsive to particular scales of
/// the objects.
///
/// Suppose we want to use m feature maps for prediction. The scale of the default boxes for each
/// feature map is computed as:
fn sk(k: usize) -> f32 {
    let smin = 0.1; // Modified to allow for smaller boxes orig (.2)
    let smax = 0.9;

    smin + ((smax - smin) / (SSDConvLayers::count() as f32 - smin)) * (k - 1) as f32
}

/// where smin is 0.1and smax is 0.9, meaning the lowest layer has a scale of 0.1 and the highest
/// layer has a scale of 0.9, and all layers in between are regularly spaced. We impose different
/// aspect ratios for the default boxes, and denote them as ar
///
///                        1   1
/// ar ⋅ ϵ ⋅ { 1 , 2 , 3 , ─ , ─ }
///                        2   3
///
pub fn ar(layer: &SSDConvLayers) -> Vec<f32> {
    match layer {
        // SSD: Single Shot MultiBox Detector - Classifier : Conv: 3x3x(4x(Classes+4)) - Fig. 2
        SSDConvLayers::Conv4_3 => vec![1., 2., 1.0 / 2.0],
        // SSD: Single Shot MultiBox Detector - Classifier : Conv: 3x3x(6x(Classes+4)) - Fig. 2
        SSDConvLayers::Conv7 => vec![1., 2., 3., 1.0 / 2.0, 1.0 / 3.0],
        SSDConvLayers::Conv8_2 => vec![1., 2., 3., 1.0 / 2.0, 1.0 / 3.0],
        SSDConvLayers::Conv9_2 => vec![1., 2., 3., 1.0 / 2.0, 1.0 / 3.0],
        // SSD: Single Shot MultiBox Detector - Classifier : Conv: 3x3x(4x(Classes+4)) - Fig. 2
        SSDConvLayers::Conv10_2 => vec![1., 2., 1.0 / 2.0],
        SSDConvLayers::Conv11_2 => vec![1., 2., 1.0 / 2.0],
    }
}

/// We can compute the width:
fn wk(k: usize, ar: f32) -> f32 {
    sk(k) * f32::sqrt(ar)
}

/// and height:
fn hk(k: usize, ar: f32) -> f32 {
    sk(k) / f32::sqrt(ar)
}

/// for each default box. For the aspect ratio of 1, we also add a default box whose scale is:
fn ar1(k: usize) -> (f32, f32) {
    let s1 = f32::sqrt(sk(k) * sk(k + 1));
    (s1, s1)
}
/// resulting in 6 default boxes per feature map location. We set the center of each default box to:
fn get_default_box_centers(k: usize) -> (Vec<f32>, Vec<f32>) {
    // width and height should always be the same, 38x38, 19x19, 10x10, ect...

    (
        (0..k).map(|i| (i as f32 + 0.5) / k as f32).collect(),
        (0..k).map(|j| (j as f32 + 0.5) / k as f32).collect(),
    )
}

// where |k| is the size of the k-th square feature map, i, j ∈ [0, |k|). In practice, one can
// also design a distribution of default boxes to best fit a specific dataset. How to design the
// optimal tiling is an open question as well. By combining predictions for all default boxes with
// different scales and aspect ratios from all locations of many feature maps, we have a diverse
// set of predictions, covering various input object sizes and shapes.

/// Splits a tensor of bounding boxes in any 4 component format (cxcywh, xyxy, ect..) into
/// individual components.
///
/// # Arguments
///
/// * `boxes` - Tensor of shape [num_boxes, 4] containing boxes in any 4 component format.
///
/// # Returns
///
/// * Tuple of four tensors `(c1, c2, c3, c4)` each of shape [num_boxes, 2].
pub fn boxes_to_components<B: Backend>(
    boxes: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let c1 = s![.., 0];
    let c2 = s![.., 2];
    let c3 = s![.., 1];
    let c4 = s![.., 3];
    (
        boxes.clone().slice(c1),
        boxes.clone().slice(c3),
        boxes.clone().slice(c2),
        boxes.slice(c4),
    )
}

/// Converts a tensor of bounding boxes from center format (cx, cy, w, h) to corner
/// format (x1, y1, x2, y2).
///
/// # Arguments
///
/// * `a` - Tensor of shape [num_boxes, 4] in cxcywh format.
///
/// # Returns
///
/// * `Tensor<B, 2>` - Tensor of shape [num_boxes, 4] in xyxy format.
pub fn cxcywh_to_x1y1x2y2<B: Backend>(a: Tensor<B, 2>) -> Tensor<B, 2> {
    let (cx, cy, w, h) = boxes_to_components(a);

    Tensor::cat(
        vec![
            cx.clone() - w.clone() * 0.5,
            cy.clone() - h.clone() * 0.5,
            cx + w * 0.5,
            cy + h * 0.5,
        ],
        1,
    )
}

pub fn xywh_to_x1y1x2y2<B: Backend>(a: Tensor<B, 2>) -> Tensor<B, 2> {
    let (x, y, w, h) = boxes_to_components(a);

    Tensor::cat(vec![x.clone(), y.clone(), x + w, y + h], 1)
}

/// Converts a tensor of bounding boxes from corner format (x1, y1, x2, y2) to center format
/// (cx, cy, w, h).
///
/// # Arguments
///
/// * `a` - Tensor of shape [num_boxes, 4] in xyxy format.
///
/// # Returns
///
/// * `Tensor<B, 2>` - Tensor of shape [num_boxes, 4] in cxcywh format.
pub fn x1y1x2y2_to_cxcywh<B: Backend>(a: Tensor<B, 2>) -> Tensor<B, 2> {
    let (x1, y1, x2, y2) = boxes_to_components(a);

    let w = x2.clone() - x1.clone();
    let h = y2.clone() - y1.clone();
    let cx = x1.clone() + w.clone() * 0.5;
    let cy = y1.clone() + h.clone() * 0.5;

    Tensor::cat(vec![cx, cy, w, h], 1)
}

/// Converts bounding box from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).
///
/// # Arguments
///
/// * `cx` - Center x-coordinate of the box.
/// * `cy` - Center y-coordinate of the box.
/// * `w` - Width of the box.
/// * `h` - Height of the box.
///
/// # Returns
///
/// * `Vec<f32>` - Vector of four floats representing [x1, y1, x2, y2].
pub fn cxcywh_to_x1y1x2y2_f32(cx: &f32, cy: &f32, w: &f32, h: &f32) -> Vec<f32> {
    vec![cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]
}


#[cfg(test)]
mod tests {
    use crate::debug::{assert_approx_eq, set_tensor_dbg_precision};

    use super::*;
    use burn::{
        backend::{NdArray, ndarray::NdArrayDevice},
        tensor::{Int, Shape, Tolerance, ops::FloatElem},
    };

    fn get_output(batches: usize) -> [Tensor<NdArray, 4>; 6] {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;
        [
            Tensor::<B, 4>::ones(Shape::new([batches, 512, 38, 38]), device), // VGG-16 through Conv5_3 layer
            Tensor::<B, 4>::ones(Shape::new([batches, 1024, 19, 19]), device), // Conv6 (FC7)
            Tensor::<B, 4>::ones(Shape::new([batches, 512, 10, 10]), device), // Conv8_2
            Tensor::<B, 4>::ones(Shape::new([batches, 256, 5, 5]), device),   // Conv9_2
            Tensor::<B, 4>::ones(Shape::new([batches, 256, 3, 3]), device),   // Conv10_2
            Tensor::<B, 4>::ones(Shape::new([batches, 256, 1, 1]), device),   // Conv11_2
        ]
    }

    #[test]
    fn generate_default_boxes_test() {
        let batches = 2;
        let output = get_output(batches);

        let default_boxes = generate_all_default_boxes(output);

        // pg 4, fig. 2, Detections:8732 per Class x2 batches (images)

        assert_eq!(default_boxes.shape().dims, [batches, 8732, 4]);
    }

    #[test]
    fn test_spacing() {
        let (ch, cw) = get_default_box_centers(10);

        assert_eq!(
            [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
            ch.as_slice()
        );
        assert_eq!(
            [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
            cw.as_slice()
        );
    }

    #[test]
    fn test_one_box() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;
        let gt_bx = Tensor::<B, 2>::from_data([[0.35725, 0.51429164, 0.40, 0.55]], &device);

        let batches = 1;

        let output = get_output(batches);

        let dflt_bxs = generate_all_default_boxes(output);

        let matches = get_overlapping_dflt_boxes(gt_bx, dflt_bxs.squeeze(0), 0.1);
        let matches = Tensor::cat(matches.greater_elem(-1).nonzero(), 0);

        Tensor::<B, 1, Int>::from_data(
            [
                2791, 2795, 2938, 2941, 2942, 2943, 2945, 2946, 2947, 2949, 2950, 2954, 3090, 3093,
                3094, 3095, 3097, 3098, 3099, 3101, 3102, 3106, 3245, 3247, 3249, 3251, 3253, 3403,
            ],
            device,
        )
        .into_data()
        .assert_eq(&matches.to_data(), true);
    }

    #[test]
    fn test_iou() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        let boxes1 = Tensor::<B, 2>::from_data(
            [
                [0.12, 0.15, 0.30, 0.40],
                [0.05, 0.05, 0.25, 0.20],
                [0.33, 0.20, 0.50, 0.45],
                [0.60, 0.10, 0.85, 0.35],
            ],
            &device,
        );

        let boxes2 = Tensor::<B, 2>::from_data(
            [
                [0.10, 0.10, 0.30, 0.30],
                [0.20, 0.25, 0.40, 0.45],
                [0.60, 0.50, 0.80, 0.70],
                [0.35, 0.15, 0.55, 0.35],
                [0.50, 0.60, 0.70, 0.80],
                [0.25, 0.40, 0.45, 0.60],
            ],
            &device,
        );

        let iou = get_iou(boxes1, boxes2);

        Tensor::<B, 2>::from_data(
            [
                [0.46551722, 0.21428573, 0.0, 0.0, 0.0, 0.0],
                [0.27272725, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.20437954, 0.0, 0.375, 0.0, 0.07843133],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            &device,
        )
        .into_data()
        .assert_approx_eq::<FloatElem<B>>(&iou.to_data(), Tolerance::default());
    }

    #[test]
    fn test_assign_target_labels_to_default_boxes() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        let target_labels =
            Tensor::<B, 1, Int>::from_data([1, 2, 0, 4, 5, 4, 7, 8, 9, 10, 12, 12, 13, 7], device);

        let gt_boxes = Tensor::<B, 2>::from_data(
            [
                [0.12, 0.15, 0.30, 0.40],
                [0.05, 0.05, 0.25, 0.20],
                [0.33, 0.20, 0.50, 0.45],
                [0.60, 0.10, 0.85, 0.35],
                [0.40, 0.60, 0.65, 0.85],
                [0.15, 0.30, 0.35, 0.55],
                [0.70, 0.70, 0.90, 0.95],
                [0.25, 0.25, 0.45, 0.50],
                [0.50, 0.40, 0.75, 0.65],
                [0.10, 0.60, 0.30, 0.80],
                [0.55, 0.20, 0.75, 0.40],
                [0.20, 0.10, 0.45, 0.35],
                [0.35, 0.50, 0.60, 0.75],
                [0.65, 0.30, 0.85, 0.55],
            ],
            device,
        );

        let dflt_boxes = Tensor::<B, 2>::from_data(
            [
                [0.10, 0.10, 0.30, 0.30],
                [0.20, 0.25, 0.40, 0.45],
                [0.60, 0.50, 0.80, 0.70],
                [0.35, 0.15, 0.55, 0.35],
                [0.50, 0.60, 0.70, 0.80],
                [0.25, 0.40, 0.45, 0.60],
                [0.05, 0.65, 0.25, 0.85],
                [0.70, 0.25, 0.90, 0.45],
                [0.10, 0.50, 0.30, 0.70],
                [0.55, 0.05, 0.75, 0.25],
                [0.45, 0.75, 0.65, 0.95],
                [0.20, 0.60, 0.40, 0.80],
                [0.65, 0.40, 0.85, 0.60],
                [0.30, 0.35, 0.50, 0.55],
                [0.15, 0.20, 0.35, 0.40],
                [0.40, 0.10, 0.60, 0.30],
            ],
            device,
        );

        set_tensor_dbg_precision(2);

        let matching = get_overlapping_dflt_boxes(gt_boxes.clone(), dflt_boxes.clone(), 0.5);

        let trget_lbs = target_lbls_to_default_boxes(&matching, target_labels, dflt_boxes);

        Tensor::<B, 1, Int>::from_data(
            [7, 8, 9, 0, 13, 0, 10, 0, 0, 12, 0, 0, 7, 0, 12, 0],
            device,
        )
        .into_data()
        .assert_eq(&trget_lbs.to_data(), true);
    }

    #[test]
    fn test_check_bboxes_overlap() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        let boxes1 = Tensor::<B, 2>::from_data(
            [
                [0.12, 0.15, 0.30, 0.40],
                [0.05, 0.05, 0.25, 0.20],
                [0.33, 0.20, 0.50, 0.45],
                [0.60, 0.10, 0.85, 0.35],
                [0.40, 0.60, 0.65, 0.85],
                [0.15, 0.30, 0.35, 0.55],
                [0.70, 0.70, 0.90, 0.95],
                [0.25, 0.25, 0.45, 0.50],
                [0.50, 0.40, 0.75, 0.65],
                [0.10, 0.60, 0.30, 0.80],
                [0.55, 0.20, 0.75, 0.40],
                [0.20, 0.10, 0.45, 0.35],
                [0.35, 0.50, 0.60, 0.75],
                [0.65, 0.30, 0.85, 0.55],
            ],
            &device,
        );

        let boxes2 = Tensor::<B, 2>::from_data(
            [
                [0.10, 0.10, 0.30, 0.30],
                [0.20, 0.25, 0.40, 0.45],
                [0.60, 0.50, 0.80, 0.70],
                [0.35, 0.15, 0.55, 0.35],
                [0.50, 0.60, 0.70, 0.80],
                [0.25, 0.40, 0.45, 0.60],
                [0.05, 0.65, 0.25, 0.85],
                [0.70, 0.25, 0.90, 0.45],
                [0.10, 0.50, 0.30, 0.70],
                [0.55, 0.05, 0.75, 0.25],
                [0.45, 0.75, 0.65, 0.95],
                [0.20, 0.60, 0.40, 0.80],
                [0.65, 0.40, 0.85, 0.60],
                [0.30, 0.35, 0.50, 0.55],
                [0.15, 0.20, 0.35, 0.40],
                [0.40, 0.10, 0.60, 0.30],
            ],
            &device,
        );

        let matching = get_overlapping_dflt_boxes(boxes1, boxes2, 0.5);

        Tensor::<B, 1, Int>::from_data(
            [6, 7, 8, 2, 12, -1, 9, -1, -1, 10, -1, -1, 13, -1, 11, -1],
            device,
        )
        .into_data()
        .assert_eq(&matching.to_data(), true);
    }

    #[test]
    fn test_box_regression() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;
        type FT = FloatElem<B>;
        let gt_boxes =
            Tensor::<B, 2>::from_data([[0.35725, 0.51429164, 0.61651564, 0.7677916]], device);

        let dflt_bxs =
            Tensor::<B, 2>::from_data([[0.4080761, 0.42141542, 0.5919239, 0.7891109]], device);

        let br = box_regression(gt_boxes.clone(), dflt_bxs.clone(), (10.0, 10.0, 5.0, 5.0));

        let gen_boxes = box_generation(gt_boxes, dflt_bxs, (10.0, 10.0, 5.0, 5.0));

        Tensor::<B, 2>::from_data([[-0.7134, 0.9730, 1.718, -1.859]], device)
            .into_data()
            .assert_approx_eq::<FT>(&br.to_data(), Tolerance::default());

        Tensor::<B, 2>::from_data([[0.4025, 0.4098, 0.6105, 0.8385]], device)
            .into_data()
            .assert_approx_eq::<FT>(&gen_boxes.to_data(), Tolerance::default());
    }

    #[test]
    fn feature_box_test() {
        for (a, b) in [
            (0.296, 0.296),
            (0.236, 0.236),
            (0.333, 0.167),
            (0.408, 0.136),
            (0.167, 0.333),
            (0.136, 0.408),
        ]
        .iter()
        .zip(get_default_boxes(&SSDConvLayers::Conv7))
        {
            assert_approx_eq(&a.0, &b.0, 1e-3);
            assert_approx_eq(&a.1, &b.1, 1e-3);
        }

        for (a, b) in [
            (0.153, 0.153),
            (0.100, 0.100),
            (0.141, 0.071),
            (0.071, 0.141),
        ]
        .iter()
        .zip(get_default_boxes(&SSDConvLayers::Conv4_3))
        {
            assert_approx_eq(&a.0, &b.0, 1e-3);
            assert_approx_eq(&a.1, &b.1, 1e-3);
        }

        for (a, b) in [
            (0.843, 0.843),
            (0.778, 0.778),
            (1.100, 0.550),
            (0.550, 1.100),
        ]
        .iter()
        .zip(get_default_boxes(&SSDConvLayers::Conv11_2))
        {
            assert_approx_eq(&a.0, &b.0, 1e-3);
            assert_approx_eq(&a.1, &b.1, 1e-3);
        }
    }
}
