use burn::{
    record::{CompactRecorder, Recorder},
    tensor::{Tensor, activation::softmax, backend::Backend},
};

use crate::{
    boxes::{box_generation, generate_all_default_boxes},
    broadcast,
    config::{HEIGHT, WIDTH},
    labels::SSDRemapCOCOID,
    models::ssd::SSD,
    nms,
    transforms::pipeline::Transform,
};

/// Runs inference on a single image using a trained SSD (Single Shot Detector) model,
/// producing bounding box predictions and saving the annotated result to disk.
///
/// # Parameters
/// - `image_path: &str`: Path to the input image file.
/// - `weights_file: &str`: Path to the model checkpoint containing trained SSD weights.
/// - `coco_remap: &SSDRemapCOCOID`: Mapping from model output class indices to COCO dataset labels.
/// - `device: &B::Device`: Computational device (CPU or GPU) where inference should run.
///
/// # Output
/// - Produces a file `ssd300_output.jpg` showing the input image with detected objects
///   annotated by bounding boxes and class labels.
/// 
pub fn infer<B: Backend>(
    image_path: &str,
    weights_file: &str,
    coco_remap: &SSDRemapCOCOID,
    device: &B::Device,
    iou_overlap_thresh: &f32,
    conf_score_thresh: &f32
) {
    let record = CompactRecorder::new()
        .load(weights_file.into(), device)
        .unwrap_or_else(|_| panic!("Trained model not found at {}", image_path));

    let image = image::open(image_path).unwrap().to_rgb8();

    let (image_t, _, _) = Transform::new(image.clone(), None, None, device)
        .resize_bilinear(WIDTH, HEIGHT)
        .normalize()
        .finish()
        .unwrap();

    let model: SSD<B> = SSD::new(device, record, coco_remap.count());

    // The object detection model generates multiple bounding boxes for objects in an image
    // each with N confidence scores (one per class) indicating the likelihood of the presence
    // of that object in that box. The class with the highest confidence score is chosen as
    // the class inside the box. 
    //
    // Most boxes will contain the 0 class or 'background' as the highest score and will be
    // ignored. Box predictors are box 'translation' or 'regression' predictors these contain
    // translation values on how to generate new boxes from it's corresponding default box
    // so the default box is better able to enclose the object that is being detected.

    let (class_predictors, box_predictors, conv_layers) =
        model.forward(image_t.clone().unsqueeze());

    let default_boxes: Tensor<B, 2> = generate_all_default_boxes(conv_layers).squeeze(0);
    let box_predictions = box_predictors.squeeze::<2>(0);
    let class_predictors_sm = softmax(class_predictors.clone(), 2);

    // Convert the predictions to a softmax score and pick the highest confidence for each box.
    let class_predictions = class_predictors_sm.clone().argmax(2).flatten::<1>(0, 2);

    // Filter out boxes with background detections, we don't care about these. Get the indexes
    // for the rest
    let class_pred_index = Tensor::cat(class_predictions.clone().bool().nonzero(), 0).unsqueeze();

    // Get default boxes containing a detected class (other than a background class) and use the
    // predicted translations to position these correctly

    let default_boxes = default_boxes.select(0, class_pred_index.clone());
    let box_predictions = box_predictions.select(0, class_pred_index.clone());

    let gen_box_predictions = box_generation(
        box_predictions.clone(),
        default_boxes.clone(),
        (10.0, 10.0, 5.0, 5.0),
    );

    // Get the corresponding default boxes confidence levels containing a detected class (other
    // than a background class) and use these when deciding which boxes to keep during the
    // non-maximum suppression algorithm

    let class_confidence = class_predictors_sm.clone().select(1, class_pred_index);

    let nms_boxes = nms::nms(
        gen_box_predictions.clone().unsqueeze(),
        class_confidence.clone(),
        *iou_overlap_thresh,
        *conf_score_thresh,
    );

    let mut labels_list = vec![];
    let mut bbox_list = vec![];

    for batch_gr in nms_boxes.iter() {
        for class_grp in batch_gr.iter() {
            for bbox in class_grp.iter() {
                bbox_list.push(bbox.to_tensor::<B>(device));
                labels_list.push(Tensor::from_data([bbox.cls_id], device));
            }
        }
    }

    let bboxes = Tensor::cat(bbox_list, 0).reshape([-1, 4]);
    let labels = Tensor::cat(labels_list, 0);

    let aspect_ratio: Tensor<B, 1> = Tensor::from_floats(
        [image.width(), image.height(), image.width(), image.height()],
        device,
    );

    let (a, b) = broadcast!(bboxes: Tensor<B, 2>, aspect_ratio: Tensor<1>);
    let bboxes = a * b;

    // draw and save the original image with the detected types

    Transform::new(image, Some(bboxes), Some(labels), device).clean_boxes().unwrap()
        .save_as("./ssd300_output.jpg".into(), coco_remap);
}
