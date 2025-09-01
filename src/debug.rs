use burn::{
    prelude::Backend,
    tensor::{PrintOptions, set_print_options},
};
use num::Float;

use crate::{labels::SSDRemapCOCOID, nms};

#[macro_export]
macro_rules! check_nan {
    ($a:expr, $b:expr, $c: expr) => {{
        use burn::tensor::cast::ToElement;
        if $a.clone().is_nan().int().sum().into_scalar().to_i32() > 0 {
            println!("A = {}", $a);
            println!("B = {}", $b);
            println!("C = {}", $c);
            panic!("found nan in line {}", line!());
        }
    }};
}

#[macro_export]
macro_rules! check_nan_1 {
    ($a:expr) => {{
        use burn::tensor::cast::ToElement;
        if $a.clone().is_nan().int().sum().into_scalar().to_i32() > 0 {
            println!("A = {}", $a);
            panic!("found nan in line {}", line!());
        };
    }};
}

#[macro_export]
macro_rules! check_neg {
    ($a:expr) => {
        if $a.clone().lower_elem(0).int().sum().into_scalar().to_i32() > 0 {
            println!("A = {}", $a);
            panic!("found negative in line {}", line!());
        };
    };
}

pub fn set_tensor_dbg_precision(prec: usize) {
    let po = PrintOptions {
        precision: Option::Some(prec),
        edge_items: 10,
        ..Default::default()
    };

    set_print_options(po);
}

// Helper function for comparing floats
pub fn assert_approx_eq<F>(a: &F, b: &F, epsilon: F)
where
    F: Float + std::fmt::Display + std::fmt::Debug,
{
    assert!(
        (*a - *b).abs() <= epsilon,
        "Values differ: {:?} vs {:?} (tolerance: {:?})",
        *a,
        *b,
        epsilon
    );
}

pub fn to_torch_python<B: Backend>(
    image_path: &str,
    nms_boxes: Vec<Vec<Vec<nms::BoundingBox>>>,
    lbl_remap: &SSDRemapCOCOID,
) {
    let mut clist = vec![];
    let mut tlist = vec![];

    for bv in nms_boxes.iter() {
        for bv in bv.iter() {
            for t in bv.iter() {
                tlist.push(t.to_tensor::<B>(&B::Device::default()).to_data());
                clist.push(t.cls_id);
            }
        }
    }

    // Output for torch debugging:

    println!("image = Image.open('{}').convert('RGB')", image_path);

    println!("bboxes = torch.Tensor([");
    tlist.iter().for_each(|s| print!("{},", s));
    println!("])");

    println!("labels = [");
    clist
        .iter()
        .for_each(|s| print!("'{}',", lbl_remap.model_id_to_coco_name(s)));
    println!("]");
}
