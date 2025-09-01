use burn::{
    prelude::Backend,
    tensor::{Device, Int, Tensor},
};

use crate::broadcast;

pub fn linspace<B: Backend>(
    start: f32,
    end: f32,
    steps: usize,
    device: &Device<B>,
) -> Tensor<B, 1> {
    if steps == 1 {
        return Tensor::<B, 1>::from_floats([start], device);
    }

    Tensor::<B, 1, Int>::arange(0..steps as i64, device)
        .float()
        .mul_scalar((end - start) / (steps as f32 - 1.0))
        .add_scalar(start)
}

pub fn meshgrid<B: Backend>(x: Tensor<B, 1>, y: Tensor<B, 1>) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let x = x.reshape([-1, 1]);
    broadcast!(x:Tensor<B, 2>, y: Tensor<1>)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{NdArray, ndarray::NdArrayDevice};

    #[test]
    pub fn test_linspace() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;
        type FT = burn::tensor::ops::FloatElem<B>;
        let ls = linspace::<B>(3.0, 10.0, 5, device);
        Tensor::<B, 1>::from_data([3.0, 4.75, 6.5, 8.25, 10.0], device)
            .into_data()
            .assert_approx_eq::<FT>(&ls.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    pub fn test_meshgrid() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;
        type FT = burn::tensor::ops::FloatElem<B>;

        let x = Tensor::<B, 1>::from_data([1, 2, 3], device);
        let y = Tensor::<B, 1>::from_data([4, 5, 6], device);

        let (x, y) = meshgrid(x, y);

        Tensor::<B, 2>::from_data([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], device)
            .into_data()
            .assert_approx_eq::<FT>(&x.to_data(), burn::tensor::Tolerance::default());

        Tensor::<B, 2>::from_data([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]], device)
            .into_data()
            .assert_approx_eq::<FT>(&y.to_data(), burn::tensor::Tolerance::default());
    }
}
