
/// Convolutional feature map layers used in the SSD (Single Shot MultiBox Detector) architecture.
///
/// Each variant represents a specific convolutional layer from which the SSD model
/// generates default (anchor) boxes and predictions.  
/// The layers are listed in the order they appear in the detection pipeline,
/// from early, high-resolution layers to deeper, lower-resolution layers.
/// Relevant excerpt from
/// 
/// “SSD: Single Shot MultiBox Detector”
/// Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
///          Scott Reed, Cheng-Yang Fu, Alexander C. Berg
/// Link (official): https://arxiv.org/abs/1512.02325
/// 
/// - Liu et al. - Pg 3
/// 
/// Multi-scale feature maps for detection
///
/// We add convolutional feature layers to the end of the truncated base network. These
/// layers decrease in size progressively and allow predictions of detections at multiple
/// scales. The convolutional model for predicting detections is different for each feature
/// layer (cf Overfeat[4] and YOLO[5] that operate on a single scale feature map).
///
#[repr(usize)]
#[derive(Debug, Clone, Copy)]
pub enum SSDConvLayers {
    /// `Conv4_3` — fourth VGG16 convolution block, 3×3 kernel.
    Conv4_3,
    /// `Conv7` — converted fully connected layer (originally FC7 in VGG16).
    Conv7,
    /// `Conv8_2` — additional SSD feature extraction layer.
    Conv8_2,
    /// `Conv9_2` — additional SSD feature extraction layer.
    Conv9_2,
    /// `Conv10_2` — additional SSD feature extraction layer.
    Conv10_2,
    /// `Conv11_2` — final SSD feature extraction layer.
    Conv11_2,
}

impl SSDConvLayers {
    /// Returns all SSD convolution layers in the order used for multi-scale detection.
    ///
    /// This order reflects how SSD aggregates predictions from multiple
    /// feature map scales — starting from higher-resolution layers (`Conv4_3`)
    /// and moving to progressively smaller spatial resolutions.
    ///
    pub fn as_list() -> Vec<SSDConvLayers> {
        vec![
            SSDConvLayers::Conv4_3,
            SSDConvLayers::Conv7,
            SSDConvLayers::Conv8_2,
            SSDConvLayers::Conv9_2,
            SSDConvLayers::Conv10_2,
            SSDConvLayers::Conv11_2,
        ]
    }

    /// Returns the total number of convolution layers used for SSD predictions.
    ///
    pub fn count () -> usize {
        Self::as_list().len()
    }

    /// Returns a numeric identifier for the layer variant.
    ///
    /// This is useful for mapping SSD layers to dataset annotations,  
    /// logging, or serialized model formats where layers are numbered.
    ///
    pub fn get_id(&self) -> usize {
        let val = match self {
            SSDConvLayers::Conv4_3 => SSDConvLayers::Conv4_3 as usize,
            SSDConvLayers::Conv7 => SSDConvLayers::Conv7 as usize,
            SSDConvLayers::Conv8_2 => SSDConvLayers::Conv8_2 as usize,
            SSDConvLayers::Conv9_2 => SSDConvLayers::Conv9_2 as usize,
            SSDConvLayers::Conv10_2 => SSDConvLayers::Conv10_2 as usize,
            SSDConvLayers::Conv11_2 => SSDConvLayers::Conv11_2 as usize,
        };

        val + 1
    }

    /// Returns the number of output channels (feature depth) for the given layer.
    ///
    /// This value is used when constructing the SSD prediction heads,  
    /// as it determines the convolutional input size for both the classifier
    /// and bounding-box regression layers.
    ///
    pub fn output_size(&self) -> usize {
        match self {
            SSDConvLayers::Conv4_3 => 512,
            SSDConvLayers::Conv7 => 1024,
            SSDConvLayers::Conv8_2 => 512,
            SSDConvLayers::Conv9_2 => 256,
            SSDConvLayers::Conv10_2 => 256,
            SSDConvLayers::Conv11_2 => 256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_list_order_and_contents() {
        let layers = SSDConvLayers::as_list();
        assert_eq!(layers.len(), 6);
        assert!(matches!(layers[0], SSDConvLayers::Conv4_3));
        assert!(matches!(layers[1], SSDConvLayers::Conv7));
        assert!(matches!(layers[2], SSDConvLayers::Conv8_2));
        assert!(matches!(layers[3], SSDConvLayers::Conv9_2));
        assert!(matches!(layers[4], SSDConvLayers::Conv10_2));
        assert!(matches!(layers[5], SSDConvLayers::Conv11_2));
    }

    #[test]
    fn test_get_id_is_one_based() {
        let layers = SSDConvLayers::as_list();
        for (index, layer) in layers.iter().enumerate() {
            assert_eq!(layer.get_id(), index + 1);
        }
    }

    #[test]
    fn test_output_size_values() {
        assert_eq!(SSDConvLayers::Conv4_3.output_size(), 512);
        assert_eq!(SSDConvLayers::Conv7.output_size(), 1024);
        assert_eq!(SSDConvLayers::Conv8_2.output_size(), 512);
        assert_eq!(SSDConvLayers::Conv9_2.output_size(), 256);
        assert_eq!(SSDConvLayers::Conv10_2.output_size(), 256);
        assert_eq!(SSDConvLayers::Conv11_2.output_size(), 256);
    }
}
