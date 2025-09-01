/// The Microsoft COCO (Common Objects in Context) is a novel dataset with the purpose of improving 
/// the state-of-the-art in object detection by aligning the problem within the larger framework of 
/// scene understanding.
///
/// Although the dataset is a great resource for images, it has also been found to have numerous 
/// erroneous entries. Be careful when using it for important work, as it could lead to unintended 
/// outcomes. Interestingly enough, a lot of the datasets available on the internet do not contain a 
/// complete list of all the actual labels found in the datasets located at https:///cocodataset.org/ 
/// and only contain a subset of 80 of the actual full list of 92. This can cause issues when using 
/// the list as a look up table for training or inference. This is because in the actual datasets
/// only 80 of the 92 listed classes have actually been labeled. Categories like “street sign”,
/// “shoe”, “hat”, ect.. are missing from the annotations.
///
/// “What object categories / labels are in COCO Dataset? (n.d.).”
/// Amikelive | Technology Blog. 
/// https:///tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
///
/// "How I found nearly 300,000 errors in MS COCO"
/// Jamie Murdoch. (n.d.).
/// Medium.
/// https:///medium.com/@jamie_34747/how-i-found-nearly-300-000-errors-in-ms-coco-79d382edf22b
///
/// “Microsoft COCO: Common Objects in Context”
/// Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro
/// Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár
/// https:///arxiv.org/abs/1405.0312
/// 
pub const COCO_LABELS: [&str; 92] = [
    "background", // Placeholder
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "hair brush",
];
