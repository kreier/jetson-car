config_path, checkpoint_path = download_detection_model(MODEL, 'data')

frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path,
    score_threshold=0.3,
    iou_threshold=0.5,
    batch_size=1
)

import tensorflow.contrib.tensorrt as trt

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

with open('./data/trt_graph.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())

# Download the tensorRT graph .pb file from colab to your local machine.
from google.colab import files

files.download('./data/trt_graph.pb')

import tensorflow as tf

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

# The TensorRT inference graph file downloaded from Colab or your local machine.
pb_fname = "./model/trt_graph.pb"
trt_graph = get_frozen_graph(pb_fname)

input_names = ['image_tensor']

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

import cv2
IMAGE_PATH = "./data/dogs.jpg"
image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, (300, 300))

scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
    tf_input: image[None, ...]
})
boxes = boxes[0]  # index by 0 to remove batch dimension
scores = scores[0]
classes = classes[0]
num_detections = int(num_detections[0])

from IPython.display import Image as DisplayImage

# Boxes unit in pixels (image coordinates).
boxes_pixels = []
for i in range(num_detections):
    # scale box to image coordinates
    box = boxes[i] * np.array([image.shape[0],
                               image.shape[1], image.shape[0], image.shape[1]])
    box = np.round(box).astype(int)
    boxes_pixels.append(box)
boxes_pixels = np.array(boxes_pixels)

# Remove overlapping boxes with non-max suppression, return picked indexes.
pick = non_max_suppression(boxes_pixels, scores[:num_detections], 0.5)


for i in pick:
    box = boxes_pixels[i]
    box = np.round(box).astype(int)
    # Draw bounding box.
    image = cv2.rectangle(
        image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
    label = "{}:{:.2f}".format(int(classes[i]), scores[i])
    # Draw label (class index and probability).
    draw_label(image, (box[1], box[0]), label)

# Save and display the labeled image.
save_image(image[:, :, ::-1])
DisplayImage(filename="./data/img.png")