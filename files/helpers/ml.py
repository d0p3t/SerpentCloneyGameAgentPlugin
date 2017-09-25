import numpy as np
import tensorflow as tf
import offshoot


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            name="",
        )

    return graph

def predict(tf_model, frame):

    categories = {1: {'id': 1, 'name': 'leaves'}}
    image = np.expand_dims(frame, axis=0)

    image_tensor = tf_model.get_tensor_by_name('image_tensor:0')
    detection_boxes = tf_model.get_tensor_by_name('detection_boxes:0')
    detection_scores = tf_model.get_tensor_by_name('detection_scores:0')
    detection_classes = tf_model.get_tensor_by_name('detection_classes:0')
    num_detections = tf_model.get_tensor_by_name('num_detections:0')

    output = [detection_boxes, detection_scores, detection_classes, num_detections]

    with tf.Session(graph=tf_model) as sess:
        (boxes, scores, classes, num) = sess.run(output, feed_dict={image_tensor: image})

        for i,b in enumerate(boxes[0]):
            if classes[0][i] == 1:
                class_name = categories[classes[0][i]]['name']
                if scores[0][i] >= 0.5:
                    top = boxes[0][i][0] * 576
                    left = boxes[0][i][1] * 1024
                    bottom = boxes[0][i][2] * 576
                    right = boxes[0][i][3] * 1024

                    box_left = (top, left), (bottom, left)
                    box_right = (top, right), (bottom, right)
                    box_top = (top, left), (top, right)
                    box_bottom = (bottom, left), (bottom, right)

                    return box_left, box_right, box_top, box_bottom
