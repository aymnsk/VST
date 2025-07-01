from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from io import BytesIO

app = Flask(__name__)
CORS(app)

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_img(img):
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = 256.0 / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    return img[tf.newaxis, :]

def crop_center(image):
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    return tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = tf.cast(tensor, tf.uint8)
    if tensor.shape[0] == 1:
        tensor = tensor[0]
    return Image.fromarray(tensor.numpy())

def load_content_style_img(style_img, content_img):
    style_img = np.array(style_img)
    content_img = np.array(content_img)

    width, height = content_img.shape[1], content_img.shape[0]
    content_img = load_img(content_img)
    content_img = tf.image.resize(content_img, (width, height), preserve_aspect_ratio=True)

    style_img = load_img(style_img)
    style_img = crop_center(style_img)
    style_img = tf.image.resize(style_img, (256, 256), preserve_aspect_ratio=True)
    style_img = tf.nn.avg_pool(style_img, ksize=[3, 3], strides=[1, 1], padding='SAME')

    return style_img, content_img

@app.route('/', methods=['POST'])
def stylize():
    if 'style' not in request.files or 'content' not in request.files:
        return "Missing images", 400

    style_file = request.files['style']
    content_file = request.files['content']

    style_img = Image.open(style_file).convert('RGB')
    content_img = Image.open(content_file).convert('RGB')

    style_tensor, content_tensor = load_content_style_img(style_img, content_img)
    outputs = hub_module(tf.constant(content_tensor), tf.constant(style_tensor))
    stylized_tensor = outputs[0]

    output_img = tensor_to_image(stylized_tensor)

    img_io = BytesIO()
    output_img.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

