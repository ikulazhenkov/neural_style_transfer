import numpy as np
import tensorflow as tf
import time
from tensorflow.python.keras.applications.densenet import preprocess_input
from image_process import tensor_to_image
tf.config.run_functions_eagerly(True)

from StyleContentModel import *

class StyleTransfer:
    def __init__(self, base_model, content_layers, style_layers, model_name, content_image, style_image,
                style_weight=1e-2, content_weight=1e4, learning_rate= 0.02, total_var_weight=30):
        self.model_name = model_name
        self.base_model = tf.keras.models.load_model(base_model, compile=False)
        self.base_model.trainable = False
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.content_image = content_image
        self.style_image = style_image
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.learning_rate = learning_rate
        self.total_var_weight = total_var_weight
        self.extractor = StyleContentModel(style_layers=self.style_layers, content_layers=self.content_layers, model_name=self.model_name, base_model=base_model)

    def get_targets(self):
        style_targets = self.extractor(self.style_image)['style']
        content_targets = self.extractor(self.content_image)['content']
        return style_targets, content_targets

    def style_loss(self, outputs,targets):
        style_loss = tf.add_n([tf.reduce_mean((outputs[name]-targets[name])**2) 
                            for name in outputs.keys()])
        style_loss *= self.style_weight / self.num_style_layers
        return style_loss

    def content_loss(self, outputs, targets):
        content_loss = tf.add_n([tf.reduce_mean((outputs[name]-targets[name])**2) 
                                for name in outputs.keys()])
        content_loss *= self.content_weight / self.num_content_layers
        return content_loss

    def style_content_loss(self, image):
        outputs = self.extractor(image)
        style_targets, targets = self.get_targets()
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = self.style_loss(style_outputs, style_targets)
        content_loss = self.content_loss(content_outputs, targets)
        variation_loss = self.total_var_weight*tf.image.total_variation(image)
        total_loss = style_loss + content_loss + variation_loss
        return {'total_loss':total_loss, 'style_loss':style_loss, 'content_loss': content_loss}

    def clip_0_1(self, image):
      return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            loss = self.style_content_loss(image)
            total_loss = loss['total_loss']
        opt = tf.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.99, epsilon=1e-1)
        grad = tape.gradient(total_loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(self.clip_0_1(image))
        return loss

    def optimize(self, epochs, steps, image):
        total_loss_hist = []
        style_loss_hist = []
        content_loss_hist = []
        duration = []
        start = time.time()
        step = 0
        images = []
        for n in range(epochs):
            for m in range(steps):
                if step % steps== 0:
                    images.append(tensor_to_image(image))
                step += 1
                loss = self.train_step(image)
                total_loss_hist.append(loss['total_loss'])
                style_loss_hist.append(loss['style_loss'])
                content_loss_hist.append(loss['content_loss'])
                duration.append(time.time() - start)
                image.assign(self.clip_0_1(image))
                print(">", end='',flush=True)
            print("Train step: {}".format(step))
        images.append(tensor_to_image(image))
        end = time.time()
        print("Total time: {:.1f}".format(end-start))
        return images, total_loss_hist, style_loss_hist, content_loss_hist, duration