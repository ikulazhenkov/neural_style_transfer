import tensorflow as tf

class StyleContentModel(tf.keras.models.Model):
        def __init__(self, style_layers, content_layers, model_name, base_model):
            super(StyleContentModel, self).__init__()
            self.base_model = base_model
            self.model = self.model_layers(content_layers + style_layers)
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.model_name = model_name
            #self.vgg.trainable = False

        def model_layers(self, names):
            base_model = tf.keras.models.load_model(self.base_model, compile=False)
            base_model.trainable = False
            outputs = [base_model.get_layer(name).output for name in names]
            model = tf.keras.Model([base_model.input], outputs)
            return model

        def gram_matrix(self, input_tensor):
            result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
            input_shape = tf.shape(input_tensor)
            num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
            return result/(num_locations)

        def call(self, inputs):
            "Expects float input in [0,1]"
            inputs = inputs*255.0
            if self.model_name == 'vgg':
                preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
            # elif self.model_name == 'nasnet':
            #     preprocessed_input = tf.keras.applications.nasnet.preprocess_input(inputs)
            elif self.model_name == 'xception':
                preprocessed_input = tf.keras.applications.xception.preprocess_input(inputs)
            # elif self.model_name == 'inception':
            #     preprocessed_input = tf.keras.applications.inception_resnet_v2.preprocess_input(inputs)
            elif self.model_name == 'resnet50' or self.model_name == 'resnet152':
                preprocessed_input = tf.keras.applications.resnet.preprocess_input(inputs)
            else:
                print('CRITICAL ERROR, MODEL NOT SUPPORTED')
            outputs = self.model(preprocessed_input)
            style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                            outputs[self.num_style_layers:])

            style_outputs = [self.gram_matrix(style_output)
                            for style_output in style_outputs]

            content_dict = {content_name: value
                            for content_name, value
                            in zip(self.content_layers, content_outputs)}

            style_dict = {style_name: value
                        for style_name, value
                        in zip(self.style_layers, style_outputs)}

            return {'content': content_dict, 'style': style_dict}
