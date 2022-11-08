import tensorflow as tf

def model_layers(model_name):
    content_layers = []
    style_layers = []
    model = tf.keras.models.load_model(model_name)
    if model_name == './checkpoints/vgg_notop':
        content_layers.append('block5_conv2')
        style_layers.extend(['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'])

    elif model_name == './checkpoints/xception_notop':
        content_layers.append('block13_sepconv2_act')
        style_layers.extend(['block2_sepconv2_act','block3_sepconv2_act','block4_sepconv2_act','block5_sepconv3_act',
        'block6_sepconv3_act','block7_sepconv3_act','block8_sepconv3_act','block9_sepconv3_act','block10_sepconv3_act','block11_sepconv3_act','block12_sepconv3_act','block13_sepconv2_act','block14_sepconv2_act'])

    elif model_name =='./checkpoints/resnet50_notop':
        content_layers.append('conv4_block6_2_relu')
        style_layers.extend(['conv2_block3_2_relu', 'conv3_block4_2_relu','conv4_block6_2_relu','conv5_block3_2_relu'])

    elif model_name =='./checkpoints/resnet152_notop':
        content_layers.append('conv4_block36_2_relu')
        style_layers.extend(['conv2_block3_2_relu', 'conv3_block8_2_relu','conv4_block36_2_relu','conv5_block3_2_relu'])

        
    return content_layers, style_layers

        # for idx in range(len(model.layers)):
        #     if 'sepconv1' in model.get_layer(index = idx).name and not 'sepconv1_' in model.get_layer(index = idx).name:
        #         #if idx % 3 == 0:
        #         style_layers.append(model.get_layer(index = idx).name)

        #style_layers.extend(['conv2_block2_1_bn','conv3_block1_0_bn','conv3_block1_0_conv','conv4_block5_2_bn'])


    # elif model_name == './checkpoints/nasnet_notop':
    #     content_layers.extend(['separable_conv_2_normal_left2_18','separable_conv_2_normal_right2_18'])
    #     for idx in range(len(model.layers)):
    #         if 'separable_conv' in model.get_layer(index = idx).name and ('normal_left1' in model.get_layer(index = idx).name or 'normal_right1' in model.get_layer(index = idx).name) and not '_bn_' in model.get_layer(index = idx).name:
    #             if idx % 20 == 0:
    #                     style_layers.append(model.get_layer(index = idx).name)
    # elif model_name == './checkpoints/inceptresnet_notop':
    #     content_layers.append('block8_2_conv')
    #     for idx in range(len(model.layers)):
    #         if 'conv' in model.get_layer(index = idx).name and 'block' in model.get_layer(index = idx).name:
    #             #if idx % 5 == 0:
    #              style_layers.append(model.get_layer(index = idx).name)