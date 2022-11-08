from image_process import *
from StyleTransfer import *
from architecture_layers import *
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


content_image_path = './images/bear.jpg'
style_image_path = './style_images/picasso.jpg'
model_outputs= {}
base_model = ['./checkpoints/vgg_notop', './checkpoints/resnet50_notop', './checkpoints/resnet152_notop', './checkpoints/xception_notop']
model_name = ['vgg','resnet50','resnet152', 'xception']
content_image = load_image(content_image_path, max_dim= 768)
style_image = load_image(style_image_path, max_dim= 768)
epochs = 50
steps = 20

content_layers, style_layers = model_layers(base_model[0])
print(style_layers)
print(content_layers)

print("Loading Content Image VGG...")
print("Loading Style Image VGG...")

my_model = StyleTransfer(base_model[0],content_layers,style_layers,model_name[0],content_image,style_image, style_weight=1e-2,learning_rate=0.02, content_weight=1e4)
image = tf.Variable(content_image)
print("Optimizing...")
images, train_total_loss, style_loss, content_loss, duration = my_model.optimize(epochs, steps, image)
model_outputs['vgg'] = [images, train_total_loss, style_loss, content_loss, duration]



content_layers, style_layers = model_layers(base_model[1])
print(style_layers)
print(content_layers)

print("Loading Content Image ResNet50...")
print("Loading Style Image ResNet50...")

my_model = StyleTransfer(base_model[1],content_layers,style_layers,model_name[1],content_image,style_image, style_weight=1e9, content_weight=1,learning_rate=0.01)
image = tf.Variable(content_image)
print("Optimizing...")
images, train_total_loss, style_loss, content_loss, duration = my_model.optimize(epochs, steps, image)
model_outputs['resnet50'] = [images, train_total_loss, style_loss, content_loss, duration]


content_layers, style_layers = model_layers(base_model[2])
print(style_layers)
print(content_layers)

print("Loading Content Image ResNet152...")
print("Loading Style Image ResNet152...")

my_model = StyleTransfer(base_model[2],content_layers,style_layers,model_name[2],content_image,style_image, style_weight=1e9, content_weight=1, learning_rate=0.01)
image = tf.Variable(content_image)
print("Optimizing...")
images, train_total_loss, style_loss, content_loss, duration = my_model.optimize(epochs, steps, image)
model_outputs['resnet152'] = [images, train_total_loss, style_loss, content_loss, duration]


content_layers, style_layers = model_layers(base_model[3])
print(style_layers)
print(content_layers)

print("Loading Content Image Xception...")
print("Loading Style Image Xception...")

my_model = StyleTransfer(base_model[3],content_layers,style_layers,model_name[3],content_image,style_image, style_weight=1e9, content_weight=1,learning_rate=0.01)
image = tf.Variable(content_image)
print("Optimizing...")
images, train_total_loss, style_loss, content_loss, duration = my_model.optimize(epochs, steps, image)
model_outputs['xception'] = [images, train_total_loss, style_loss, content_loss, duration]




# saving stylized images
for item in model_outputs.keys():
    for i,img in enumerate(model_outputs[item][0]):
        out_path ='./out_images/'  + content_image_path.split('/')[2].split('.')[0] + '_' + style_image_path.split('/')[2].split('.')[0] + '_' + item + '_' + str(i) +'.png'
        img.save(out_path)
#plt.show()

#plotting images per model per 10 epochs
#Prepare img checkerboard plot
#Params
nrows, ncols = 4, 5  # array of sub-plots
figsize = [30, 22]     # figure size, inches
  

# create figure (fig), and array of axes (ax)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
epoch_num = ['10','20','30','40','50']
# plot on each sub-plot 4x5 images
for i, axes in enumerate(ax.flat):
        rowid = i // ncols
        colid = i % ncols
        img = model_outputs[model_name[rowid]][0][10+ (10*colid)]
        axes.imshow(img)
        axes.set_title('Model: ' + model_name[rowid] + ' Epoch: ' + epoch_num[colid])

plt.show()


#plotting total loss, content loss and style loss for each model
f, axis = plt.subplots(3,4, figsize=(30,15))
axis[0][0].plot(range(len(model_outputs['vgg'][1])),np.array(model_outputs['vgg'][1]),color='indianred')
axis[0][1].plot(range(len(model_outputs['resnet50'][1])),np.array(model_outputs['resnet50'][1]),color='blue')
axis[0][2].plot(range(len(model_outputs['resnet152'][1])),np.array(model_outputs['resnet152'][1]),color='green')
axis[0][3].plot(range(len(model_outputs['xception'][1])),np.array(model_outputs['xception'][1]),color='darkorange')
axis[0][0].set_title('Total Loss VGG')
axis[0][1].set_title('Total Loss ResNet50')
axis[0][2].set_title('Total Loss ResNet152')
axis[0][3].set_title('Total Loss Xception')

axis[1][0].plot(range(len(model_outputs['vgg'][2])),np.array(model_outputs['vgg'][2]),color='indianred')
axis[1][1].plot(range(len(model_outputs['resnet50'][2])),np.array(model_outputs['resnet50'][2]),color='blue')
axis[1][2].plot(range(len(model_outputs['resnet152'][2])),np.array(model_outputs['resnet152'][2]),color='green')
axis[1][3].plot(range(len(model_outputs['xception'][2])),np.array(model_outputs['xception'][2]),color='darkorange')
axis[1][0].set_title('Style Loss VGG')
axis[1][1].set_title('Style Loss ResNet50')
axis[1][2].set_title('Style Loss ResNet152')
axis[1][3].set_title('Style Loss Xception')

axis[2][0].plot(range(len(model_outputs['vgg'][3])),np.array(model_outputs['vgg'][3]),color='indianred')
axis[2][1].plot(range(len(model_outputs['resnet50'][3])),np.array(model_outputs['resnet50'][3]),color='blue')
axis[2][2].plot(range(len(model_outputs['resnet152'][3])),np.array(model_outputs['resnet152'][3]),color='green')
axis[2][3].plot(range(len(model_outputs['xception'][3])),np.array(model_outputs['xception'][3]),color='darkorange')
axis[2][0].set_title('Content Loss VGG')
axis[2][1].set_title('Content Loss ResNet50')
axis[2][2].set_title('Content Loss ResNet152')
axis[2][3].set_title('Content Loss Xception')

plt.show()


# Duration vs total loss
f, axis = plt.subplots(1,4, figsize=(21,7))
axis[0].plot(model_outputs['vgg'][4], model_outputs['vgg'][1])
axis[0].set_title('Total Loss by Duration VGG')
axis[1].plot(model_outputs['resnet50'][4], model_outputs['resnet50'][1])
axis[1].set_title('Total Loss by Duration ResNet50')
axis[2].plot(model_outputs['resnet152'][4], model_outputs['resnet152'][1])
axis[2].set_title('Total Loss by Duration ResNet152')
axis[3].plot(model_outputs['xception'][4], model_outputs['xception'][1])
axis[3].set_title('Total Loss by Duration Xception')
plt.show()