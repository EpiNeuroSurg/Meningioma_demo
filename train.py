import math
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import argparse
import Models , LoadBatches
from Models.Segnet_crf_res import segnet_crf_res
from Models.VGGSegnet import VGGSegnet
from Models.VGGUnet import VGGUnet
from Models.VGGUnet import VGGUnet2
from Models.FCN8 import FCN8
from Models.FCN32 import FCN32
from Models.Segnet import segnet
from Models.Segnet_transpose import segnet_transposed
from Models.Segnet_res import segnet_res


# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.000001
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "")
parser.add_argument("--val_annotations", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 100 )
parser.add_argument("--batch_size", type = int, default = 1 )
parser.add_argument("--val_batch_size", type = int, default = 1 )
parser.add_argument("--load_weights", type = str , default = "data/vgg16_weights_th_dim_ordering_th_kernels.h5")

parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = { 'vgg_segnet':VGGSegnet , 'vgg_unet':VGGUnet , 'vgg_unet2':VGGUnet2 , 'fcn8':FCN8 , 'fcn32':FCN32, 'segnet':segnet, 'segnet_transposed':segnet_transposed, 'segnet_res':segnet_res, 'segnet_res_crf':segnet_crf_res}
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.compile(loss='categorical_crossentropy',
      optimizer= optimizer_name ,
      metrics=['accuracy'])


#if len( load_weights ) > 0:
#	m.load_weights(load_weights)


lrate = LearningRateScheduler(step_decay)
filepath="weights_360_480_res_with_crf.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]



print "Model output shape" ,  m.output_shape

output_height = m.outputHeight
output_width = m.outputWidth
class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 0.6823, 6.2478, 7.3614]
G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


if validate:
	G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

if not validate:
	for ep in range( epochs ):
		m.fit_generator( G , 512  , epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep ) )
		m.save( save_weights_path + ".model." + str( ep ) )
else:
	m.fit_generator( G , 512  , validation_data=G2 , validation_steps=200 , callbacks=callbacks_list,class_weight=class_weighting, epochs=epochs, verbose=1)
