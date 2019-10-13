from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_gan as tfgan
import tensorflow as tf
import time
import tensorflow_datasets as tfds
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

numpyTrain = np.load("picsScaled.npy", allow_pickle=True)
numpyPredict = np.load("picsScaled.npy", allow_pickle=True)

datagen = ImageDataGenerator(
      rotation_range=25,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.1,
      horizontal_flip=True,
      fill_mode='nearest')

def input_fn(mode, params):
    assert 'batch_size' in params
    assert 'noise_dims' in params
    bs = params['batch_size']
    nd = params['noise_dims']

    split = 'train' if mode == tf.estimator.ModeKeys.TRAIN else 'test'
    shuffle = (mode == tf.estimator.ModeKeys.TRAIN)
    just_noise = (mode == tf.estimator.ModeKeys.PREDICT)
    
    noise_ds = (tf.data.Dataset.from_tensors(0).repeat()
              .map(lambda _: tf.random_normal([bs, nd])))


    if just_noise:
        return noise_ds
    
    #images_ds = tf.data.Dataset.from_tensor_slices(datagen.flow(numpyTrain, batch_size=bs).__next__()).repeat(100).map(lambda x: tf.dtypes.cast(x, tf.float32))
    images_ds = tf.data.Dataset.from_tensor_slices(numpyTrain).repeat(100).map(lambda x: tf.dtypes.cast(x, tf.float32))

    if shuffle:
        images_ds = images_ds.shuffle(
            buffer_size=10000, reshuffle_each_iteration=True)
    images_ds = (images_ds.batch(bs, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))

    return tf.data.Dataset.zip((noise_ds, images_ds))    

def _dense(inputs, units, l2_weight):
    return tf.layers.dense(
        inputs, units, None,
        kernel_initializer = tf.keras.initializers.glorot_uniform,
        kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
        bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))

def _batch_norm(inputs, is_training):
    return tf.layers.batch_normalization(
        inputs, momentum=0.999, epsilon=0.001, training=is_training
    )

def _deconv2d(inputs, filters, kernel_size, stride, l2_weight):
    return tf.layers.conv2d_transpose(
        inputs, filters, [kernel_size, kernel_size], strides=[stride,stride],
        activation=tf.nn.relu, padding="same",
        kernel_initializer = tf.keras.initializers.glorot_uniform,
        kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
        bias_regularizer=tf.keras.regularizers.l2(l=l2_weight)
    )

def _conv2d(inputs, filters, kernel_size, stride, l2_weight):
    return tf.layers.conv2d(
        inputs, filters, [kernel_size, kernel_size], strides=[stride,stride],
        activation=None, padding="same",
        kernel_initializer = tf.keras.initializers.glorot_uniform,
        kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
        bias_regularizer=tf.keras.regularizers.l2(l=l2_weight)
    )

def unconditional_generator(noise, mode, weight_decay=2.5e-5):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    net = _dense(noise, 1024, weight_decay)
    net = _batch_norm(net, is_training)
    net = tf.nn.relu(net)

    net = _dense(net, 16 * 16 * 256, weight_decay)
    net = _batch_norm(net, is_training)
    net = tf.nn.relu(net)

    net = tf.reshape(net, [-1, 16, 16, 256])
    net = _deconv2d(net, 64, 4, 2, weight_decay)
    net = _deconv2d(net, 64, 4, 2, weight_decay)

    net = _conv2d(net, 3, 4, 1, 0.0)
    net = tf.tanh(net)

    return net

_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def unconditional_discriminator(img, unused_conditioning, mode, weight_decay=2.5e-5):
    del unused_conditioning
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    net = _conv2d(img, 64, 4, 2, weight_decay)
    net = _leaky_relu(net)

    net = _conv2d(net, 128, 4, 2, weight_decay)
    net = _leaky_relu(net)

    net = tf.layers.flatten(net)

    net = _dense(net, 1024, weight_decay)
    net = _batch_norm(net, is_training)
    net = _leaky_relu(net)

    net = _dense(net, 1, weight_decay)

    return net

def get_eval_metric_options_fn(gan_model):
    real_data_logits = tf.reduce_mean(gan_model.discriminator_real_outputs)
    gen_data_logits = tf.reduce_mean(gan_model.discriminator_gen_outputs)
    return {
        'real_data_logits': tf.metrics.mean(real_data_logits),
        'gen_data_logits': tf.metrics.mean(gen_data_logits)
    }

params = {'batch_size': 64, 'noise_dims':64}
with tf.Graph().as_default():
  ds = input_fn(tf.estimator.ModeKeys.TRAIN, params)
  numpy_imgs = tfds.as_numpy(ds).__next__()[1]
img_grid = tfgan.eval.python_image_grid(numpy_imgs, grid_shape=(8, 8))
plt.axis('off')
plt.imshow(np.squeeze(img_grid))
plt.show()

train_batch_size = 64
noise_dimensions = 32
generator_lr = 0.0001
discriminator_lr = 0.00005

def gen_opt():
    gstep = tf.train.get_or_create_global_step()
    base_lr = generator_lr
    lr = tf.cond(gstep < 1000, lambda: base_lr, lambda: base_lr / 2.0)
    return tf.train.AdamOptimizer(lr, 0.5)

gan_estimator = tfgan.estimator.GANEstimator(
    generator_fn = unconditional_generator,
    discriminator_fn = unconditional_discriminator,
    generator_loss_fn = tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn = tfgan.losses.wasserstein_discriminator_loss,
    params = {"batch_size": train_batch_size, "noise_dims": noise_dimensions},
    generator_optimizer = gen_opt,
    discriminator_optimizer = tf.train.AdamOptimizer(discriminator_lr, 0.5),
    get_eval_metric_ops_fn = get_eval_metric_options_fn
)

steps_per_eval = 1000
max_train_steps = 30000
batches_for_eval_metrics = 100

steps = []
real_logits, fake_logits = [], []

current_step = 0
start_time = time.time()

while current_step < max_train_steps:
    next_step = min(current_step + steps_per_eval, max_train_steps)

    start = time.time()
    gan_estimator.train(input_fn, max_steps=next_step)
    steps_taken = next_step - current_step
    time_taken = time.time() - start
    print('Time since start: %.2f min' % ((time.time() - start_time) / 60.0))
    print('Trained from step %i to %i in %.2f steps / sec' % (
        current_step, next_step, steps_taken / time_taken))
    current_step = next_step

    metrics = gan_estimator.evaluate(input_fn, steps=batches_for_eval_metrics)
    steps.append(current_step)
    real_logits.append(metrics['real_data_logits'])
    fake_logits.append(metrics['gen_data_logits'])

    print('Average discriminator output on Real: %.2f  Fake: %.2f' % (
      real_logits[-1], fake_logits[-1]))

    # Vizualize some images.
    iterator = gan_estimator.predict(
        input_fn, hooks=[tf.train.StopAtStepHook(num_steps=21)])
    try:
        imgs = np.array([iterator.__next__() for _ in range(20)])
    except StopIteration:
        pass
    for pic in imgs:
        for row in pic:
            for col in row:
                col[0] = (col[0] + 1) / 2
                col[1] = (col[1] + 1) / 2
                col[2] = (col[2] + 1) / 2
    tiled = tfgan.eval.python_image_grid(imgs, grid_shape=(2, 10))
    plt.axis('off')
    plt.imshow(np.squeeze(tiled))
    plt.savefig(f".\\dogGeneratorFigures\\{str(time.time())}-{current_step}steps.png")