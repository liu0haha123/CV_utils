import tensorflow  as tf
import keras
from keras.callbacks import TensorBoard
import  scipy
from scipy import io
import os
import scipy.ndimage as  nd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.backend.tensorflow_backend import set_session
import numpy as np
from keras import Model,Sequential
import glob
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
# todo 炸显存again

def build_generator():
    """
    Create a Generator Model with hyperparameters values defined as follows
    """
    z_size = 200
    gen_filters = [512, 256, 128, 64, 1]
    gen_kernel_sizes = [4, 4, 4, 4, 4]
    gen_strides = [1, 2, 2, 2, 2]
    gen_input_shape = (1, 1, 1, z_size)
    gen_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
    gen_convolutional_blocks = 5

    input_layer = Input(shape=gen_input_shape)

    # First 3D transpose convolution(or 3D deconvolution) block
    a = Deconv3D(filters=gen_filters[0],
                 kernel_size=gen_kernel_sizes[0],
                 strides=gen_strides[0])(input_layer)
    a = BatchNormalization()(a, training=True)
    a = Activation(activation='relu')(a)

    # Next 4 3D transpose convolution(or 3D deconvolution) blocks
    for i in range(gen_convolutional_blocks - 1):
        a = Deconv3D(filters=gen_filters[i + 1],
                     kernel_size=gen_kernel_sizes[i + 1],
                     strides=gen_strides[i + 1], padding='same')(a)
        a = BatchNormalization()(a, training=True)
        a = Activation(activation=gen_activations[i + 1])(a)

    gen_model = Model(inputs=[input_layer], outputs=[a])
    return gen_model


def build_discriminator():
    """
    Create a Discriminator Model using hyperparameters values defined as follows
    """

    dis_input_shape = (64, 64, 64, 1)
    dis_filters = [64, 128, 256, 512, 1]
    dis_kernel_sizes = [4, 4, 4, 4, 4]
    dis_strides = [2, 2, 2, 2, 1]
    dis_paddings = ['same', 'same', 'same', 'same', 'valid']
    dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
    dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu',
                       'leaky_relu', 'sigmoid']
    dis_convolutional_blocks = 5

    dis_input_layer = Input(shape=dis_input_shape)

    # The first 3D Convolutional block
    a = Conv3D(filters=dis_filters[0],
               kernel_size=dis_kernel_sizes[0],
               strides=dis_strides[0],
               padding=dis_paddings[0])(dis_input_layer)
    # a = BatchNormalization()(a, training=True)
    a = LeakyReLU(dis_alphas[0])(a)

    # Next 4 3D Convolutional Blocks
    for i in range(dis_convolutional_blocks - 1):
        a = Conv3D(filters=dis_filters[i + 1],
                   kernel_size=dis_kernel_sizes[i + 1],
                   strides=dis_strides[i + 1],
                   padding=dis_paddings[i + 1])(a)
        a = BatchNormalization()(a, training=True)
        if dis_activations[i + 1] == 'leaky_relu':
            a = LeakyReLU(dis_alphas[i + 1])(a)
        elif dis_activations[i + 1] == 'sigmoid':
            a = Activation(activation='sigmoid')(a)

    dis_model = Model(inputs=[dis_input_layer], outputs=[a])
    return dis_model
def write_log(callback,name,value,batch_no):
    summary =tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary,batch_no)
    callback.writer.flush()

def getVoxelsFromMat(path,CUBE_LEN = 64):
    voxels = io.loadmat(path)["instance"]

    voxels = np.pad(voxels,(1,1),"constant",constant_values=(0,0))
    if CUBE_LEN!=32 and CUBE_LEN==64:
        voxels = nd.zoom(voxels,(2,2,2),mode="constant",order=0)
    return  voxels



def get3DImages(data_dir):
    all_files = np.random.choice(glob.glob(data_dir), size=10)
    # all_files = glob.glob(data_dir)
    all_volumes = np.asarray([getVoxelsFromMat(f) for f in all_files], dtype=np.bool)
    return all_volumes


def saveFromVoxels(voxels, path):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig(path)


def plotAndSaveVoxel(file_path, voxel):
    """
    Plot a voxel
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(voxel, edgecolor="red")
    # plt.show()
    plt.savefig(file_path)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    gen_LR = 0.0025
    dis_LR = 0.00001
    import time
    obj_name = "chair"
    beta = 0.5
    batch_size = 8
    z_size = 200
    data_dir = "3DShapeNets/volumetric_data/{}/30/train/*.mat".format(obj_name)
    generated_volume_dir = "Generated_volume"
    log_dir = "logs"

    mode ="train"
    epoches = 10
    #初始化模型
    generator = build_generator()
    discriminator = build_discriminator()

    gen_optim = Adam(lr=gen_LR, beta_1=beta)
    dis_optim = Adam(lr=dis_LR, beta_1=beta)

    generator.compile(loss="binary_crossentropy", optimizer="adam")
    discriminator.compile(loss="binary_crossentropy", optimizer=dis_optim)
    #完整的GAN结构
    discriminator.trainable = False
    input_layer = Input(shape=(1,1,1,z_size))
    generated_volumes = generator(input_layer)
    validity = discriminator(generated_volumes)
    # 完整的网络
    GAN_model = Model(inputs=[input_layer],outputs=[validity])
    GAN_model.compile(loss='binary_crossentropy', optimizer=gen_optim)

    print("Loading data...")
    volumes = get3DImages(data_dir=data_dir)
    volumes = volumes[..., np.newaxis].astype(np.float)
    print("Data loaded...")

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    labels_real = np.reshape(np.ones((batch_size,)), (-1, 1, 1, 1, 1))
    labels_fake = np.reshape(np.zeros((batch_size,)), (-1, 1, 1, 1, 1))

    if mode == "train":
        for epoch in range(epoches):
            print("Epoch:",epoch)

            gen_losses = []
            dis_losses = []

            number_of_batches = int(volumes.shape[0] / batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:", index + 1)

                z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                volumes_batch = volumes[index * batch_size:(index + 1) * batch_size, :, :, :]

                # 利用噪声和生成器随机生成图像
                gen_volumes = generator.predict_on_batch(z_sample)

                """
                训练判别器:
                分别用真实图像和生成图像训练判别器，损失函数取二者平均，先训练判别器的区分能力
                """
                discriminator.trainable = True
                if index%2==0:
                    loss_real = discriminator.train_on_batch(volumes_batch,labels_real)
                    loss_fake = discriminator.train_on_batch(gen_volumes,labels_fake)
                    d_loss = 0.5 * np.add(loss_real, loss_fake)
                    print("d_loss:{}".format(d_loss))

                else:
                    d_loss = 0.0

                discriminator.trainable = False

                """
                训练生成器:
                固定判别器，然后训练完整的对抗生成网络
                """
                z = np.random(0,0.33,size=[batch_size,1,1,1,z_size])
                g_loss = GAN_model.trainable(z,labels_real)
                print("g_loss{}".format(g_loss))

                gen_losses.append(g_loss)
                dis_losses.append(d_loss)

                # Every 10th mini-batch, generate volumes and save them
                if index % 10 == 0:
                    z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                    generated_volumes = generator.predict(z_sample2, verbose=3)
                    for i, generated_volume in enumerate(generated_volumes[:5]):
                        voxels = np.squeeze(generated_volume)
                        voxels[voxels < 0.5] = 0.
                        voxels[voxels >= 0.5] = 1.
                        saveFromVoxels(voxels, "results/img_{}_{}_{}".format(epoch, index, i))

                # Write losses to Tensorboard
            write_log(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
            write_log(tensorboard, 'd_loss', np.mean(dis_losses), epoch)

        """
        Save models
        """
        generator.save_weights(os.path.join("models", "generator_weights.h5"))
        discriminator.save_weights(os.path.join("models", "discriminator_weights.h5"))

    if mode == 'predict':
        # Create models
        generator = build_generator()
        discriminator = build_discriminator()

        # Load model weights
        generator.load_weights(os.path.join("models", "generator_weights.h5"), True)
        discriminator.load_weights(os.path.join("models", "discriminator_weights.h5"), True)

        # Generate 3D models
        z_sample = np.random.normal(0, 1, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
        generated_volumes = generator.predict(z_sample, verbose=3)

        for i, generated_volume in enumerate(generated_volumes[:2]):
            voxels = np.squeeze(generated_volume)
            voxels[voxels < 0.5] = 0.
            voxels[voxels >= 0.5] = 1.
            saveFromVoxels(voxels, "results/gen_{}".format(i))

