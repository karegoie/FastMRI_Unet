# -*- coding: utf-8 -*-
"""Teleo Tamgu.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FJT9q6WBuhRREhwOLGn77A_VP8ZNJZEN

# 처음부터 building
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py

import os
from pathlib import Path
from glob import glob
from tqdm import tqdm


NUM_val = 10
NUM_train = 407 - NUM_val  # MAX 407
NUM_test = 5  # MAX 5


mri_train_fnames = os.listdir('./Data/train')
base_train_dir = Path('./Data/train')


mri_val_fnames = mri_train_fnames[:NUM_val]
mri_train_fnames = mri_train_fnames[NUM_val:]

print(np.shape(mri_train_fnames), np.shape(mri_val_fnames))

os.makedirs('./Data/image', exist_ok=True)
base_dir = Path('./Data/image')

# 여기 주석

def data_train_load(NUM, target, dir):
    cnt = 0
    for i in tqdm(range(0, NUM)):
        os.makedirs(f'{base_dir}/{dir}', exist_ok=True)
        temp_dir = os.path.join(base_train_dir, mri_train_fnames[i])
        temp = h5py.File(temp_dir, 'r')
        if (np.array(temp[f'{target}']).size == 16 * 384 * 384):
            for j in range(0, 16):
                temp_arr = np.array(temp[f'{target}'])[j, :, :]
                mri_train = np.zeros([384, 384], np.float32)
                mri_train = temp_arr
                np.save(f'{base_dir}/{dir}/{cnt}.npy', mri_train)
                cnt += 1


data_train_load(NUM_train, 'image_input', 'mri_train_input')
data_train_load(NUM_val, 'image_input', 'mri_val_input')
data_train_load(NUM_train, 'image_label', 'mri_train_label')
data_train_load(NUM_val, 'image_label', 'mri_val_label')

mri_test_fnames = os.listdir('./Data/val')
base_test_dir = Path('./Data/val')

print(np.shape(mri_test_fnames))


def data_test_load(NUM, target, dir):
    cnt = 0
    for i in tqdm(range(0, NUM)):
        os.makedirs(f'{base_dir}/{dir}', exist_ok=True)
        temp_dir = os.path.join(base_train_dir, mri_train_fnames[i])
        temp = h5py.File(temp_dir, 'r')
        if (np.array(temp[f'{target}']).size == 16 * 384 * 384):
            for j in range(0, 16):
                temp_arr = np.array(temp[f'{target}'])[j, :, :]
                mri_test = np.zeros([384, 384], np.float32)
                mri_test = temp_arr
                np.save(f'{base_dir}/{dir}/{cnt}.npy', mri_test)
                cnt += 1


data_test_load(NUM_test, 'image_input', 'mri_test_input')
data_test_load(NUM_test, 'image_label', 'mri_test_label')

# 여기 주석

# glob 모듈을 이용해서 파일의 경로를 담은 리스트를 지정.
mri_train_input_files = glob(f'{base_dir}/mri_train_input/*.npy')
mri_train_label_files = glob(f'{base_dir}/mri_train_label/*.npy')

mri_val_input_files = glob(f'{base_dir}/mri_val_input/*.npy')
mri_val_label_files = glob(f'{base_dir}/mri_val_label/*.npy')

mri_test_input_files = glob(f'{base_dir}/mri_test_input/*.npy')
mri_test_label_files = glob(f'{base_dir}/mri_test_label/*.npy')

num_train_input = len(mri_train_input_files)
num_val_input = len(mri_val_input_files)
num_test_input = len(mri_test_input_files)

num_train_label = len(mri_train_label_files)
num_val_label = len(mri_val_label_files)
num_test_label = len(mri_test_label_files)

print(num_train_input, num_train_label)
print('\n')
print(num_val_input, num_val_label)
print('\n')
print(num_test_input, num_test_label)



# tensorflow에서 제공하는 dataset 모듈을 이용해 경로 리스트를 하나씩 슬라이스해주는 tensor dataset을 구성.
mri_train_data_files = tf.data.Dataset.from_tensor_slices((mri_train_input_files, mri_train_label_files))
mri_val_data_files = tf.data.Dataset.from_tensor_slices((mri_val_input_files, mri_val_label_files))
mri_test_data_files = tf.data.Dataset.from_tensor_slices((mri_test_input_files, mri_test_label_files))

# 알맞은 수가 리스트에 잘 담겼는지 확인.
print(len(mri_train_input_files), len(mri_train_label_files))


# 만들어진 경로 tensor dataset으로부터 실제 데이터셋을 불러오는 map function을 만들기
def map_func(inp_path, targ_path):
    inp = []
    targ = []
    for i, t in zip(inp_path, targ_path):
        temp_i = np.load(i)
        temp_i = temp_i.astype(np.float32).tolist()
        inp.append(temp_i)
        temp_t = np.load(t)
        temp_t = temp_t.astype(np.float32).tolist()
        targ.append(temp_t)
    inp = tf.constant(np.expand_dims(inp, 3), dtype=tf.float32)
    targ = tf.constant(np.expand_dims(targ, 3), dtype=tf.float32)
    return inp, targ


# 혹시나 있을지 모를 모양의 변화를 교정해주는 map function
def _fixup_shape(images, labels):
    images.set_shape([None, 384, 384, 1])
    labels.set_shape([None, 384, 384, 1])
    return images, labels


# 가중치를 업데이트하는 주기인 Batch_size를 지정. 현재의 경우 한 사진마다 가중치를 backpropagation을 통해 업데이트 함.
BATCH = 1

# 경로 텐서 데이터셋을 데이터를 담은 텐서 데이터셋으로 변환해주는 셀.
# map 함수와 파이썬 람다식을 이용했음.
# prefetch와 CPU 처리 방식은 모두 오토튠을 활용해서 처리.
train_data = mri_train_data_files.batch(BATCH)
train_data = train_data.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.float32]),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_data = train_data.map(_fixup_shape)

val_data = mri_val_data_files.batch(BATCH)
val_data = val_data.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.float32]),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_data = val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_data = val_data.map(_fixup_shape)

test_data = mri_test_data_files.batch(BATCH)
test_data = test_data.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_data = test_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_data = test_data.map(_fixup_shape)

# 신경망에 feed할 데이터가 모양이 잘 잡혔나 확인.
print(next(iter(train_data))[0].shape, next(iter(train_data))[1].shape, next(iter(val_data))[0].shape,
      next(iter(val_data))[1].shape)

# 신경망(U-Net)을 설명.
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras import Model


def UNet(input_size=(None, None, 1)):
    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inp, outputs=[conv10])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])

    return model


# 모델을 지정하고 서머리를 출력.
model = UNet(input_size=(384, 384, 1))
model.summary()

# tf.keras.utils.plot_model(model, show_shapes=True, dpi=42)

# 인공지능 학습
history = model.fit(train_data, epochs=10, verbose=1, validation_data=val_data)

# model 저장
model.save('./Data/UNet_model.h5')
# 결과 사진 출력
pred = model.predict(test_data)

plt.subplot(1, 3, 1).axis("off")
img1 = np.load(f'{base_dir}/mri_test_input/0.npy')
plt.imshow(img1, cmap='gray')
plt.subplot(1, 3, 2).axis("off")
plt.imshow(pred[0, :, :, 0], cmap='gray')
plt.subplot(1, 3, 3).axis("off")
img2 = np.load(f'{base_dir}/mri_test_label/0.npy')
plt.imshow(img2, cmap='gray')

plt.figure(figsize=(30, 20))
for i in range(20):
    plt.subplot(5, 4, i+1).axis("off")
    plt.imshow(pred[i, :, :, 0], cmap='gray')
    plt.title(f'pred_{i}')

# 결과 그래프 출력
import pandas as pd

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['mae', 'loss']].plot()

plt.show()


