import numpy as np
import tensorflow as tf
import math


# 构造可训练参数
def make_var(name, shape, trainable=True):
    return tf.get_variable(name, shape, trainable=trainable)


# 定义卷积层
def conv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="conv2d", biased=False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
        return output


# 定义空洞卷积层
def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding="SAME", name="atrous_conv2d", biased=False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
        return output


# 定义反卷积层
def deconv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="deconv2d"):
    input_dim = input_.get_shape()[-1]
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim],
                                        [1, 2, 2, 1], padding="SAME")
        return output


# 定义batchnorm(批次归一化)层
def batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_ - mean) * inv
        output = scale * normalized + offset
        return output


# 定义lrelu激活层
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


# 定义生成器，采用UNet架构，主要由8个卷积层和8个反卷积层组成
def generator(image, gf_dim=64, reuse=False, name="generator"):
    input_dim = int(image.get_shape()[-1])  # 获取输入通道
    dropout_rate = 0.5  # 定义dropout的比例
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        # 第一个卷积层，输出尺度[1, 128, 128, 64]
        e1 = batch_norm(conv2d(input_=image, output_dim=gf_dim, kernel_size=4, stride=2, name='g_e1_conv'),
                        name='g_bn_e1')
        # 第二个卷积层，输出尺度[1, 64, 64, 128]
        e2 = batch_norm(conv2d(input_=lrelu(e1), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_e2_conv'),
                        name='g_bn_e2')
        # 第三个卷积层，输出尺度[1, 32, 32, 256]
        e3 = batch_norm(conv2d(input_=lrelu(e2), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_e3_conv'),
                        name='g_bn_e3')
        # 第四个卷积层，输出尺度[1, 16, 16, 512]
        e4 = batch_norm(conv2d(input_=lrelu(e3), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_e4_conv'),
                        name='g_bn_e4')
        # 第五个卷积层，输出尺度[1, 8, 8, 512]
        e5 = batch_norm(conv2d(input_=lrelu(e4), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_e5_conv'),
                        name='g_bn_e5')

        # 第一个反卷积层，输出尺度[1, 16, 16, 512]
        d4 = deconv2d(input_=tf.nn.relu(e5), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_d4')
        # 第一个skip 连接
        d4 = tf.concat([batch_norm(d4, name='g_bn_d4'), e4], 3)
        # 第二个反卷积层，输出尺度[1, 32, 32, 256]
        d5 = deconv2d(input_=tf.nn.relu(d4), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_d5')
        # 第二个 skip 连接
        d5 = tf.concat([batch_norm(d5, name='g_bn_d5'), e3], 3)
        # 第三个反卷积层，输出尺度[1, 64, 64, 128]
        d6 = deconv2d(input_=tf.nn.relu(d5), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_d6')
        # 第三个skip 连接
        d6 = tf.concat([batch_norm(d6, name='g_bn_d6'), e2], 3)
        # 第四个反卷积层，输出尺度[1, 128, 128, 64]
        d7 = deconv2d(input_=tf.nn.relu(d6), output_dim=gf_dim, kernel_size=4, stride=2, name='g_d7')
        #第四个skip连接
        d7 = tf.concat([batch_norm(d7, name='g_bn_d7'), e1], 3)
        # 第五个反卷积层，输出尺度[1, 256, 256, 3]
        d8 = deconv2d(input_=tf.nn.relu(d7), output_dim=input_dim, kernel_size=4, stride=2, name='g_d8')
        return tf.nn.tanh(d8)


# 定义判别器
def discriminator(image, targets, df_dim=64, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        dis_input = tf.concat([image, targets], 3)
        # 第1个卷积模块，输出尺度: 1*128*128*64
        h0 = lrelu(conv2d(input_=dis_input, output_dim=df_dim, kernel_size=4, stride=2, name='d_h0_conv'))
        # 第2个卷积模块，输出尺度: 1*64*64*128
        h1 = lrelu(batch_norm(conv2d(input_=h0, output_dim=df_dim * 2, kernel_size=4, stride=2, name='d_h1_conv'),
                              name='d_bn1'))
        # 第3个卷积模块，输出尺度: 1*32*32*256
        h2 = lrelu(batch_norm(conv2d(input_=h1, output_dim=df_dim * 4, kernel_size=4, stride=2, name='d_h2_conv'),
                              name='d_bn2'))
        # 第4个卷积模块，输出尺度: 1*32*32*512
        h3 = lrelu(batch_norm(conv2d(input_=h2, output_dim=df_dim * 8, kernel_size=4, stride=1, name='d_h3_conv'),
                              name='d_bn3'))
        # 最后一个卷积模块，输出尺度: 1*32*32*1
        output = conv2d(input_=h3, output_dim=1, kernel_size=4, stride=1, name='d_h4_conv')
        dis_out = tf.sigmoid(output)  # 在输出之前经过sigmoid层，因为需要进行log运算
        return dis_out