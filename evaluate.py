import argparse
import sys
import math
import tensorflow as tf
import numpy as np
import glob
import cv2

from image_reader import *
from net import *

parser = argparse.ArgumentParser(description='')

parser.add_argument("--test_picture_path", default='./dataset/test_picture/', help="path of test datas.")  # 网络测试输入的图片路径
parser.add_argument("--test_label_path", default='./dataset/test_label/', help="path of test datas.")  # 网络测试输入的标签路径
parser.add_argument("--image_size", type=int, default=256, help="load image size")  # 网络输入的尺度
parser.add_argument("--test_picture_format", default='.png', help="format of test pictures.")  # 网络测试输入的图片的格式
parser.add_argument("--test_label_format", default='.jpg', help="format of test labels.")  # 网络测试时读取的标签的格式
parser.add_argument("--snapshots", default='./snapshots/', help="Path of Snapshots")  # 读取训练好的模型参数的路径
parser.add_argument("--out_dir", default='./test_output/', help="Output Folder")  # 保存网络测试输出图片的路径

args = parser.parse_args()  # 用来解析命令行参数


def cv_inv_proc(img):  # cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = (img + 1.) * 127.5
    return img_rgb.astype(np.float32)  # 返回bgr格式的图像，方便cv2写图像


def get_write_picture(picture, gen_label, label, height, width):  # get_write_picture函数得到网络测试的结果
    picture_image = cv_inv_proc(picture)  # 还原输入的图像
    gen_label_image = cv_inv_proc(gen_label[0])  # 还原生成的结果
    label_image = cv_inv_proc(label)  # 还原读取的标签
    inv_picture_image = cv2.resize(picture_image, (width, height))  # 将输入图像还原到原大小
    inv_gen_label_image = cv2.resize(gen_label_image, (width, height))  # 将生成的结果还原到原大小
    inv_label_image = cv2.resize(label_image, (width, height))  # 将标签还原到原大小
    output = np.concatenate((inv_picture_image, inv_gen_label_image, inv_label_image), axis=1)  # 拼接得到输出结果
    return output


def main():
    if not os.path.exists(args.out_dir):  # 如果保存测试结果的文件夹不存在则创建
        os.makedirs(args.out_dir)

    test_picture_list = glob.glob(os.path.join(args.test_picture_path, "*"))  # 得到测试输入图像路径名称列表
    test_picture = tf.placeholder(tf.float32, shape=[1, 256, 256, 3], name='test_picture')  # 测试输入的图像

    gen_label = generator(image=test_picture, gf_dim=64, reuse=False, name='generator')  # 得到生成器的生成结果

    restore_var = [v for v in tf.global_variables() if 'generator' in v.name]  # 需要载入的已训练的模型参数

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 设定显存不超量使用
    sess = tf.Session(config=config)  # 建立会话层

    saver = tf.train.Saver(var_list=restore_var, max_to_keep=1)  # 导入模型参数时使用
    checkpoint = tf.train.latest_checkpoint(args.snapshots)  # 读取模型参数
    saver.restore(sess, checkpoint)  # 导入模型参数

    for step in range(len(test_picture_list)):
        picture_name, _ = os.path.splitext(os.path.basename(test_picture_list[step]))  # 得到一张网络测试的输入图像名字
        # 读取一张测试图片，一张标签，以及相应的高和宽
        picture_resize, label_resize, picture_height, picture_width = ImageReader(file_name=picture_name,
                                                                                  picture_path=args.test_picture_path,
                                                                                  label_path=args.test_label_path,
                                                                                  picture_format=args.test_picture_format,
                                                                                  label_format=args.test_label_format,
                                                                                  size=args.image_size)
        batch_picture = np.expand_dims(np.array(picture_resize).astype(np.float32), axis=0)  # 填充维度
        feed_dict = {test_picture: batch_picture}  # 构造feed_dict
        gen_label_value = sess.run(gen_label, feed_dict=feed_dict)  # 得到生成结果
        write_image = get_write_picture(picture_resize, gen_label_value, label_resize, picture_height,
                                        picture_width)  # 得到一张需要存的图像
        write_image_name = args.out_dir + picture_name + ".png"  # 为上述的图像构造保存路径与文件名
        cv2.imwrite(write_image_name, write_image)  # 保存测试结果
        print('step {:d}'.format(step))


if __name__ == '__main__':
    main()