# coding=utf-8

# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import numpy as np
import facenet
import os
import cv2
import face_lib
import detect_face
import urllib, urllib2
import argparse
import base64
from pyinotify import WatchManager, Notifier, ProcessEvent, IN_DELETE, IN_CREATE, IN_MODIFY, IN_CLOSE_WRITE

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor


def recon_view(pnet, rnet, onet, model, image_path, ip, margin=44, image_size=160):
    print('begin recon_view!!')
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            # Load the model
            facenet.load_model(model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # embeddings：输入端的特征向量
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # 加载库中的人脸并映射成embdings
            print('加载人脸库。。。。。。')
            embding, images_label_list = face_lib.get_face_lib()
            print('读取完毕.....')

            path_exp = os.path.expanduser(image_path)
            classes = os.listdir(path_exp)
            if not len(classes) == 0:
                classes.sort()
                nrof_classes = len(classes)
                for i in range(nrof_classes):
                    images_path = os.path.join(path_exp, classes[i])
                    # get_face(sess, images_path, margin, image_size, images_placeholder, embeddings,
                    #          phase_train_placeholder, embding, images_label_list, pnet, rnet, onet)
                    if os.path.isfile(images_path):
                        get_video(sess, images_path, margin, image_size, images_placeholder, embeddings,
                                  phase_train_placeholder, embding, images_label_list, pnet, rnet, onet, ip)
                        os.remove(images_path)

            wm = WatchManager()
            mask = IN_DELETE | IN_CREATE | IN_MODIFY | IN_CLOSE_WRITE
            notifier = Notifier(wm, EventHandler(sess, image_path, margin, image_size, images_placeholder, embeddings,
                                                 phase_train_placeholder, embding, images_label_list, pnet, rnet, onet,
                                                 ip))
            wm.add_watch(image_path, mask, auto_add=True, rec=True)
            print('Please input video')
            while True:
                try:
                    notifier.process_events()
                    if notifier.check_events():
                        notifier.read_events()
                except KeyboardInterrupt:
                    notifier.stop()
                    break

                    # print('寻找%s下图片文件' % image_path)


def get_video(sess, video_path, margin, image_size, images_placeholder, embeddings, phase_train_placeholder, embding,
              images_label_list, pnet, rnet, onet, ip):
    print('reading {} file'.format(video_path))
    dict = {}
    get_dict = {}
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        while ret:
            video_path, label, save_image_dir, ip = get_face_video(sess, frame, margin, image_size, images_placeholder,
                                                                   embeddings, phase_train_placeholder,
                                                                   embding,
                                                                   images_label_list, pnet, rnet, onet, dict,
                                                                   video_path, ip)
            ret, frame = cap.read()
        print('read {} over'.format(video_path))
        cap.release()
        cv2.destroyAllWindows()


def get_face_video(sess, frame, margin, image_size, images_placeholder, embeddings, phase_train_placeholder, embding,
                   images_label_list, pnet, rnet, onet, dict, video_path, ip):
    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]  # 人脸数目

    image_output_name = os.path.join(os.path.split(video_path)[0], 'tmp')
    if not os.path.exists(image_output_name):
        os.mkdir(image_output_name)
    if nrof_faces > 0:
        # 人脸对x和y的坐标范围集合
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(frame.shape)[0:2]
        face_num = det.shape[0]
        output_list = []

        for i in range(face_num):
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[i][0] - margin / 2, 0)
            bb[1] = np.maximum(det[i][1] - margin / 2, 0)
            bb[2] = np.minimum(det[i][2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[i][3] + margin / 2, img_size[0])

            cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            image_output_name_dir = os.path.join(image_output_name, 'k_' + str(i) + '.png')
            cv2.imwrite(image_output_name_dir, scaled)
            output_list.append(image_output_name_dir)
        if not len(output_list) == 0:
            images = facenet.load_data(output_list, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            embeddings_face = sess.run(embeddings, feed_dict=feed_dict)
            label = ''
            min_dist = 2
            for i in range(embding.shape[0]):
                dist = np.sum(np.square(np.subtract(embeddings_face[0, :], embding[i, :])))
                if min_dist > dist:
                    min_dist = dist
                    label = images_label_list[i]
            if min_dist < 0.5:
                if dict.has_key(label):
                    dict[label] = dict[label] + 1
                else:
                    dict[label] = 0
                    save_image_dir = os.path.join(image_output_name, str(label) + '.png')
                    cv2.imwrite(save_image_dir, frame)
                    f = open(save_image_dir, 'rb')
                    ls_f = base64.b64encode(f.read())
                    send_message(os.path.basename(video_path), label, save_image_dir, ip)
                    print(label)

            for item in output_list:
                os.remove(item)
    return os.path.basename(video_path), label, ls_f, ip


def load_mtcnn_model():
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def send_message(video, faceName, image_output_name, ip):
    f = open(image_output_name, 'rb')
    ls_f = base64.b64encode(f.read())
    data = urllib.urlencode({"videoName": video, "faceName": faceName, 'cappFace': ls_f})
    path_data = "http://" + ip + ":8080/dali/video/face"
    request = urllib2.Request(path_data, data)
    response = urllib2.urlopen(request)
    file = response.read()
    if response.code != 200:
        return 'error code' + response.code
    else:
        return file
    f.close()


class EventHandler(ProcessEvent):
    def __init__(self, sess, images_path, margin, image_size, images_placeholder, embeddings,
                 phase_train_placeholder, embding, images_label_list, pnet, rnet, onet, ip):
        self.sess = sess
        self.images_path = images_path
        self.margin = margin
        self.image_size = image_size
        self.images_placeholder = images_placeholder
        self.embeddings = embeddings
        self.phase_train_placeholder = phase_train_placeholder
        self.embding = embding
        self.images_label_list = images_label_list
        self.pnet = pnet
        self.rnet = rnet
        self.onet = onet
        self.ip = ip

    def process_IN_CREATE(self, event):
        print("Create file: %s " % os.path.join(event.path, event.name))

    def process_IN_CLOSE(self, event):
        images_path = os.path.join(event.path, event.name)
        get_video(self.sess, images_path, self.margin, self.image_size, self.images_placeholder, self.embeddings,
                  self.phase_train_placeholder, self.embding, self.images_label_list, self.pnet, self.rnet, self.onet,
                  self.ip)
        os.remove(images_path)
        print("close file: %s " % os.path.join(event.path, event.name))


def FSMonitor(path='.'):
    wm = WatchManager()
    mask = IN_DELETE | IN_CREATE | IN_MODIFY | IN_CLOSE_WRITE
    notifier = Notifier(wm, EventHandler())
    wm.add_watch(path, mask, auto_add=True, rec=True)
    print('now starting monitor %s' % (path))
    while True:
        try:
            notifier.process_events()
            if notifier.check_events():
                notifier.read_events()
        except KeyboardInterrupt:
            notifier.stop()
            break


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str,
                        help='Directory image vedio path', default='/home/video')
    parser.add_argument('--lib_path', type=str,
                        help='Directory lib image path', default='/home/images')
    parser.add_argument('--model', type=str,
                        help='Directory face recon model path', default='/home/ubuntu/project/dali/models')
    parser.add_argument('--ip', type=str,
                        help='Request ip to send', default='192.168.1.18')
    return parser.parse_args(argv)


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])
    model = args.model
    image_path = args.image_path
    lib_path = args.lib_path
    ip = args.ip
    # model = ''
    # alig_path = '/home/shi/data/db/tmp'
    align_path = os.path.join(lib_path, 'tmp')
    if not os.path.exists(align_path):
        os.mkdir(align_path)
    pnet, rnet, onet = load_mtcnn_model()
    face_lib.set_fold_align(lib_path, pnet, rnet, onet, 160, 64)
    face_lib.set_face_lib(align_path, model, image_size=160)
    recon_view(pnet, rnet, onet, model, image_path, ip)
