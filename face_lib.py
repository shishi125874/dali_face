# coding=utf-8
import os
import tensorflow as tf
import facenet
import numpy as np
import cv2
import detect_face
import imghdr


# save the face embings,the face is aligned
def set_face_lib(paths_foder, model, image_size=160):
    batch = 300
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        with tf.Session(config=config) as sess:
            # Load the model
            facenet.load_model(model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # embeddings：输入端的特征向量
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # 加载库中的人脸并映射成embdings
            print('loading face db。。。。。。')
            path_exp = os.path.expanduser(paths_foder)
            classes = os.listdir(path_exp)
            images_path_list = []
            images_label_list = []
            embdings = []
            for i in range(len(classes)):
                if not classes[i][0] == '.':
                    images_full = os.path.join(path_exp, classes[i])
                    image_type = imghdr.what(images_full)
                    if not image_type == None or image_type == '.gif':
                        images_path_list.append(images_full)
                        images_label_list.append(os.path.splitext(classes[i])[0])
            for i in range(len(images_path_list / batch)):
                path_batch = images_path_list[i * batch:(i + 1) * batch]
                images_emb = facenet.load_data(path_batch, False, False, image_size)
                feed_dicts = {images_placeholder: images_emb, phase_train_placeholder: False}
                embding = sess.run(embeddings, feed_dict=feed_dicts)
                embdings.append(embding)
            np.savez('embbings', image=embdings, images_label_list=images_label_list)

            #            images_emb = facenet.load_data(images_path_list, False, False, image_size)
            #
            #            print('save over.....')
            #
            #            feed_dicts = {images_placeholder: images_emb, phase_train_placeholder: False}
            #            embding = sess.run(embeddings, feed_dict=feed_dicts)
            #            print(embding.shape[0])
            #            print(type(embding))
            #            np.savez('embbings', image=embding, images_label_list=images_label_list)
            # sess.close()


def get_face_lib():
    r = np.load('embbings.npz')
    print r['image'].shape[0]
    print len(r['images_label_list'])
    return r['image'], r['images_label_list']


def img_align(img_path, pnet, rnet, onet, image_size, margin=44):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    images_lists = []
    img = cv2.imread(os.path.expanduser(img_path))
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    nrof_faces = bounding_boxes.shape[0]
    print('detected image {} Detected_FaceNum: {}'.format(img_path, nrof_faces))

    k = 0
    for face_position in bounding_boxes:
        face_position = face_position.astype(int)
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        face_frame = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = cv2.resize(face_frame, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        output_name = os.path.join(os.path.split(img_path)[0], 'tmp')
        new_name = os.path.splitext(os.path.split(img_path)[1])[0] + '.png'
        output_name = os.path.join(output_name, new_name)
        cv2.imwrite(output_name, aligned)


def set_fold_align(input, pnet, rnet, onet, image_size, margin=44):
    print('align db face in image')
    path_exp = os.path.expanduser(input)
    classes = os.listdir(path_exp)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        images_path = os.path.join(path_exp, classes[i])
        if os.path.isfile(images_path):
            img_align(images_path, pnet, rnet, onet, image_size, margin)


def load_mtcnn_model():
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


if __name__ == '__main__':
    model = '/home/shi/project/face/facenet/20170512-110547'
    zyst_lib = '/home/shi/project/face/zyst_align_lib'
    # # set_face_lib(zyst_lib,model)
    # path = 'embbings.txt'
    # get_face_lib()
    # set_face_lib(zyst_lib, model,182)
    pnet, rnet, onet = load_mtcnn_model()
    set_fold_align('/home/shi/data/db', pnet, rnet, onet, 160, 64)
