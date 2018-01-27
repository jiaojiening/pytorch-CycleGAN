import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import scipy.io
from util.visualizer import Visualizer

# print("test!")
# import pydevd
# pydevd.settrace('172.18.218.190', port=10000, stdoutToServer=True, stderrToServer=True)
# print("test!")

def get_id(img_path):
    camera_id = []
    labels = []
    # for path, v in img_path:
    for path in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 50    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    # opt.display_id = 1   # visdom display
    opt.display_ncols = 4
    opt.display_server = "http://localhost"
    opt.display_port = 8097
    opt.display_env = 'main'

    # create query(low-resolution) set
    opt.dataset_type = 'B'
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # create and load the model
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    total_cams = []
    total_labels = []
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test_reid()    # compute the forward()

        img_path = model.get_image_paths()
        camera_id, labels = get_id(img_path)
        total_cams.extend(camera_id)
        total_labels.extend(labels)

        visualizer.reset()
        if i % 1 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path[0]))
            save_result = True
            visualizer.display_current_results(model.get_current_visuals(), i, save_result)

    query_feature = model.get_features()
    query_cam = total_cams
    query_label = total_labels

    # create gallery(high-resolution) set
    opt.dataset_type = 'A'
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    total_cams = []
    total_labels = []
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test_reid()  # compute the forward()

        img_path = model.get_image_paths()
        camera_id, labels = get_id(img_path)
        total_cams.extend(camera_id)
        total_labels.extend(labels)

        visualizer.reset()
        if i % 10 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path[0]))
            save_result = True
            visualizer.display_current_results(model.get_current_visuals(), i, save_result)

    gallery_feature = model.get_features()
    gallery_cam = total_cams
    gallery_label = total_labels

    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
    # if not os.path.exists(opt.reid_results_dir):
    #     os.makedirs(opt.reid_results_dir)
    # scipy.io.savemat(os.path.join(opt.reid_results_dir, 'Duke_base_result.mat'), result)

    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    if not os.path.exists(web_dir):
        os.makedirs(web_dir)
    save_path = os.path.join(web_dir, 'reid_result.mat')
    scipy.io.savemat(save_path, result)



