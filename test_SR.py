import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html

# print("test!")
# import pydevd
# pydevd.settrace('172.18.218.190', port=10000, stdoutToServer=True, stderrToServer=True)
# print("test!")

if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display

    # load the train part data
    opt.phase = 'train'
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    opt.phase = 'test'
    model = create_model(opt)
    model.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    avg_psnr = 0
    avg_bicubic_psnr = 0
    avg_ssim = 0
    avg_bicubic_ssim = 0
    for i, data in enumerate(dataset):
        # if i >= opt.num_test:
            # break
        model.set_input(data)
        model.test()    # compute the forward(), psnr, ssim
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        # if i < opt.num_test:
        # if i < 10:
        #     save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        if i % 100 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        bicubic_psnr, psnr = model.get_psnr()
        # print(psnr)
        avg_bicubic_psnr += bicubic_psnr
        avg_psnr += psnr

        bicubic_ssim, ssim = model.get_ssim()
        # print(ssim.item())
        avg_bicubic_ssim += bicubic_ssim
        avg_ssim += ssim
    # print("===> Avg. bibubic PSNR: {:.4f} dB".format(avg_bicubic_psnr / opt.num_test))
    # print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / opt.num_test))
    # print("===> Avg. bibubic SSIM: {:.4f} dB".format(avg_bicubic_ssim / opt.num_test))
    # print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim / opt.num_test))

    print("===> Avg. bibubic PSNR: {:.4f} dB".format(avg_bicubic_psnr / len(dataset)))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(dataset)))
    print("===> Avg. bibubic SSIM: {:.4f} dB".format(avg_bicubic_ssim / len(dataset)))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim / len(dataset)))
    print(len(dataset))
    # save the website
    webpage.save()
