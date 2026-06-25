import torch
import numpy as np
import cv2
from tqdm import tqdm
import time

from logger import Logger
from option import get_option
from data import import_loader
from loss import import_loss
from model import import_model
from skimage.metrics import structural_similarity as ssim


def train(opt, logger):
    logger.info('task: {}, model task: {}'.format(opt.task, opt.model_task))

    train_loader, valid_loader = import_loader(opt)
    lr = float(opt.config['train']['lr'])
    lr_warmup = float(opt.config['train']['lr_warmup'])

    loss_warmup = import_loss('warmup').to(opt.device)
    loss_training = import_loss(opt.model_task).to(opt.device)
    net = import_model(opt)
    # logger.info(net)

    net.train()
    # Phase Warming-up
    if opt.config['train']['warmup']:
        logger.info('start warming-up')

        optim_warm = torch.optim.Adam(net.parameters(), lr_warmup, weight_decay=0)
        epochs = opt.config['train']['warmup_epoch']
        for epo in range(epochs):
            scaler_warm = torch.amp.GradScaler('cuda')
            loss_li = []
            for img_inp, img_gt, _ in tqdm(train_loader, ncols=80):
                img_inp, img_gt = img_inp.to(opt.device, non_blocking=True), img_gt.to(opt.device, non_blocking=True)
                optim_warm.zero_grad()
                with torch.amp.autocast('cuda'):
                    warmup_out1, warmup_out2 = net.forward_warm(img_inp)
                    loss = loss_warmup(img_inp, img_gt, warmup_out1, warmup_out2)
                scaler_warm.scale(loss).backward()
                scaler_warm.step(optim_warm)
                scaler_warm.update()
                loss_li.append(loss.item())

            logger.info('epoch: {}, train_loss: {}'.format(epo+1, sum(loss_li)/len(loss_li)))
            torch.save(net.state_dict(), r'{}\model_pre.pkl'.format(opt.save_model_dir))
        logger.info('warming-up phase done')

    # Phase Training
    best_psnr = 0
    epochs = int(opt.config['train']['epoch'])
    optim = torch.optim.Adam(net.parameters(), lr, weight_decay=0)
    lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50, 2, 1e-7)

    logger.info('start training')
    scaler = torch.amp.GradScaler('cuda')
    for epo in range(epochs):
        loss_li = []
        test_psnr = []
        net.train()
        for img_inp, img_gt, _ in tqdm(train_loader, ncols=80):
            img_inp, img_gt = img_inp.to(opt.device, non_blocking=True), img_gt.to(opt.device, non_blocking=True)
            optim.zero_grad()
            with torch.amp.autocast('cuda'):
                out = net(img_inp)
                loss = loss_training(out, img_gt)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            loss_li.append(loss.item())
        lr_sch.step()

        # Validation
        net.eval()
        for img_inp, img_gt, _ in tqdm(valid_loader, ncols=80):
            img_inp, img_gt = img_inp.to(opt.device, non_blocking=True), img_gt.to(opt.device, non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    out = net(img_inp)
                # Keep MSE calculation in standard precision
                out = out.float()
                img_gt = img_gt.float()
                mse = ((out - img_gt)**2).mean((2, 3))
                psnr = (1 / mse).log10().mean() * 10
            test_psnr.append(psnr.item())
        mean_psnr = sum(test_psnr)/len(test_psnr)

        if (epo+1) % int(opt.config['train']['save_every']) == 0:
            torch.save(net.state_dict(), r'{}\model_{}.pkl'.format(opt.save_model_dir, epo+1))

        logger.info('epoch: {}, training loss: {}, validation psnr: {}'.format(
            epo+1, sum(loss_li) / len(loss_li), sum(test_psnr) / len(test_psnr)
        ))

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save(net.state_dict(), r'{}\model_best.pkl'.format(opt.save_model_dir))
            if opt.config['train']['save_slim']:
                net_slim = net.slim().to(opt.device)
                torch.save(net_slim.state_dict(), r'{}\model_best_slim.pkl'.format(opt.save_model_dir))
                logger.info('best model saved and re-parameterized in epoch {}'.format(epo+1))
            else:
                logger.info('best model saved in epoch in epoch {}'.format(epo+1))

    logger.info('training done')


def test(opt, logger):
    test_loader = import_loader(opt)
    net = import_model(opt)
    net.eval()

    psnr_list = []
    time_list = []

    logger.info('start testing')

    # 🔹 Start total test timer
    total_test_start = time.perf_counter()

    # Warmup (ignore timing)
    for img_inp, img_gt, _ in test_loader:
        img_inp, img_gt = img_inp.to(opt.device, non_blocking=True), img_gt.to(opt.device, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                _ = net(img_inp)
        break

    for (img_inp, img_gt, img_name) in test_loader:
        img_inp, img_gt = img_inp.to(opt.device, non_blocking=True), img_gt.to(opt.device, non_blocking=True)

        start_time = time.perf_counter()

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = net(img_inp)

        # 🔹 Synchronize for accurate GPU timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        inference_time = end_time - start_time
        time_list.append(inference_time)

        out = out.float()
        img_gt = img_gt.float()
        mse = ((out - img_gt)**2).mean((2, 3))
        psnr = (1 / mse).log10().mean() * 10

        if opt.config['test']['save']:
            out_img = (out.clip(0, 1)[0] * 255)\
                .permute([1, 2, 0])\
                .cpu()\
                .numpy()\
                .astype(np.uint8)[..., ::-1]

            cv2.imwrite(r'{}\{}.png'.format(opt.save_image_dir, img_name[0]), out_img)

        psnr_list.append(psnr.item())

        logger.info(
            'image: {}, psnr: {:.4f}, inference time: {:.6f} sec'.format(
                img_name[0], psnr.item(), inference_time
            )
        )

    # 🔹 End total test timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_test_end = time.perf_counter()
    total_test_time = total_test_end - total_test_start

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_time = sum(time_list) / len(time_list)

    logger.info('testing done')
    logger.info('overall psnr: {:.4f}'.format(avg_psnr))
    logger.info('average inference time per image: {:.6f} sec'.format(avg_time))
    logger.info('FPS: {:.2f}'.format(1.0 / avg_time))
    logger.info('total testing time: {:.4f} sec ({:.2f} minutes)'.format(
        total_test_time, total_test_time / 60
    ))

def demo(opt, logger):
    demo_loader = import_loader(opt)
    net = import_model(opt)
    net.eval()
    logger.info('start demonstration')
    for img_inp, img_name in demo_loader:
        img_inp = img_inp.to(opt.device, non_blocking=True)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = net(img_inp)
        out = out.float()
        out_img = (out.clip(0, 1)[0] * 255).permute([1, 2, 0]).cpu().numpy().astype(np.uint8)[..., ::-1]
        cv2.imwrite(r'{}\{}.png'.format(opt.save_image_dir, img_name[0]), out_img)
        logger.info('image name: {} output generated'.format(img_name[0]))
    logger.info('demonstration done')


if __name__ == "__main__":
    opt = get_option()
    logger = Logger(opt)

    if opt.task == 'train':
        train(opt, logger)
    elif opt.task == 'test':
        test(opt, logger)
    elif opt.task == 'demo':
        demo(opt, logger)
    else:
        raise ValueError('unknown task, please choose from [train, test, demo].')
