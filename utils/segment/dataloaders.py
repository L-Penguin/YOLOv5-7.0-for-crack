# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, distributed

from ..augmentations import augment_hsv, copy_paste, letterbox
from ..dataloaders import InfiniteDataLoader, LoadImagesAndLabels, seed_worker
from ..general import LOGGER, xyn2xy, xywhn2xyxy, xyxy2xywhn
from ..torch_utils import torch_distributed_zero_first
from .augmentations import mixup, random_perspective

import torch.nn.functional as F

RANK = int(os.getenv('RANK', -1))


def create_dataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False,
                      mask_downsample_ratio=1,
                      overlap_mask=False,
                      pre_process=False,
                      saveMosaicImg=False,
                      rotate=False,
                      mosaic9=False):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsAndMasks(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            downsample_ratio=mask_downsample_ratio,
            overlap=overlap_mask,
            pre_process=pre_process,
            saveMosaicImg=saveMosaicImg,
            rotate=rotate,
            mosaic9=mosaic9)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabelsAndMasks.collate_fn4 if quad else LoadImagesAndLabelsAndMasks.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


class LoadImagesAndLabelsAndMasks(LoadImagesAndLabels):  # for training/testing

    def __init__(
            self,
            path,
            img_size=640,
            batch_size=16,
            augment=False,
            hyp=None,
            rect=False,
            image_weights=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            pad=0,
            min_items=0,
            prefix="",
            downsample_ratio=1,
            overlap=False,
            pre_process=False,
            saveMosaicImg=False,
            rotate=False,
            mosaic9=False,
    ):
        super().__init__(path, img_size, batch_size, augment, hyp, rect, image_weights, cache_images, single_cls,
                         stride, pad, min_items, prefix)
        self.downsample_ratio = downsample_ratio
        self.overlap = overlap
        self.pre_process = pre_process
        self.saveMosaicImg = saveMosaicImg
        self.rotate = rotate
        self.mosaic9 = mosaic9

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        masks = []
        if mosaic:
            # Load mosaic
            if self.mosaic9:
                img, labels, segments = self.load_mosaic9(index)
            else:
                img, labels, segments = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels, segments = mixup(img, labels, segments, *self.load_mosaic9(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            # [array, array, ....], array.shape=(num_points, 2), xyxyxyxy
            segments = self.segments[index].copy()
            if len(segments):
                for i_s in range(len(segments)):
                    segments[i_s] = xyn2xy(
                        segments[i_s],
                        ratio[0] * w,
                        ratio[1] * h,
                        padw=pad[0],
                        padh=pad[1],
                    )
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels, segments = random_perspective(img,
                                                           labels,
                                                           segments=segments,
                                                           degrees=hyp["degrees"],
                                                           translate=hyp["translate"],
                                                           scale=hyp["scale"],
                                                           shear=hyp["shear"],
                                                           perspective=hyp["perspective"])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            if self.overlap:
                masks, sorted_idx = polygons2masks_overlap(img.shape[:2],
                                                           segments,
                                                           downsample_ratio=self.downsample_ratio)
                masks = masks[None]  # (640, 640) -> (1, 640, 640)
                labels = labels[sorted_idx]
            else:
                masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=self.downsample_ratio)

        masks = (torch.from_numpy(masks) if len(masks) else torch.zeros(1 if self.overlap else nl, img.shape[0] //
                                                                        self.downsample_ratio, img.shape[1] //
                                                                        self.downsample_ratio))
        # TODO: albumentations support
        if self.augment:
            # Albumentations
            # there are some augmentation that won't change boxes and masks,
            # so just be it for now.
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
                    masks = torch.flip(masks, dims=[1])

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
                    masks = torch.flip(masks, dims=[2])

            # Cutouts  # labels = cutout(img, labels, p=0.5)
        # 需要搭配no-overlap使用
        if self.pre_process:
            new_masks, new_labels = self.solve_masks_labels(masks, labels, img.shape)
            nl = len(new_labels)
            if new_masks.shape[0]:
                new_masks = torch.squeeze(self.tensor_dilate(torch.unsqueeze(new_masks, 0), 3), 0)
        else:
            new_masks, new_labels = masks, labels

        if self.saveMosaicImg:
            # 保存增强之后的图像
            self.save_mosaic_imgs(img, masks, labels, index, new_masks, new_labels)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(new_labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes, new_masks

    # 解决标签问题
    def solve_masks_labels(self, masks, labels, shape):
        # 将遮盖层和标签进行处理，相接的遮盖层连接在一起并将标签合并
        masks_float32 = masks.type(torch.float32)
        try:
            masks_numpy = F.interpolate(masks_float32[None],
                                        shape[:2],
                                        mode='bilinear',
                                        align_corners=False)[0].cpu().numpy()
        except RuntimeError:
            masks_numpy = masks_float32.reshape((-1, shape[0], shape[1]))

        kernel = np.ones((3, 3))
        for i, m in enumerate(masks_numpy):
            masks_numpy[i] = cv2.filter2D(m, -1, kernel)

        connectArr = []
        masks_numpy[masks_numpy > 0] = 1
        for i in range(len(masks_numpy) - 1):
            for j in range(i + 1, len(masks_numpy)):
                if (masks_numpy[i] + masks_numpy[j]).max() == 2:
                    if len(connectArr) != 0:
                        for index, arr in enumerate(connectArr):
                            if len(set(arr + [i, j])) < len(arr) + 2:
                                connectArr[index] = list(set(arr + [i, j]))
                            else:
                                connectArr.append([i, j])
                    else:
                        connectArr.append([i, j])
        new_masks, new_labels = self.concat_masks_labels(masks.cpu().numpy(), labels.copy(), connectArr, shape)
        new_masks = torch.tensor(new_masks, dtype=torch.uint8)

        return new_masks, new_labels

    # 将掩码和标签合并
    def concat_masks_labels(self, masks, labels, target, shape):
        saveIndex = np.zeros(len(masks)) == 0
        for t in target:
            # 合并mask并赋值
            masks[t[0]] = masks[t].sum(axis=0)
            masks[t[0], masks[t[0]] > 0] = 1
            # 合并labels并赋值
            labels_ = xywhn2xyxy(labels[t, 1:], shape[0], shape[1], 0, 0)
            x_labels = labels_[:, [0, 2]]
            x_min, x_max = x_labels.min(), x_labels.max()
            y_labels = labels_[:, [1, 3]]
            y_min, y_max = y_labels.min(), y_labels.max()
            labels[t[0], 1:5] = xyxy2xywhn(np.array([x_min, y_min, x_max, y_max]).reshape(1, -1),
                                           shape[0],
                                           shape[1],
                                           clip=True,
                                           eps=1e-3)

            saveIndex[t[1:]] = False
        # 去除合并所剩余的mask
        return masks[saveIndex], labels[saveIndex]

    def tensor_dilate(self, bin_img, ksize=3):
        # 首先为原图加入 padding，防止图像尺寸缩小
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        # 将原图 unfold 成 patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # 取每个 patch 中最小的值，i.e., 0
        dilate, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
        return dilate

    # 保存mosaic增强后的图像
    def save_mosaic_imgs(self, img, masks, labels, index, new_masks=None, new_labels=None):
        # 存储mosaic处理的图片
        workPath = os.getcwd()
        dirName = f"save-mosaic/mosaicImgs"
        dirPath = os.path.join(workPath, dirName)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        fileName = os.path.basename(self.im_files[index])
        oriName = os.path.splitext(fileName)[0] + '_1' + os.path.splitext(fileName)[1]
        preName = os.path.splitext(fileName)[0] + '_2' + os.path.splitext(fileName)[1]
        path_1 = os.path.join(dirPath, f'{fileName}')
        path_2 = os.path.join(dirPath, f'{oriName}')
        path_3 = os.path.join(dirPath, f'{preName}')

        # 原始标签图像
        ic_1 = img.copy()
        # 预处理标签图像
        ic_2 = img.copy()

        try:
            labels_xyxy = xywhn2xyxy(labels[:, 1:], img.shape[0], img.shape[1], 0, 0)
        except IndexError:
            # print(f'\n{"="*25}\n{fileName} is empty!\n{"="*25}\n')
            cv2.imwrite(path_1, img)
            cv2.imwrite(path_2, ic_1)
            return

        # 绘制目标检测框
        for l in labels_xyxy:
            point_1 = (int(l[0]), int(l[1]))
            point_2 = (int(l[2]), int(l[3]))
            cv2.rectangle(ic_1, point_1, point_2, (255, 0, 0), 2)
        # 绘制遮盖层
        masks_ = masks.type(torch.float32)
        if not self.overlap:
            masks_ = masks.sum(axis=0, keepdim=True)
            masks_ = masks_.type(torch.float32)
        masks_ = F.interpolate(masks_[None], ic_1.shape[:2], mode='bilinear', align_corners=False)[0]
        masks_[masks_ > 1] = 1
        mask_ = masks_.permute((1, 2, 0)) * 255
        c1 = torch.zeros(mask_.shape, dtype=mask_.dtype)
        c2 = torch.zeros(mask_.shape, dtype=mask_.dtype)
        mask = torch.cat((c1, c2, mask_), dim=2).cpu().numpy()
        mask = mask.astype(np.uint8)
        output = cv2.addWeighted(ic_1, 1, mask, 0.5, 0)
        # 展示掩码信息的拼接图像
        cv2.imwrite(path_2, output)
        # 原始拼接图像
        cv2.imwrite(path_1, img)

        # 存储与处理后图像
        if self.pre_process:
            try:
                new_labels_xyxy = xywhn2xyxy(new_labels[:, 1:], img.shape[0], img.shape[1], 0, 0)
            except IndexError:
                cv2.imwrite(path_3, ic_1)

            for l in new_labels_xyxy:
                new_point1 = (int(l[0]), int(l[1]))
                new_point2 = (int(l[2]), int(l[3]))
                cv2.rectangle(ic_2, new_point1, new_point2, (0, 0, 255), 2)

            # 绘制遮盖层
            new_masks_ = new_masks.type(torch.float32)
            if not self.overlap:
                new_masks_ = masks.sum(axis=0, keepdim=True)
                new_masks_ = new_masks_.type(torch.float32)

            new_masks_ = F.interpolate(new_masks_[None], ic_2.shape[:2], mode='bilinear', align_corners=False)[0]
            new_masks_[new_masks_ > 1] = 1
            new_masks_ = new_masks_.permute((1, 2, 0)) * 255
            c1 = torch.zeros(new_masks_.shape, dtype=mask_.dtype)
            c2 = torch.zeros(new_masks_.shape, dtype=mask_.dtype)
            new_mask = torch.cat((c1, c2, new_masks_), dim=2).cpu().numpy()
            new_mask = new_mask.astype(np.uint8)
            new_output = cv2.addWeighted(ic_2, 1, new_mask, 0.5, 0)

            cv2.imwrite(path_3, new_output)

    def load_mosaic(self, index):
        fileName = os.path.basename(self.im_files[index])
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        # print(f's: {s}')
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

        # 3 additional image indices
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices

        # 创建文件记录mosaic处理文件
        logTxt = f'./save-mosaic/mosaicLog.txt'
        content = f'{fileName.ljust(15, " ")}: '

        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            imgName = os.path.basename(self.im_files[index])
            # mosaicLog内容
            content += f'{imgName.center(15, " ")}'

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            padw = x1a - x1b
            padh = y1a - y1b

            labels, segments = self.labels[index].copy(), self.segments[index].copy()

            if labels.size:
                if self.rotate:
                    new_labels = labels.copy()
                    new_labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, 0, 0)
                    new_segments = [xyn2xy(x, w, h, 0, 0) for x in segments]
                    img, labels, segments = random_perspective(img,
                                                               new_labels,
                                                               new_segments,
                                                               degrees=180,
                                                               translate=0,
                                                               scale=0.5,
                                                               shear=0,
                                                               enlarge=True)
                    labels[:, 1:] = xyxy2xywhn(labels[:, 1:], w, h, 0, 0)
                    for x in (labels[:, 1:], *segments):
                        np.clip(x, 0, s, out=x)
                    segments = [s + (padw, padh) for s in segments]
                else:
                    segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format

            labels4.append(labels)
            segments4.extend(segments)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

        content += '\n'
        if self.saveMosaicImg:
            with open(logTxt, 'a', encoding='utf8') as f:
                f.write(content)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        # print(f'a: {segments4}')
        # print(f'1: {img4.shape}')
        img4, labels4, segments4 = random_perspective(img4,
                                                      labels4,
                                                      segments4,
                                                      degrees=self.hyp["degrees"],
                                                      translate=self.hyp["translate"],
                                                      scale=self.hyp["scale"],
                                                      shear=self.hyp["shear"],
                                                      perspective=self.hyp["perspective"],
                                                      border=self.mosaic_border)  # border to remove
        # print(f'2: {img4.shape}')
        return img4, labels4, segments4

    def load_mosaic9(self, index):
        # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                if self.rotate:
                    new_labels = labels.copy()
                    new_labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, 0, 0)
                    new_segments = [xyn2xy(x, w, h, 0, 0) for x in segments]
                    img, labels, segments = random_perspective(img,
                                                               new_labels,
                                                               new_segments,
                                                               degrees=180,
                                                               translate=0,
                                                               scale=0.5,
                                                               shear=0,
                                                               enlarge=True)
                    labels[:, 1:] = xyxy2xywhn(labels[:, 1:], w, h, 0, 0)
                    for x in (labels[:, 1:], *segments):
                        np.clip(x, 0, s, out=x)
                    segments = [s + (padx, pady) for s in segments]
                else:
                    segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format

            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        # yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y

        # img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        # labels9[:, [1, 3]] -= xc
        # labels9[:, [2, 4]] -= yc
        # c = np.array([xc, yc])  # centers
        # segments9 = [x - c for x in segments9]

        labels9[:, 1:] = labels9[:, 1:] / 3
        for (i, segment) in enumerate(segments9):
            segments9[i] = segment / 3

        img9 = cv2.resize(img9, (0, 0), fx=1/3, fy=1/3, interpolation=cv2.INTER_NEAREST)

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, p=self.hyp['copy_paste'])
        img9, labels9, segments9 = random_perspective(img9,
                                                      labels9,
                                                      segments9,
                                                      degrees=self.hyp['degrees'],
                                                      translate=self.hyp['translate'],
                                                      scale=self.hyp['scale'],
                                                      shear=self.hyp['shear'],
                                                      perspective=self.hyp['perspective'],
                                                      border=(0, 0))  # border to remove

        return img9, labels9, segments9

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, masks = zip(*batch)  # transposed
        batched_masks = torch.cat(masks, 0)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks


def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M],
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
                     dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            img_size,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index
