'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import torch as th

IMmean = th.tensor([0.485, 0.456, 0.406])
IMstd = th.tensor([0.229, 0.224, 0.225])


def renorm(im): return im.sub(IMmean[:, None, None].to(
    im.device)).div(IMstd[:, None, None].to(im.device))


def denorm(im): return im.mul(IMstd[:, None, None].to(
    im.device)).add(IMmean[:, None, None].to(im.device))


def xdnorm(im): return im.div(IMstd[:, None, None].to(
    im.device)).add(IMmean[:, None, None].to(im.device))


def chw2hwc(im): return im.transpose((0, 2, 3, 1)) if len(
    im.shape) == 4 else im.transpose((1, 2, 0))


preproc_dict = dict(mean=[0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225], axis=-3)

if __name__ == '__main__':
    x = th.rand(10, 3, 224, 224)
    print(th.sum(th.abs(x - denorm(renorm(x)))))
    print(th.sum(th.abs(x - renorm(denorm(x)))))
