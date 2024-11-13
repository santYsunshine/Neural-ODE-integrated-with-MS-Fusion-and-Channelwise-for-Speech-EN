# wujian@2018
"""
SI-SNR(scale-invariant SNR/SDR) measure of speech separation
"""

import numpy as np

from itertools import permutations
import torch
import torch.functional as F
# from pypesq import pesq
def si_snr(x, s, remove_dc=True):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    if vec_l2norm(n) == 0:
        return 0
    else:
        return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))
        # return 20 * np.log10((vec_l2norm(t) + 1e-8) / (vec_l2norm(n) + 1e-8))


def permute_si_snr(xlist, slist):
    """
    Compute SI-SNR between N pairs
    Arguments:
        x: list[vector], enhanced/separated signal
        s: list[vector], reference signal(ground truth)
    """

    def si_snr_avg(xlist, slist):
        return sum([si_snr(x, s) for x, s in zip(xlist, slist)]) / len(xlist)

    N = len(xlist)
    if N != len(slist):
        raise RuntimeError(
            "size do not match between xlist and slist: {:d} vs {:d}".format(
                N, len(slist)))
    si_snrs = []
    for order in permutations(range(N)):
        si_snrs.append(si_snr_avg(xlist, [slist[n] for n in order]))
    return max(si_snrs)

def si_snr_avg(xlist, slist):
        return sum([si_snr(x, s) for x, s in zip(xlist, slist)]) / len(xlist)
        
        
        
def cal_SISNRi(x, s, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    """
    Compute SI-SNRi
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """
    sisnr1 = si_snr(x, s)
    #sisnr2 = si_snr(x, s)
    sisnr1b = si_snr(mix, s)
    #sisnr2b = si_snr(mix, s)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    #avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    avg_SISNRi = (sisnr1 - sisnr1b)
    
    return avg_SISNRi


def compute_mask(mixture, targets_list, mask_type):
    """
    Arguments:
        mixture: STFT of mixture signal(complex result) 
        targets_list: python list of target signal's STFT results(complex result)
        mask_type: ["irm", "ibm", "iam", "psm"]
    """
    if mask_type == 'ibm':
        max_index = np.argmax(
            np.stack([np.abs(mat) for mat in targets_list]), 0)
        return [max_index == s for s in range(len(targets_list))]

    if mask_type == "irm":
        denominator = sum([np.abs(mat) for mat in targets_list])
    else:
        denominator = np.abs(mixture)
    if mask_type != "psm":
        masks = [np.abs(mat) / denominator for mat in targets_list]
    else:
        mixture_phase = np.angle(mixture)
        masks = [
            np.abs(mat) * np.cos(mixture_phase - np.angle(mat)) / denominator
            for mat in targets_list
        ]
    return masks


# def pesq_metric(y_hat, bd):
#     # PESQ
#     with torch.no_grad():
#         y_hat = y_hat.cpu().numpy()
#         y = bd['y'].cpu().numpy()  # target signal

#         sum = 0
#         for i in range(len(y)):
#             sum += pesq(y[i, 0], y_hat[i, 0], SAMPLE_RATE)

#         sum /= len(y)
#         return torch.tensor(sum)
        
        

