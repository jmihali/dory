import torch
import numpy as np

# assuming torch shapes, w must already be in uint format!
# format --> [Ko, KiMajor, Qw, KiMinor] (binary tensor)
#                          +++++++++++ --> these are *contiguous and packed*
def ne16_conv1x1_unroll(w, qw, format='KoKiHW', TP_IN=16):
    s = w.shape
    if format=='KoKiHW':
        Ko,Ki,H,W = s
        wv = w
    else:
        Ko,H,W,Ki = s
        wv = w.transpose((0,3,1,2))
    nb_ki = (Ki // TP_IN + (1 if Ki % TP_IN != 0 else 0))
    wbytes = torch.zeros((Ko * nb_ki * qw, 2), dtype=torch.uint8)
    sb = wbytes.shape
    for ko in range(Ko):
        for ki in range(Ki):
            kimaj = ki // TP_IN
            kimin = ki % TP_IN
            byte  = kimin // 8
            shift = kimin % 8
            for q in range(qw):
                index = ko*nb_ki*qw + kimaj*qw + q
                wbytes[index,byte] = torch.bitwise_or(wbytes[index,byte], 1 << shift if wv[ko,ki,0,0] & (1 << q) != 0 else 0)
    wbytes = wbytes.flatten()
    return wbytes

def ne16_conv1x1_roll(wbytes, qw, s, format='KoKiHW', TP_IN=16):
    if format=='KoKiHW':
        Ko,Ki,H,W = s
        w = np.zeros((Ko, Ki, H, W), dtype=np.uint8)
        wv = w
    else:
        Ko,H,W,Ki = s
        w = np.zeros((Ko, H, W, Ki), dtype=np.uint8)
        wv = w.transpose((0,3,1,2))
    nb_ki = (Ki // TP_IN + (1 if Ki % TP_IN != 0 else 0))
    sb = wbytes.shape
    wbytes = np.asarray(wbytes, dtype=np.uint8)
    for ko in range(Ko):
        for kimaj in range(nb_ki):
            for q in range(qw):
                for kimin in range(TP_IN):
                    byte  = kimin // 8
                    shift = kimin % 8
                    index = ko*nb_ki*qw*2 + kimaj*qw*2 + q*2 + byte
                    if kimaj*TP_IN+kimin < Ki:
                        wv[ko, kimaj*TP_IN+kimin, 0, 0] += (1 & (wbytes[index] >> shift)) << q
    return w

def ne16_conv3x3_unroll(w, qw):
    s = w.shape
    nb_ki = (s[1] // 16 + (1 if s[1] % 16 != 0 else 0))
    wbytes = torch.zeros((s[0] * nb_ki * qw * s[2] * s[3], 2), dtype=torch.uint8)
    sb = wbytes.shape
    for ko in range(s[0]):
        for ki in range(s[1]):
            for fs0 in range(s[2]):
                for fs1 in range(s[3]):
                    kimaj = ki // 16
                    kimin = ki % 16
                    byte  = kimin // 8
                    shift = kimin % 8
                    for q in range(qw):
                        index = ko*nb_ki*qw*s[2]*s[3] + kimaj*qw*s[2]*s[3] + q*s[2]*s[3] + fs0*s[3] + fs1
                        wbytes[index,byte] = torch.bitwise_or(wbytes[index,byte], 1 << shift if w[ko,ki,fs0,fs1] & (1 << q) != 0 else 0)
    wbytes = wbytes.flatten()
    return wbytes
