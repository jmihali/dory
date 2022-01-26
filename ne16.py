import torch

# assuming torch shapes, w must already be in uint format!
def ne16_conv1x1_unroll(w, qw):
    s = w.shape
    nb_ki = (s[1] // 16 + (1 if s[1] % 16 != 0 else 0))
    wbytes = torch.zeros((s[0] * nb_ki * qw, 2), dtype=torch.uint8)
    sb = wbytes.shape
    for ko in range(s[0]):
        for ki in range(s[1]):
            kimaj = ki // 16
            kimin = ki % 16
            byte  = kimin // 8
            shift = kimin % 8
            for q in range(qw):
                index = ko*nb_ki*qw + kimaj*qw + q
                wbytes[index,byte] = torch.bitwise_or(wbytes[index,byte], 1 << shift if w[ko,ki,0,0] & (1 << q) != 0 else 0)
    wbytes = wbytes.flatten()
    return wbytes

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
