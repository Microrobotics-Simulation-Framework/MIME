"""Caldag, Demir and Yesilyurt 2022 (JFM 935 A9) Table 3, a/R = 0.5 [read, via research/NOTES_Q2.md C1]; another code's FE, not exact.
Their axes: z tube axis (mine x), rho radial towards the wall (mine z), beta tangential (mine y). Normalisation: forces 6 pi mu Rs,
torques 8 pi mu Rs^3, couplings mu Rs^2 (no 6 pi). Map to my S (force = -S U; S[j, i] force j from motion i):"""
CALDAG = {0.1: dict(zz=19.5414, yy=6.3950, xx=5.2631, rzrz=1.2664, ryry=1.7191, rxrx=1.6239, G=9.0633, Gp=-9.3387),
          0.01: dict(zz=188.4593, yy=10.9018, xx=6.7815, rzrz=1.3921, ryry=2.8805, rxrx=2.6209, G=15.5173, Gp=-5.9294)}
import numpy as np


def mine(S):
    S = np.asarray(S)
    p6, p8 = 6 * np.pi, 8 * np.pi
    return dict(zz=S[2, 2] / p6, yy=S[1, 1] / p6, xx=S[0, 0] / p6, rzrz=S[5, 5] / p8, ryry=S[4, 4] / p8, rxrx=S[3, 3] / p8,
                G=abs(S[1, 3]), Gp=-abs(S[0, 4]))
