import numpy as np
import torch, torchaudio, glob
import scipy.signal
import matplotlib.pyplot as plt

class NoiseAug(object):
    def __init__(self, noise_dir='musan_small/', prob=0.5):
        self.prob = prob
        self.noises = glob.glob(noise_dir + '*.wav')

        if len(self.noises) == 0:
            print("[WARN] No noise files found, NoiseAug disabled.")
            self.prob = 0.0

    def __call__(self, x):
        if len(self.noises) == 0:
            return x

        if np.random.uniform() < self.prob:
            n = torchaudio.load(np.random.choice(self.noises))[0][0]

            if len(n) < len(x):
                n = torch.nn.functional.pad(n, (0, len(x)-len(n)))
            elif len(n) > len(x):
                t0 = np.random.randint(0, len(n) - len(x))
                n = n[t0:t0+len(x)]

            n = n.numpy()
            snr = np.random.uniform(5, 15)
            n = n * np.sqrt(x.std()**2 / (n.std()**2 + 1e-8)) * 10**(-snr/20)
            x = x + n

        return x

    
class RIRAug(object):
    def __init__(self, rir_dir='RIRS_NOISES/simulated_rirs/', prob=0.5):
        self.prob = prob
        self.rirs = glob.glob(rir_dir + '/*/*/*.wav')
        if len(self.rirs) == 0:
            print("[WARN] No RIR files found, RIRAug disabled.")
            self.prob = 0.0

    def __call__(self, x):
        if len(self.rirs) == 0:
            return x

        if np.random.uniform() < self.prob:
            n = len(x)
            rir = torchaudio.load(np.random.choice(self.rirs))[0][0]
            rir = rir.numpy()
            rir = rir / (np.max(np.abs(rir)) + 1e-8)
            x = scipy.signal.convolve(x, rir)
            t0 = np.argmax(np.abs(rir))
            x = x[t0:t0+n]

        return x
    
def identity(x):
    return x

def plot_spectrogram(x, title):
    spec = torchaudio.transforms.MelSpectrogram(
        n_fft=512,
        win_length=25*16,
        hop_length=10*16,
        n_mels=80
    )(torch.tensor(x).unsqueeze(0))
    spec = (spec + 1e-6).log()[0].numpy()

    plt.figure(figsize=(8,3))
    plt.imshow(spec, origin='lower', aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
def wer(ref, hyp):
    ref = ref.split()
    hyp = hyp.split()

    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.uint32)

    for i in range(len(ref)+1):
        d[i][0] = i
    for j in range(len(hyp)+1):
        d[0][j] = j

    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)

    return d[len(ref)][len(hyp)] / max(1, len(ref))