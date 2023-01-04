import torchaudio
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.functional.audio import signal_noise_ratio


import numpy as np

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

si_sdr = ScaleInvariantSignalDistortionRatio()
wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')


clean_wav, _ = torchaudio.load("clean_cut_2_untrained_on.wav")
SNR_5, _ = torchaudio.load("2SNR-5.wav")
SNR0, _ = torchaudio.load("2SNR0.wav")
SNR5, _ = torchaudio.load("2SNR5.wav")
SNR10, _ = torchaudio.load("2SNR10.wav")
SNR15, _ = torchaudio.load("2SNR15.wav")

print(signal_noise_ratio(SNR_5,clean_wav))
print(signal_noise_ratio(SNR0,clean_wav))
print(signal_noise_ratio(SNR5,clean_wav))
print(signal_noise_ratio(SNR10,clean_wav))
print(signal_noise_ratio(SNR15,clean_wav))