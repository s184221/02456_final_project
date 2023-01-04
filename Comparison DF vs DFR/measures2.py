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


clean_wav, _ = torchaudio.load("clean_cut.wav")

df_2SNR_5, _ = torchaudio.load("2SNR-5_DF.wav")
df_2SNR0, _ = torchaudio.load("2SNR0_DF.wav")
df_2SNR5, _ = torchaudio.load("2SNR5_DF.wav")
df_2SNR10, _ = torchaudio.load("2SNR10_DF.wav")
df_2SNR15, _ = torchaudio.load("2SNR15_DF.wav")

dfR_2SNR_5, _ = torchaudio.load("2SNR-5_DFR.wav")
dfR_2SNR0, _ = torchaudio.load("2SNR0_DFR.wav")
dfR_2SNR5, _ = torchaudio.load("2SNR5_DFR.wav")
dfR_2SNR10, _ = torchaudio.load("2SNR10_DFR.wav")
dfR_2SNR15, _ = torchaudio.load("2SNR15_DFR.wav")


print("SI_SDR DF:")
print(si_sdr(df_2SNR_5, clean_wav))
print(si_sdr(df_2SNR0, clean_wav))
print(si_sdr(df_2SNR5, clean_wav))
print(si_sdr(df_2SNR10, clean_wav))
print(si_sdr(df_2SNR15, clean_wav))
print("SI_SDR DFR:")
print(si_sdr(dfR_2SNR_5, clean_wav))
print(si_sdr(dfR_2SNR0, clean_wav))
print(si_sdr(dfR_2SNR5, clean_wav))
print(si_sdr(dfR_2SNR10, clean_wav))
print(si_sdr(dfR_2SNR15, clean_wav))
print("WB-PESQ DF:")
print(wb_pesq(df_2SNR_5, clean_wav))
print(wb_pesq(df_2SNR0, clean_wav))
print(wb_pesq(df_2SNR5, clean_wav))
print(wb_pesq(df_2SNR10, clean_wav))
print(wb_pesq(df_2SNR15, clean_wav))
print("WB-PESQ DFR:")
print(wb_pesq(dfR_2SNR_5, clean_wav))
print(wb_pesq(dfR_2SNR0, clean_wav))
print(wb_pesq(dfR_2SNR5, clean_wav))
print(wb_pesq(dfR_2SNR10, clean_wav))
print(wb_pesq(dfR_2SNR15, clean_wav))