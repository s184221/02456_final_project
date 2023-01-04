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

df_SNR_5, _ = torchaudio.load("SNR-5_DF.wav")
df_SNR0, _ = torchaudio.load("SNR0_DF.wav")
df_SNR5, _ = torchaudio.load("SNR5_DF.wav")
df_SNR10, _ = torchaudio.load("SNR10_DF.wav")
df_SNR15, _ = torchaudio.load("SNR15_DF.wav")

dfR_SNR_5, _ = torchaudio.load("SNR-5_DFR.wav")
dfR_SNR0, _ = torchaudio.load("SNR0_DFR.wav")
dfR_SNR5, _ = torchaudio.load("SNR5_DFR.wav")
dfR_SNR10, _ = torchaudio.load("SNR10_DFR.wav")
dfR_SNR15, _ = torchaudio.load("SNR15_DFR.wav")


print("SI_SDR DF:")
print(si_sdr(df_SNR_5, clean_wav))
print(si_sdr(df_SNR0, clean_wav))
print(si_sdr(df_SNR5, clean_wav))
print(si_sdr(df_SNR10, clean_wav))
print(si_sdr(df_SNR15, clean_wav))
print("SI_SDR DFR:")
print(si_sdr(dfR_SNR_5, clean_wav))
print(si_sdr(dfR_SNR0, clean_wav))
print(si_sdr(dfR_SNR5, clean_wav))
print(si_sdr(dfR_SNR10, clean_wav))
print(si_sdr(dfR_SNR15, clean_wav))
print("WB-PESQ DF:")
print(wb_pesq(df_SNR_5, clean_wav))
print(wb_pesq(df_SNR0, clean_wav))
print(wb_pesq(df_SNR5, clean_wav))
print(wb_pesq(df_SNR10, clean_wav))
print(wb_pesq(df_SNR15, clean_wav))
print("WB-PESQ DFR:")
print(wb_pesq(dfR_SNR_5, clean_wav))
print(wb_pesq(dfR_SNR0, clean_wav))
print(wb_pesq(dfR_SNR5, clean_wav))
print(wb_pesq(dfR_SNR10, clean_wav))
print(wb_pesq(dfR_SNR15, clean_wav))