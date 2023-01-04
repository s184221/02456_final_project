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


clean_wav, _ = torchaudio.load("Comparison AH vs stock DF/clean_cut.wav")

df_SNR_5, _ = torchaudio.load("Comparison AH vs stock DF/SNR-5_DeepFilterNet.wav")
df_SNR0, _ = torchaudio.load("Comparison AH vs stock DF/SNR0_DeepFilterNet.wav")
df_SNR5, _ = torchaudio.load("Comparison AH vs stock DF/SNR5_DeepFilterNet.wav")
df_SNR10, _ = torchaudio.load("Comparison AH vs stock DF/SNR10_DeepFilterNet.wav")
df_SNR15, _ = torchaudio.load("Comparison AH vs stock DF/SNR15_DeepFilterNet.wav")

df2_SNR_5, _ = torchaudio.load("Comparison AH vs stock DF/SNR-5_DeepFilterNet2.wav")
df2_SNR0, _ = torchaudio.load("Comparison AH vs stock DF/SNR0_DeepFilterNet2.wav")
df2_SNR5, _ = torchaudio.load("Comparison AH vs stock DF/SNR5_DeepFilterNet2.wav")
df2_SNR10, _ = torchaudio.load("Comparison AH vs stock DF/SNR10_DeepFilterNet2.wav")
df2_SNR15, _ = torchaudio.load("Comparison AH vs stock DF/SNR15_DeepFilterNet2.wav")

ah_SNR_5, _ = torchaudio.load("Comparison AH vs stock DF/SNR-5_AugmentedHearing.wav")
ah_SNR0, _ = torchaudio.load("Comparison AH vs stock DF/SNR0_AugmentedHearing.wav")
ah_SNR5, _ = torchaudio.load("Comparison AH vs stock DF/SNR5_AugmentedHearing.wav")
ah_SNR10, _ = torchaudio.load("Comparison AH vs stock DF/SNR10_AugmentedHearing.wav")
ah_SNR15, _ = torchaudio.load("Comparison AH vs stock DF/SNR15_AugmentedHearing.wav")

print("SI_SDR DF:")
print(si_sdr(df_SNR_5, clean_wav))
print(si_sdr(df_SNR0, clean_wav))
print(si_sdr(df_SNR5, clean_wav))
print(si_sdr(df_SNR10, clean_wav))
print(si_sdr(df_SNR15, clean_wav))
print("SI_SDR DF2:")
print(si_sdr(df2_SNR_5, clean_wav))
print(si_sdr(df2_SNR0, clean_wav))
print(si_sdr(df2_SNR5, clean_wav))
print(si_sdr(df2_SNR10, clean_wav))
print(si_sdr(df2_SNR15, clean_wav))
print("SI_SDR AH:")
print(si_sdr(ah_SNR_5, clean_wav))
print(si_sdr(ah_SNR0, clean_wav))
print(si_sdr(ah_SNR5, clean_wav))
print(si_sdr(ah_SNR10, clean_wav))
print(si_sdr(ah_SNR15, clean_wav))
print("WB-PESQ DF:")
print(wb_pesq(df_SNR_5, clean_wav))
print(wb_pesq(df_SNR0, clean_wav))
print(wb_pesq(df_SNR5, clean_wav))
print(wb_pesq(df_SNR10, clean_wav))
print(wb_pesq(df_SNR15, clean_wav))
print("WB-PESQ DF2:")
print(wb_pesq(df2_SNR_5, clean_wav))
print(wb_pesq(df2_SNR0, clean_wav))
print(wb_pesq(df2_SNR5, clean_wav))
print(wb_pesq(df2_SNR10, clean_wav))
print(wb_pesq(df2_SNR15, clean_wav))
print("WB-PESQ AH:")
print(wb_pesq(ah_SNR_5, clean_wav))
print(wb_pesq(ah_SNR0, clean_wav))
print(wb_pesq(ah_SNR5, clean_wav))
print(wb_pesq(ah_SNR10, clean_wav))
print(wb_pesq(ah_SNR15, clean_wav))
