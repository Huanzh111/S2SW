import numpy as np
import torch
import pywt
import torch.nn.functional as F
import cv2
from sklearn.decomposition import DictionaryLearning


def apply_wavelet_transform(tensor):
    wavelet = 'haar'
    wavelet_new = 'sym2'
    batch_size, height, width = tensor.shape
    wavelet1 = pywt.Wavelet(wavelet)
    wavelet_new = pywt.Wavelet(wavelet_new)
    transformed_low_features = []
    transformed_low_features_2 = []
    for img in tensor:
        img = img.numpy()
        coeffs2 = pywt.swt2(img, wavelet_new, level=1)
        approx = coeffs2[0][0]
        horizonal = np.zeros_like(img)
        vertical = np.zeros_like(img)
        diagonal = np.zeros_like(img)
        for i in range(len(coeffs2)):
            ca, (ch, cv, cd) = coeffs2[i]
            horizonal += ch
            vertical += cv
            diagonal += cd
        combined = approx * 0.1 + (horizonal + vertical + diagonal) * 0.9
        transformed_low_features_2.append(torch.tensor(combined, dtype=tensor.dtype))
    transformed_low_features_2 = torch.stack(transformed_low_features_2)

    for i in range(batch_size):
        img = tensor[i].cpu().numpy()
        coeffs1 = pywt.dwt2(img, wavelet1)
        cA1, (cH1, cV1, cD1) = coeffs1
        High = np.stack([cH1, cV1, cD1], axis=-1)
        High1 = np.mean(High, axis=-1)
        # High1 = np.where(High1 > 0.01, High1, 1e-6)
        low_frequency = cA1 * 0.9 + High1 * 0.1
        transformed_low_features.append(low_frequency)
    # 将特征转换回 PyTorch 张量
    transformed_features_low = torch.tensor(np.array(transformed_low_features), dtype=tensor.dtype, device=tensor.device)
    # 插值调整回原始维度 # cA的尺寸可能会小于原始尺寸，这里做一个简单的插值处理
    transformed_features_low_1 = F.interpolate(transformed_features_low.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False)
    transformed_features_low_1 = transformed_features_low_1.squeeze(1)
    transformed_features = transformed_features_low_1 * 0.99 + transformed_low_features_2 * 0.01
    return transformed_features


