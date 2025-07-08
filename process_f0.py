import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np
import json
import os
from infer.lib.rmvpe import RMVPE
from tqdm import tqdm
from collections import defaultdict
import glob
import argparse

model_path = "/mnt/workspace/yinghaotian/code/whisperVC-master/ckpts/rmvpe.pt"
rmvpe = RMVPE(model_path, is_half=False, device='cuda')

class AudioFolderDataset(Dataset):
    def __init__(self, audio_folder, sampling_rate=16000, extensions=('.wav', '.mp3', '.flac', '.m4a')):
        # 获取文件夹中所有音频文件
        self.audio_files = []
        for ext in extensions:
            self.audio_files.extend(glob.glob(os.path.join(audio_folder, f"**/*{ext}"), recursive=True))
        
        self.sampling_rate = sampling_rate
        print(f"Found {len(self.audio_files)} audio files in {audio_folder}")
        
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        wav_path = self.audio_files[idx]
        
        # 加载音频
        try:
            waveform, sr = torchaudio.load(wav_path)
            waveform = waveform.squeeze(0)  # 单通道
            if sr != self.sampling_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)(waveform)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # 跳过错误文件

        return waveform, wav_path

def collate_fn(batch):
    waveforms, wav_paths = zip(*batch)
    max_len = max(w.shape[0] for w in waveforms)
    waveforms_tensor = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        waveforms_tensor[i, :w.shape[0]] = w
    return {
        "waveform": waveforms_tensor,
        "wav_path": wav_paths
    }

def batch_extract_f0_from_folder(dataloader, device='cuda'):
    """
    从文件夹中批量提取F0并计算统计信息
    Args:
        dataloader: DataLoader 实例
        device: 'cuda' or 'cpu'
    Returns:
        results: List of dict 包含 wav_path, f0_curve
        f0_stats: {median, mean, std}
    """
    results = []
    all_voiced_f0s = []

    for batch in tqdm(dataloader, desc="Processing audio files"):
        waveforms = batch["waveform"].to(device)
        wav_paths = batch["wav_path"]

        # 批量提取F0
        with torch.no_grad():
            clean_16k = waveforms.cpu().numpy()
            f0s_gt_raw = rmvpe.infer_from_audio_batch(clean_16k, thred=0.03)

        # 处理每个样本
        for i in range(len(f0s_gt_raw)):
            f0 = f0s_gt_raw[i]
            uv_mask = (f0 == 0)
            
            # 构建结果条目
            result = {
                "wav_path": wav_paths[i],
                "f0_curve": f0.tolist(),
                "voiced_frames": int(torch.sum(~uv_mask)),
                "total_frames": len(f0)
            }
            results.append(result)

            # 收集 voiced F0 用于统计
            voiced_f0 = f0[~uv_mask]
            if len(voiced_f0) > 0:
                all_voiced_f0s.extend(voiced_f0.tolist())

    # 计算统计信息
    if all_voiced_f0s:
        f0_stats = {
            "median": float(np.median(all_voiced_f0s)),
            "mean": float(np.mean(all_voiced_f0s)),
            "std": float(np.std(all_voiced_f0s)),
            "min": float(np.min(all_voiced_f0s)),
            "max": float(np.max(all_voiced_f0s)),
            "total_voiced_frames": len(all_voiced_f0s)
        }
    else:
        f0_stats = {
            "median": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "total_voiced_frames": 0
        }

    return results, f0_stats

def process_audio_folder(audio_folder, output_file=None, batch_size=32, num_workers=4):
    """
    处理音频文件夹，提取F0并计算统计信息
    Args:
        audio_folder: 音频文件夹路径
        output_file: 输出JSON文件路径（可选）
        batch_size: 批处理大小
        num_workers: 数据加载器工作进程数
    """
    # 创建数据集
    dataset = AudioFolderDataset(audio_folder)
    
    if len(dataset) == 0:
        print("No audio files found in the folder!")
        return None, None
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 批量提取 F0
    results, f0_stats = batch_extract_f0_from_folder(dataloader)

    # 打印统计信息
    print("\n=== F0 统计信息 ===")
    print(f"总音频文件数: {len(results)}")
    print(f"总有声帧数: {f0_stats['total_voiced_frames']}")
    if f0_stats['median'] is not None:
        print(f"F0 中位数: {f0_stats['median']:.2f} Hz")
        print(f"F0 平均值: {f0_stats['mean']:.2f} Hz")
        print(f"F0 标准差: {f0_stats['std']:.2f} Hz")
        print(f"F0 最小值: {f0_stats['min']:.2f} Hz")
        print(f"F0 最大值: {f0_stats['max']:.2f} Hz")
        
        # 保存F0统计信息到单独的JSON文件
        f0_stats_file = os.path.join(os.path.dirname(output_file) if output_file else audio_folder, "f0_statistics.json")
        f0_stats_output = {
            "total_audio_files": len(results),
            "total_voiced_frames": f0_stats['total_voiced_frames'],
            "f0_median_hz": f0_stats['median'],
            "f0_mean_hz": f0_stats['mean'],
            "f0_std_hz": f0_stats['std'],
            "f0_min_hz": f0_stats['min'],
            "f0_max_hz": f0_stats['max']
        }
        
        with open(f0_stats_file, 'w', encoding='utf-8') as f:
            json.dump(f0_stats_output, f, indent=4, ensure_ascii=False)
        
        print(f"\nF0统计信息已保存到: {f0_stats_file}")
    else:
        print("未检测到有声帧")

    # # 保存结果
    # if output_file:
    #     output_data = {
    #         "f0_statistics": f0_stats,
    #         "file_results": results
    #     }
        
    #     with open(output_file, 'w') as f:
    #         json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    #     print(f"\n结果已保存到: {output_file}")

    return results, f0_stats

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量提取音频文件夹中的F0并计算统计信息")
    
    # 必需参数
    parser.add_argument("audio_folder", type=str, help="音频文件夹路径")
    
    # 可选参数
    parser.add_argument("--output_file", "-o", type=str, default=None, 
                       help="输出JSON文件路径（可选）")
    parser.add_argument("--batch_size", "-b", type=int, default=32, 
                       help="批处理大小 (默认: 32)")
    parser.add_argument("--num_workers", "-w", type=int, default=4, 
                       help="数据加载器工作进程数 (默认: 4)")
    parser.add_argument("--device", "-d", type=str, default="cuda", 
                       choices=["cuda", "cpu"], help="计算设备 (默认: cuda)")
    
    args = parser.parse_args()
    
    # 如果没有指定输出文件，则在音频文件夹中创建默认文件名
    if args.output_file is None:
        args.output_file = os.path.join(args.audio_folder, "f0_results.json")
    
    print(f"音频文件夹: {args.audio_folder}")
    print(f"输出文件: {args.output_file}")
    print(f"批处理大小: {args.batch_size}")
    print(f"工作进程数: {args.num_workers}")
    print(f"计算设备: {args.device}")
    
    # 处理文件夹
    results, f0_stats = process_audio_folder(
        audio_folder=args.audio_folder,
        output_file=args.output_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )