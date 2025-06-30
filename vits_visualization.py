def visualize_training_step(self, net_g, net_d, save_path):
        """可视化完整的训练步骤"""
        print("可视化训练步骤...")
        
        net_g.train()
        net_d.train()
        
        # 检查模型兼容性
        if not self.check_model_compatibility(net_g):
            print("模型兼容性检查失败，跳过训练步骤可视化")
            return
        
        sample_data = self.create_sample_data_safe(net_g)
        
        try:
            # 前向传播
            if self.hps.if_f0 == 1:
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = sample_data
                output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = sample_data
                output = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            
            y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = output
            
            # Mel频谱图处理
            mel = spec_to_mel_torch(
                spec,
                self.hps.data.filter_length,
                self.hps.data.n_mel_channels,
                self.hps.data.sampling_rate,
                self.hps.data.mel_fmin,
                self.hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, self.hps.train.segment_size // self.hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.float().squeeze(1),
                self.hps.data.filter_length,
                self.hps.data.n_mel_channels,
                self.hps.data.sampling_rate,
                self.hps.data.hop_length,
                self.hps.data.win_length,
                self.hps.data.mel_fmin,
                self.hps.data.mel_fmax,
            )
            
            wave = commons.slice_segments(
                wave, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size
            )
            
            # 判别器前向传播
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            
            # 计算损失
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * self.hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            
            # 创建训练损失的计算图
            dot = make_dot(loss_gen_all, 
                          params=dict(list(net_g.named_parameters()) + list(net_d.named_parameters())),
                          show_attrs=False, show_saved=False)
            
            dot.attr(rankdir='TB')
            dot.attr('node', shape='box', fontsize='8')
            dot.attr('edge', fontsize='6')
            dot.format = 'png'
            
            dot.render(save_path.replace('.png', ''), cleanup=True)
            print(f"训练步骤可视化保存到: {save_path}")
            
            print(f"损失值 - 判别器: {loss_disc:.4f}, 生成器: {loss_gen_all:.4f}")
            print(f"  - Mel损失: {loss_mel:.4f}, KL损失: {loss_kl:.4f}, FM损失: {loss_fm:.4f}")
            
        except Exception as e:
            print(f"可视化训练步骤失败: {e}")
            import traceback
            traceback.print_exc()
import os
import sys
import logging
import argparse
import torch
from torchviz import make_dot
import graphviz

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from infer.lib.train import utils
from infer.lib.train.data_utils import (
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)
from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.infer_pack import commons

# 导入模型
try:
    from infer.lib.infer_pack.models import MultiPeriodDiscriminator
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0_v1
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0_v1
    from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid as RVC_Model_f0_v2
    from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0_v2
    from infer.lib.infer_pack.models import MultiPeriodDiscriminatorV2
except ImportError as e:
    print(f"模型导入失败: {e}")
    sys.exit(1)

class RVCModelVisualizer:
    def __init__(self, hps):
        self.hps = hps
        self.device = torch.device('cpu')  # 可视化时使用CPU
        self.output_dir = "./rvc_training_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_models(self):
        """创建生成器和判别器模型"""
        print(f"创建模型 - 版本: {self.hps.version}, F0: {self.hps.if_f0}")
        
        # 将model对象转换为字典
        model_config = {
            'inter_channels': self.hps.model.inter_channels,
            'hidden_channels': self.hps.model.hidden_channels,
            'filter_channels': self.hps.model.filter_channels,
            'n_heads': self.hps.model.n_heads,
            'n_layers': self.hps.model.n_layers,
            'kernel_size': self.hps.model.kernel_size,
            'p_dropout': self.hps.model.p_dropout,
            'resblock': self.hps.model.resblock,
            'resblock_kernel_sizes': self.hps.model.resblock_kernel_sizes,
            'resblock_dilation_sizes': self.hps.model.resblock_dilation_sizes,
            'upsample_rates': self.hps.model.upsample_rates,
            'upsample_initial_channel': self.hps.model.upsample_initial_channel,
            'upsample_kernel_sizes': self.hps.model.upsample_kernel_sizes,
            'spk_embed_dim': self.hps.model.spk_embed_dim,
            'gin_channels': self.hps.model.gin_channels,
        }
        
        # 创建生成器
        if self.hps.version == "v1":
            if self.hps.if_f0 == 1:
                net_g = RVC_Model_f0_v1(
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    **model_config,
                    is_half=False,  # 可视化时不使用半精度
                    sr=self.hps.sample_rate,
                )
            else:
                net_g = RVC_Model_nof0_v1(
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    **model_config,
                    is_half=False,
                )
            net_d = MultiPeriodDiscriminator(self.hps.model.use_spectral_norm)
        else:  # v2
            if self.hps.if_f0 == 1:
                net_g = RVC_Model_f0_v2(
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    **model_config,
                    is_half=False,
                    sr=self.hps.sample_rate,
                )
            else:
                net_g = RVC_Model_nof0_v2(
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    **model_config,
                    is_half=False,
                )
            net_d = MultiPeriodDiscriminatorV2(self.hps.model.use_spectral_norm)
        
        return net_g.to(self.device), net_d.to(self.device)
    
    def check_model_compatibility(self, net_g):
        """检查模型兼容性"""
        print("检查模型兼容性...")
        
        # 检查说话人嵌入层
        if hasattr(net_g, 'emb_g'):
            num_speakers = net_g.emb_g.num_embeddings
            embedding_dim = net_g.emb_g.embedding_dim
            print(f"说话人嵌入层: {num_speakers} 个说话人, 维度 {embedding_dim}")
            
            # 确保我们的配置与模型匹配
            if num_speakers == 1:
                print("检测到单说话人模型")
                return True
            else:
                print(f"检测到多说话人模型 ({num_speakers} 个说话人)")
                # 更新配置以匹配模型
                self.hps.model.spk_embed_dim = num_speakers
                return True
        else:
            print("未找到说话人嵌入层")
            return False
    
    def create_sample_data_safe(self, net_g):
        """安全地创建示例数据"""
        batch_size = min(self.hps.train.batch_size, 2)  # 限制批大小
        
        # 音素序列长度
        phone_len = 50  # 减少序列长度
        spec_len = 50
        segment_size = min(self.hps.train.segment_size, 4096)  # 减少段大小
        
        # 创建示例数据
        if self.hps.version == "v1":
            phone_dim = 256
        else:
            phone_dim = 768
            
        phone = torch.randn(batch_size, phone_len, phone_dim)
        phone_lengths = torch.tensor([phone_len] * batch_size)
        spec = torch.randn(batch_size, self.hps.data.filter_length // 2 + 1, spec_len)
        spec_lengths = torch.tensor([spec_len] * batch_size)
        wave = torch.randn(batch_size, 1, segment_size)
        wave_lengths = torch.tensor([segment_size] * batch_size)
        
        # 安全地创建说话人ID
        if hasattr(net_g, 'emb_g'):
            max_speakers = net_g.emb_g.num_embeddings
            sid = torch.zeros(batch_size, dtype=torch.long)  # 始终使用说话人0
            if max_speakers <= sid.max():
                print(f"警告: 说话人ID {sid.max()} 超出范围 [0, {max_speakers-1}]")
                sid = torch.zeros(batch_size, dtype=torch.long)
        else:
            sid = torch.zeros(batch_size, dtype=torch.long)
        
        if self.hps.if_f0 == 1:
            pitch = torch.randint(1, 256, (batch_size, phone_len))
            pitchf = torch.randn(batch_size, phone_len)
            return phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid
        else:
            return phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid
    
    def visualize_generator_forward(self, net_g, save_path):
        """可视化生成器前向传播"""
        print("可视化生成器前向传播...")
        net_g.eval()
        
        # 检查模型兼容性
        if not self.check_model_compatibility(net_g):
            print("模型兼容性检查失败，跳过生成器可视化")
            return
        
        sample_data = self.create_sample_data_safe(net_g)
        
        try:
            with torch.no_grad():
                if self.hps.if_f0 == 1:
                    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = sample_data
                    print(f"输入形状检查:")
                    print(f"  phone: {phone.shape}")
                    print(f"  phone_lengths: {phone_lengths}")
                    print(f"  pitch: {pitch.shape}")
                    print(f"  pitchf: {pitchf.shape}")
                    print(f"  spec: {spec.shape}")
                    print(f"  spec_lengths: {spec_lengths}")
                    print(f"  sid: {sid} (range: {sid.min()}-{sid.max()})")
                    
                    output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                else:
                    phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = sample_data
                    print(f"输入形状检查:")
                    print(f"  phone: {phone.shape}")
                    print(f"  phone_lengths: {phone_lengths}")
                    print(f"  spec: {spec.shape}")
                    print(f"  spec_lengths: {spec_lengths}")
                    print(f"  sid: {sid} (range: {sid.min()}-{sid.max()})")
                    
                    output = net_g(phone, phone_lengths, spec, spec_lengths, sid)
                
                y_hat = output[0]  # 生成的音频
                
                # 创建计算图
                dot = make_dot(y_hat, params=dict(net_g.named_parameters()),
                              show_attrs=False, show_saved=False)
                
                # 设置图形属性
                dot.attr(rankdir='TB')
                dot.attr('node', shape='box', fontsize='10')
                dot.attr('edge', fontsize='8')
                dot.format = 'png'
                
                # 保存图形
                dot.render(save_path.replace('.png', ''), cleanup=True)
                print(f"生成器可视化保存到: {save_path}")
                
                # 打印模型信息
                total_params = sum(p.numel() for p in net_g.parameters())
                print(f"生成器参数量: {total_params:,}")
                print(f"输出形状: {y_hat.shape}")
                
        except Exception as e:
            print(f"可视化生成器失败: {e}")
            print("尝试调试模型结构...")
            
            # 调试信息
            if hasattr(net_g, 'emb_g'):
                print(f"说话人嵌入层形状: {net_g.emb_g.weight.shape}")
                print(f"可接受的说话人ID范围: 0-{net_g.emb_g.num_embeddings-1}")
            
            import traceback
            traceback.print_exc()
    
    def visualize_discriminator_forward(self, net_d, save_path):
        """可视化判别器前向传播"""
        print("可视化判别器前向传播...")
        net_d.eval()
        
        # 创建示例音频数据
        batch_size = self.hps.train.batch_size
        segment_size = self.hps.train.segment_size
        audio_real = torch.randn(batch_size, 1, segment_size)
        audio_fake = torch.randn(batch_size, 1, segment_size)
        
        try:
            with torch.no_grad():
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(audio_real, audio_fake)
                
                # 选择第一个尺度的输出进行可视化
                main_output = y_d_hat_r[0] if isinstance(y_d_hat_r, list) else y_d_hat_r
                
                dot = make_dot(main_output, params=dict(net_d.named_parameters()),
                              show_attrs=False, show_saved=False)
                
                dot.attr(rankdir='TB')
                dot.attr('node', shape='box', fontsize='10')
                dot.attr('edge', fontsize='8')
                dot.format = 'png'
                
                dot.render(save_path.replace('.png', ''), cleanup=True)
                print(f"判别器可视化保存到: {save_path}")
                
                total_params = sum(p.numel() for p in net_d.parameters())
                print(f"判别器参数量: {total_params:,}")
                
        except Exception as e:
            print(f"可视化判别器失败: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_training_step(self, net_g, net_d, save_path):
        """可视化完整的训练步骤"""
        print("可视化训练步骤...")
        
        net_g.train()
        net_d.train()
        
        sample_data = self.create_sample_data_safe(net_g)
        
        try:
            # 前向传播
            if self.hps.if_f0 == 1:
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = sample_data
                output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = sample_data
                output = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            
            y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = output
            
            # Mel频谱图处理
            mel = spec_to_mel_torch(
                spec,
                self.hps.data.filter_length,
                self.hps.data.n_mel_channels,
                self.hps.data.sampling_rate,
                self.hps.data.mel_fmin,
                self.hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, self.hps.train.segment_size // self.hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.float().squeeze(1),
                self.hps.data.filter_length,
                self.hps.data.n_mel_channels,
                self.hps.data.sampling_rate,
                self.hps.data.hop_length,
                self.hps.data.win_length,
                self.hps.data.mel_fmin,
                self.hps.data.mel_fmax,
            )
            
            wave = commons.slice_segments(
                wave, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size
            )
            
            # 判别器前向传播
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            
            # 计算损失
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * self.hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            
            # 创建训练损失的计算图
            dot = make_dot(loss_gen_all, 
                          params=dict(list(net_g.named_parameters()) + list(net_d.named_parameters())),
                          show_attrs=False, show_saved=False)
            
            dot.attr(rankdir='TB')
            dot.attr('node', shape='box', fontsize='8')
            dot.attr('edge', fontsize='6')
            dot.format = 'png'
            
            dot.render(save_path.replace('.png', ''), cleanup=True)
            print(f"训练步骤可视化保存到: {save_path}")
            
            print(f"损失值 - 判别器: {loss_disc:.4f}, 生成器: {loss_gen_all:.4f}")
            print(f"  - Mel损失: {loss_mel:.4f}, KL损失: {loss_kl:.4f}, FM损失: {loss_fm:.4f}")
            
        except Exception as e:
            print(f"可视化训练步骤失败: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_data_flow(self, save_path):
        """可视化数据流"""
        print("可视化数据流...")
        
        # 创建数据流图
        dot = graphviz.Digraph(comment='RVC Data Flow with Parameters')
        dot.attr(rankdir='LR')
        dot.attr('node', shape='box', fontname='Arial')
        dot.attr('edge', fontname='Arial')
        
        # 获取配置参数
        batch_size = self.hps.train.batch_size
        phone_dim = 768 if self.hps.version == "v2" else 256
        spec_dim = self.hps.data.filter_length // 2 + 1  # 1025
        hop_length = self.hps.data.hop_length
        segment_size = self.hps.train.segment_size
        sample_rate = self.hps.sample_rate
        
        # 输入数据节点
        dot.node('A', f'Raw Audio\n[batch_size, 1, {segment_size}]\nSample Rate: {sample_rate}Hz', 
                 style='filled', fillcolor='lightblue')
        dot.node('B', f'Phone Sequence\n[batch_size, seq_len, {phone_dim}]\nHuBERT Features', 
                 style='filled', fillcolor='lightblue')
        if self.hps.if_f0:
            dot.node('C', f'F0 Information\nPitch: [batch_size, seq_len]\nPitchF: [batch_size, seq_len]\nRange: 50-1100Hz', 
                     style='filled', fillcolor='lightblue')
        else:
            dot.node('C', f'Spectrogram\n[batch_size, {spec_dim}, seq_len]\nMel Features', 
                     style='filled', fillcolor='lightblue')
        dot.node('D', f'Speaker ID\n[batch_size]\nRange: 0-{self.hps.model.spk_embed_dim-1}', 
                 style='filled', fillcolor='lightblue')
        
        # 编码器模块
        dot.node('E1', f'Text Encoder\nInput: [batch_size, seq_len, {phone_dim}]\nOutput: [batch_size, {self.hps.model.hidden_channels}, seq_len]', 
                 style='filled', fillcolor='lightgreen')
        dot.node('E2', f'Posterior Encoder\nInput: [batch_size, {spec_dim}, seq_len]\nOutput: [batch_size, {self.hps.model.inter_channels}, seq_len]', 
                 style='filled', fillcolor='lightgreen')
        
        # 说话人嵌入
        dot.node('E3', f'Speaker Embedding\nInput: [batch_size]\nOutput: [batch_size, {self.hps.model.gin_channels}]\nEmbedding Dim: {self.hps.model.gin_channels}', 
                 style='filled', fillcolor='lightgreen')
        
        # 随机采样模块
        dot.node('G', f'Stochastic Sampling\nLatent Dim: {self.hps.model.inter_channels}\nKL Divergence Loss', 
                 style='filled', fillcolor='lightyellow')
        
        # 流模块 (如果有F0)
        if self.hps.if_f0:
            dot.node('F', f'Flow Module\nF0 Conditioning\nInput: Pitch + PitchF\nNormalizing Flows', 
                     style='filled', fillcolor='lightyellow')
        
        # 解码器
        upsample_rates = self.hps.model.upsample_rates
        total_upsample = 1
        for rate in upsample_rates:
            total_upsample *= rate
        
        dot.node('H', f'Decoder (Generator)\nUpsample Rates: {upsample_rates}\nTotal Upsample: {total_upsample}x\nInput: [batch_size, {self.hps.model.inter_channels}, seq_len]\nOutput: [batch_size, 1, {segment_size}]', 
                 style='filled', fillcolor='orange')
        
        # 判别器
        dot.node('I', f'Multi-Period Discriminator\nPeriods: [2,3,5,7,11]\nInput: [batch_size, 1, {segment_size}]\nOutput: Real/Fake + Feature Maps', 
                 style='filled', fillcolor='pink')
        
        # 损失计算
        dot.node('L1', f'Generator Losses\nMel Loss (L1): weight={self.hps.train.c_mel}\nKL Loss: weight={self.hps.train.c_kl}\nFeature Matching Loss\nAdversarial Loss', 
                 style='filled', fillcolor='lightcoral')
        dot.node('L2', f'Discriminator Loss\nReal vs Fake\nMulti-Scale Loss', 
                 style='filled', fillcolor='lightcoral')
        
        # 输出
        dot.node('J', f'Generated Audio\n[batch_size, 1, {segment_size}]\nSample Rate: {sample_rate}Hz', 
                 style='filled', fillcolor='lightgray')
        
        # 连接关系
        # 编码路径
        dot.edge('B', 'E1', label='phone')
        dot.edge('A', 'E2', label='spec')
        dot.edge('D', 'E3', label='sid')
        
        # 流向采样模块
        dot.edge('E1', 'G', label=f'[{self.hps.model.hidden_channels}, seq_len]')
        dot.edge('E2', 'G', label=f'[{self.hps.model.inter_channels}, seq_len]')
        
        # F0处理
        if self.hps.if_f0:
            dot.edge('C', 'F', label='pitch info')
            dot.edge('F', 'H', label='conditioned z')
            dot.edge('G', 'F', label='latent z')
        else:
            dot.edge('G', 'H', label='latent z')
            dot.edge('C', 'H', label='spec features')
        
        # 说话人信息到解码器
        dot.edge('E3', 'H', label=f'[{self.hps.model.gin_channels}]')
        
        # 解码器到输出
        dot.edge('H', 'J', label='audio')
        
        # 判别器路径
        dot.edge('A', 'I', label='real audio')
        dot.edge('J', 'I', label='fake audio')
        
        # 损失计算
        dot.edge('I', 'L1', label='D features')
        dot.edge('I', 'L2', label='D outputs')
        dot.edge('H', 'L1', label='G outputs')
        
        # 添加模型信息标签
        model_info = f'RVC {self.hps.version.upper()} Model\\n'
        model_info += f'F0: {"Enabled" if self.hps.if_f0 else "Disabled"}\\n'
        model_info += f'Sample Rate: {sample_rate}Hz\\n'
        model_info += f'Hop Length: {hop_length}\\n'
        model_info += f'Batch Size: {batch_size}'
        
        dot.node('INFO', model_info, shape='note', style='filled', fillcolor='lightyellow')
        
        dot.render(save_path.replace('.png', ''), format='png', cleanup=True)
        print(f"数据流图保存到: {save_path}")
        
        # 打印参数摘要
        print("\nData Flow Parameters Summary:")
        print(f"  Model Version: RVC {self.hps.version.upper()}")
        print(f"  F0 Mode: {'Enabled' if self.hps.if_f0 else 'Disabled'}")
        print(f"  Phone Dimension: {phone_dim}")
        print(f"  Spectrogram Dimension: {spec_dim}")
        print(f"  Hidden Channels: {self.hps.model.hidden_channels}")
        print(f"  Inter Channels: {self.hps.model.inter_channels}")
        print(f"  Speaker Embedding Dim: {self.hps.model.gin_channels}")
        print(f"  Upsample Rates: {upsample_rates}")
        print(f"  Total Upsample Factor: {total_upsample}x")
        print(f"  Segment Size: {segment_size}")
        print(f"  Sample Rate: {sample_rate}Hz")
    
    def print_model_summary(self, net_g, net_d):
        """打印模型摘要"""
        print("\n" + "="*60)
        print("RVC模型架构摘要")
        print("="*60)
        
        # 生成器信息
        total_params_g = sum(p.numel() for p in net_g.parameters())
        trainable_params_g = sum(p.numel() for p in net_g.parameters() if p.requires_grad)
        
        print(f"生成器 ({self.hps.version}):")
        print(f"  总参数量: {total_params_g:,}")
        print(f"  可训练参数: {trainable_params_g:,}")
        print(f"  F0模式: {'启用' if self.hps.if_f0 else '禁用'}")
        print(f"  采样率: {self.hps.sample_rate}")
        
        # 判别器信息
        total_params_d = sum(p.numel() for p in net_d.parameters())
        trainable_params_d = sum(p.numel() for p in net_d.parameters() if p.requires_grad)
        
        print(f"\n判别器 ({self.hps.version}):")
        print(f"  总参数量: {total_params_d:,}")
        print(f"  可训练参数: {trainable_params_d:,}")
        
        print(f"\n总计:")
        print(f"  所有参数: {total_params_g + total_params_d:,}")
        print(f"  模型大小估计: {(total_params_g + total_params_d) * 4 / (1024**2):.1f} MB")
        
        # 训练配置
        print(f"\n训练配置:")
        print(f"  批大小: {self.hps.train.batch_size}")
        print(f"  学习率: {self.hps.train.learning_rate}")
        print(f"  总轮数: {self.hps.train.epochs}")
        print(f"  保存间隔: {self.hps.save_every_epoch}")
    
    def run_visualization(self):
        """运行完整的可视化流程"""
        print("开始RVC模型可视化...")
        
        try:
            # 创建模型
            net_g, net_d = self.create_models()
            
            # 打印模型摘要
            self.print_model_summary(net_g, net_d)
            
            # 可视化数据流
            self.visualize_data_flow(
                os.path.join(self.output_dir, f"data_flow_{self.hps.version}.png")
            )
            
            # 可视化生成器
            self.visualize_generator_forward(
                net_g, 
                os.path.join(self.output_dir, f"generator_{self.hps.version}_f0{self.hps.if_f0}.png")
            )
            
            # 可视化判别器
            self.visualize_discriminator_forward(
                net_d,
                os.path.join(self.output_dir, f"discriminator_{self.hps.version}.png")
            )
            
            # 可视化训练步骤 (可能会很大，可选)
            print("\n是否可视化完整训练步骤? (这可能会生成很大的图片)")
            response = input("输入 y/yes 继续: ").lower()
            if response in ['y', 'yes']:
                self.visualize_training_step(
                    net_g, net_d,
                    os.path.join(self.output_dir, f"training_step_{self.hps.version}.png")
                )
            
            print(f"\n✓ 可视化完成! 文件保存在: {self.output_dir}")
            print("\n生成的文件:")
            for file in sorted(os.listdir(self.output_dir)):
                if file.endswith('.png'):
                    print(f"  - {file}")
                    
        except Exception as e:
            print(f"可视化过程出错: {e}")
            import traceback
            traceback.print_exc()

def load_hparams_from_config(config_path):
    """从配置文件加载超参数"""
    try:
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        class HParams:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, type('obj', (object,), {})())
                        for sub_key, sub_value in value.items():
                            setattr(getattr(self, key), sub_key, sub_value)
                    else:
                        setattr(self, key, value)
        
        hps = HParams(config_data)
        
        # 确保必要的属性存在
        if not hasattr(hps, 'version'):
            hps.version = "v2"
        if not hasattr(hps, 'if_f0'):
            hps.if_f0 = 1
        if not hasattr(hps, 'sample_rate'):
            hps.sample_rate = 48000
        if not hasattr(hps, 'save_every_epoch'):
            hps.save_every_epoch = 5
            
        return hps
        
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        print("使用默认配置...")
        return None

def create_dummy_hparams(version="v2", if_f0=1):
    """创建虚拟的超参数配置"""
    class HParams:
        def __init__(self):
            self.version = version
            self.if_f0 = if_f0
            self.sample_rate = 48000 if version == "v2" else 40000
            self.save_every_epoch = 5
            
            # 数据配置
            self.data = type('obj', (object,), {})()
            self.data.filter_length = 2048
            self.data.hop_length = 512
            self.data.win_length = 2048
            self.data.n_mel_channels = 128
            self.data.sampling_rate = self.sample_rate
            self.data.mel_fmin = 0
            self.data.mel_fmax = None
            self.data.training_files = 'dummy'
            
            # 训练配置
            self.train = type('obj', (object,), {})()
            self.train.batch_size = 4
            self.train.learning_rate = 2e-4
            self.train.epochs = 1000
            self.train.segment_size = 8192
            self.train.c_mel = 45
            self.train.c_kl = 1.0
            self.train.fp16_run = False
            self.train.log_interval = 100
            self.train.seed = 1234
            self.train.betas = [0.8, 0.99]
            self.train.eps = 1e-9
            self.train.lr_decay = 0.999875
            
            # 模型配置
            self.model = type('obj', (object,), {})()
            self.model.inter_channels = 192
            self.model.hidden_channels = 192
            self.model.filter_channels = 768
            self.model.n_heads = 2
            self.model.n_layers = 6
            self.model.kernel_size = 3
            self.model.p_dropout = 0
            self.model.resblock = "1"
            self.model.resblock_kernel_sizes = [3, 7, 11]
            self.model.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            self.model.upsample_rates = [12, 10, 2, 2] if version == "v2" else [10, 10, 2, 2]
            self.model.upsample_initial_channel = 512
            self.model.upsample_kernel_sizes = [24, 20, 4, 4] if version == "v2" else [16, 16, 4, 4]
            # 修复: spk_embed_dim 是说话人数量，gin_channels 是嵌入维度
            self.model.spk_embed_dim = 1  # 单说话人模型
            self.model.gin_channels = 256  # 嵌入维度
            self.model.use_spectral_norm = False
    
    return HParams()

def main():
    parser = argparse.ArgumentParser(description='RVC模型可视化工具')
    parser.add_argument('--config', type=str, help='配置文件路径 (可选)')
    parser.add_argument('--version', type=str, default='v2', choices=['v1', 'v2'], 
                       help='模型版本')
    parser.add_argument('--f0', type=int, default=1, choices=[0, 1],
                       help='是否使用F0 (1=是, 0=否)')
    
    args = parser.parse_args()
    
    print("RVC训练模型可视化工具")
    print("="*50)
    
    # 检查依赖
    try:
        import torchviz
        import graphviz
        print("✓ 依赖检查通过")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请安装: pip install torchviz graphviz")
        return
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        print(f"从配置文件加载: {args.config}")
        hps = load_hparams_from_config(args.config)
        if hps is None:
            hps = create_dummy_hparams(args.version, args.f0)
    else:
        print(f"使用默认配置 - 版本: {args.version}, F0: {args.f0}")
        hps = create_dummy_hparams(args.version, args.f0)
    
    # 创建可视化器
    visualizer = RVCModelVisualizer(hps)
    visualizer.run_visualization()

if __name__ == "__main__":
    main()