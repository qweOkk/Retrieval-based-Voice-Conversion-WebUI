#!/usr/bin/env python3
import os
import gc
import platform
import traceback
import argparse
import sys
import logging
import numpy as np
import faiss
from faiss import index_factory, extract_index_ivf
from sklearn.cluster import MiniBatchKMeans

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置类（如果没有单独的config模块）
class Config:
    def __init__(self):
        # 默认CPU核心数，可以根据需要调整
        self.n_cpu = os.cpu_count() or 4

config = Config()

# 外部索引根目录（可以根据需要修改）
outside_index_root = "assets/indices"

def train_index(exp_dir1, version19):
    """
    训练FAISS索引
    
    Args:
        exp_dir1: 实验目录名称
        version19: 版本标识 ("v1" 或 "v2")
    """
    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    
    print(f"开始处理特征目录: {feature_dir}")
    print(f"找到 {len(listdir_res)} 个特征文件")
    
    npys = []
    for name in sorted(listdir_res):
        if name.endswith('.npy'):
            print(f"加载特征文件: {name}")
            phone = np.load("%s/%s" % (feature_dir, name))
            npys.append(phone)
    
    if len(npys) == 0:
        return "特征目录中没有找到.npy文件！"
    
    big_npy = np.concatenate(npys, 0)
    print(f"合并后的特征形状: {big_npy.shape}")
    
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    
    if big_npy.shape[0] > 2e5:
        print("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
            print(f"K-means聚类后的形状: {big_npy.shape}")
        except:
            info = traceback.format_exc()
            logger.info(info)
            print(info)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    print(f"保存总特征文件: {exp_dir}/total_fea.npy")
    
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    print("%s,%s" % (big_npy.shape, n_ivf))
    
    feature_dim = 256 if version19 == "v1" else 768
    index = faiss.index_factory(feature_dim, "IVF%s,Flat" % n_ivf)
    
    print("开始训练索引...")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    
    trained_index_path = "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19)
    faiss.write_index(index, trained_index_path)
    print("成功构建索引 trained_IVF%s_Flat_nprobe_%s_%s_%s.index" % (n_ivf, index_ivf.nprobe, exp_dir1, version19))
    
    print("开始添加数据到索引...")
    
    # 分小批次添加数据
    batch_size = 8192  # 使用很小的批次
    total_batches = (big_npy.shape[0] + batch_size - 1) // batch_size
    
    print(f"开始添加数据，批次大小={batch_size}, 总批次={total_batches}")
    
    for i in range(0, big_npy.shape[0], batch_size):
        batch_end = min(i + batch_size, big_npy.shape[0])
        batch_data = big_npy[i:batch_end].copy()
        
        if not batch_data.flags['C_CONTIGUOUS']:
            batch_data = np.ascontiguousarray(batch_data)
            
        index.add(batch_data)
        
        batch_num = i // batch_size + 1
        if batch_num % 20 == 0 or batch_end == big_npy.shape[0]:
            progress = batch_end / big_npy.shape[0] * 100
            print(f"进度: {progress:.1f}% ({batch_num}/{total_batches})")
            gc.collect()  # 清理内存
    
    added_index_path = "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19)
    faiss.write_index(index, added_index_path)
    print("成功构建索引 added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (n_ivf, index_ivf.nprobe, exp_dir1, version19))
    
    # 创建链接到外部索引目录
    try:
        os.makedirs(outside_index_root, exist_ok=True)
        link = os.link if platform.system() == "Windows" else os.symlink
        
        external_index_path = "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
            outside_index_root,
            exp_dir1,
            n_ivf,
            index_ivf.nprobe,
            exp_dir1,
            version19,
        )
        
        # 如果目标文件已存在，先删除
        if os.path.exists(external_index_path):
            os.remove(external_index_path)
            
        link(added_index_path, external_index_path)
        print("链接索引到外部-%s" % (outside_index_root))
    except Exception as e:
        print("链接索引到外部-%s失败: %s" % (outside_index_root, str(e)))

    print("索引训练完成!")
    return "训练成功完成!"

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练RVC FAISS索引")
    
    parser.add_argument("--exp-dir", type=str, help="实验目录名称（在logs/下）")
    parser.add_argument("--version", "-v", type=str, choices=["v1", "v2"], default="v2",
                        help="模型版本 (v1: 256维特征, v2: 768维特征)")
    parser.add_argument("--n-cpu", type=int, default=None,
                        help="CPU核心数 (默认: 自动检测)")
    parser.add_argument("--outside-index-root", type=str, default="assets/indices",
                        help="外部索引根目录 (默认: assets/indices)")
    parser.add_argument("--batch-size", type=int, default=8192,
                        help="批处理大小 (默认: 8192)")
    parser.add_argument("--verbose", "-V", action="store_true",
                        help="显示详细信息")
    
    return parser.parse_args()

def check_requirements():
    """检查必要的目录和文件"""
    try:
        import faiss
        import sklearn
        print("✓ 依赖检查通过")
        return True
    except ImportError as e:
        print(f"✗ 缺少必要的依赖: {e}")
        print("请安装: pip install faiss-cpu scikit-learn numpy")
        return False

def main():
    """主函数"""
    print("=== RVC FAISS索引训练工具 ===")
    
    # 检查依赖
    if not check_requirements():
        sys.exit(1)
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置全局变量
    global outside_index_root
    outside_index_root = args.outside_index_root
    
    if args.n_cpu:
        config.n_cpu = args.n_cpu
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 显示配置信息
    print(f"实验目录: {args.exp_dir}")
    print(f"模型版本: {args.version}")
    print(f"CPU核心数: {config.n_cpu}")
    print(f"外部索引目录: {outside_index_root}")
    print(f"批处理大小: {args.batch_size}")
    print("-" * 40)
    
    # 检查实验目录
    exp_dir = f"logs/{args.exp_dir}"
    if not os.path.exists(exp_dir):
        print(f"错误: 实验目录不存在: {exp_dir}")
        print("请确保已经完成模型训练和特征提取")
        sys.exit(1)
    
    # 检查特征目录
    feature_dir = f"{exp_dir}/3_feature256" if args.version == "v1" else f"{exp_dir}/3_feature768"
    if not os.path.exists(feature_dir):
        print(f"错误: 特征目录不存在: {feature_dir}")
        print("请先进行特征提取!")
        sys.exit(1)
    
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
    if len(feature_files) == 0:
        print(f"错误: 特征目录中没有.npy文件: {feature_dir}")
        print("请先进行特征提取!")
        sys.exit(1)
    
    print(f"找到 {len(feature_files)} 个特征文件")
    
    try:
        # 开始训练索引
        result = train_index(args.exp_dir, args.version)
        
        if "成功" in result:
            print("\n=== 训练完成 ===")
            print(result)
            
            # 显示生成的文件
            print("\n生成的文件:")
            for file in os.listdir(exp_dir):
                if file.endswith('.index') or file == 'total_fea.npy':
                    file_path = os.path.join(exp_dir, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    print(f"  {file} ({file_size:.1f} MB)")
        else:
            print(f"训练失败: {result}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断训练")
        sys.exit(1)
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

