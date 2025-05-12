# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from pathlib import Path

import tqdm

# 支持的模型类型及其对应的ModelScope模型ID
MODEL_CONFIGS = {
    "t2v-14B": {
        "model_id": "Wan-AI/Wan2.1-T2V-14B",
        "description": "文本到视频模型(14B)，支持480P和720P分辨率"
    },
    "t2v-1.3B": {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "description": "轻量级文本到视频模型(1.3B)，支持480P分辨率，适合消费级GPU"
    },
    "i2v-14B-720P": {
        "model_id": "Wan-AI/Wan2.1-I2V-14B-720P",
        "description": "图像到视频模型(14B)，支持720P分辨率"
    },
    "i2v-14B-480P": {
        "model_id": "Wan-AI/Wan2.1-I2V-14B-480P",
        "description": "图像到视频模型(14B)，支持480P分辨率"
    },
    "flf2v-14B": {
        "model_id": "Wan-AI/Wan2.1-FLF2V-14B-720P",
        "description": "首帧末帧到视频模型(14B)，支持720P分辨率"
    }
}


def check_modelscope():
    """检查是否已安装modelscope，如果没有则安装"""
    try:
        import modelscope
        return True
    except ImportError:
        print("ModelScope未安装，正在安装...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
            return True
        except Exception as e:
            print(f"安装ModelScope失败: {e}")
            return False


def download_model(model_type, output_dir, resume=True, force=False):
    """从ModelScope下载模型

    Args:
        model_type: 模型类型，必须是MODEL_CONFIGS中的一个键
        output_dir: 模型保存目录
        resume: 是否断点续传
        force: 是否强制重新下载
    """
    if not check_modelscope():
        print("请先安装ModelScope: pip install modelscope")
        return False

    if model_type not in MODEL_CONFIGS:
        print(f"不支持的模型类型: {model_type}")
        print(f"支持的模型类型: {', '.join(MODEL_CONFIGS.keys())}")
        return False

    # 导入modelscope
    from modelscope import snapshot_download

    model_id = MODEL_CONFIGS[model_type]["model_id"]
    description = MODEL_CONFIGS[model_type]["description"]

    print(f"\n开始下载 {model_type} 模型")
    print(f"模型描述: {description}")
    print(f"模型ID: {model_id}")
    print(f"保存目录: {output_dir}\n")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 使用进度条显示下载进度
        print("正在下载模型文件，这可能需要一些时间...")
        model_dir = snapshot_download(
            model_id=model_id,
            cache_dir=output_dir
        )
        print(f"\n模型下载完成！保存在: {model_dir}")
        return True
    except Exception as e:
        print(f"下载模型时出错: {e}")
        return False


def list_models():
    """列出所有可下载的模型"""
    print("\n可下载的万相(Wan2.1)模型列表:")
    print("-" * 80)
    print(f"{'模型类型':<15} | {'模型ID':<40} | {'描述'}")
    print("-" * 80)
    for model_type, config in MODEL_CONFIGS.items():
        print(f"{model_type:<15} | {config['model_id']:<40} | {config['description']}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="从ModelScope下载万相(Wan2.1)大模型")
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=list(MODEL_CONFIGS.keys()),
        help="要下载的模型类型"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./",
        help="模型保存目录"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="列出所有可下载的模型"
    )
    parser.add_argument(
        "--no_resume", 
        action="store_true", 
        help="不使用断点续传"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="强制重新下载"
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not args.model_type:
        list_models()
        print("\n请使用 --model_type 参数指定要下载的模型类型")
        return

    # 下载模型
    download_model(
        model_type=args.model_type,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        force=args.force
    )


if __name__ == "__main__":
    main()