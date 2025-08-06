#!/usr/bin/env python3
"""
MultiTalk 测试脚本（无UI版本）
用于测试多说话人视频生成功能
"""

import torch
import os
import traceback
from datetime import datetime

# 导入核心组件
from wan.configs import WAN_CONFIGS
from wan.any2video import WanAny2V
from wan.multitalk.multitalk import (
    get_full_audio_embeddings, 
    get_window_audio_embeddings,
    parse_speakers_locations,
    get_target_masks
)

# 内置默认配置 - 原版 MultiTalk 480p
CONFIG = {
    # 模型配置
    "model_type": "multitalk",  # 使用原版multitalk而不是vace版本
    "checkpoint_dir": "ckpts",
    
    # 音频输入 - One Person Speaking Only 模式
    "audio_file1": "resources/speaker1.mp3",  # 说话人音频文件
    "audio_file2": None,  # 单人模式不需要第二个音频
    "audio_combination_type": "para",  # 音频组合方式（单人模式下无影响）
    
    # 说话人位置（百分比）
    "speakers_locations": "25:75",  # 单人说话位置（中心位置）
    
    # 生成参数
    "prompt": "Two soldiers speaking at the battleground.",  # 匹配原版测试
    "negative_prompt": "",  # 原版使用空字符串
    "width": 832,  # 原版multitalk分辨率
    "height": 480,  # 原版multitalk分辨率
    "video_length": 129,  # 原版multitalk帧数
    "fps": 25,
    "seed": 5730212,  # 匹配原版测试种子
    "num_inference_steps": 20,  # 原版使用60步
    "guidance_scale": 5.0,  # 原版使用5
    "flow_shift": 7.0,  # 原版使用7
    "embedded_guidance_scale": 6.0,  # 原版使用6
    "audio_guidance_scale": 4.0,  # 原版使用4
    
    # 高级参数
    "VAE_tile_size": None,  # 自动调整
    "joint_pass": None,     # 自动调整
}

# 全局变量
wan_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_audio_from_video(video_path, sr=16000):
    """从视频中提取音频（简单实现）"""
    import librosa
    try:
        # 使用librosa直接从视频提取音频
        audio, _ = librosa.load(video_path, sr=sr)
        return audio
    except Exception as e:
        print(f"警告：无法从视频提取音频，尝试使用ffmpeg: {e}")
        # 如果librosa失败，可以尝试ffmpeg
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            cmd = ['ffmpeg', '-i', video_path, '-ar', str(sr), '-ac', '1', tmp_file.name, '-y']
            subprocess.run(cmd, check=True, capture_output=True)
            audio, _ = librosa.load(tmp_file.name, sr=sr)
            os.unlink(tmp_file.name)
            return audio

def load_multitalk_model():
    """加载原版 MultiTalk 480p 模型"""
    global wan_model
    
    print("正在加载原版 MultiTalk 480p 模型...")
    
    # 原版multitalk使用的模型文件（单个主模型文件）
    model_files = {
        "主模型": "wan2.1_image2video_480p_14B_quanto_mbf16_int8.safetensors",  # 原版multitalk模型文件
        "VAE": "Wan2.1_VAE.safetensors",
    }
    
    text_encoder_file = "ckpts/umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors"
    
    # 检查文件是否存在
    missing_files = []
    for name, filename in model_files.items():
        file_path = f"ckpts/{filename}"
        if not os.path.exists(file_path):
            missing_files.append(f"{name}: {file_path}")
    
    if not os.path.exists(text_encoder_file):
        missing_files.append(f"文本编码器: {text_encoder_file}")
    
    if missing_files:
        print("❌ 以下模型文件不存在:")
        for missing in missing_files:
            print(f"  - {missing}")
        return False
    
    try:
        # 使用 i2v 配置但需要添加 multitalk_output_dim
        cfg = WAN_CONFIGS['i2v-14B']
        
        # 模型文件（原版multitalk只需要一个主模型文件）
        model_filename = f"ckpts/{model_files['主模型']}"
        
        # 模型定义 - 原版multitalk配置
        temp_model_def = {
            "name": "MultiTalk 480p",
            "architecture": "multitalk", 
            "modules": [],  # 原版multitalk没有额外模块
            "multitalk_output_dim": 768,  # 这个参数启用multitalk功能
            "auto_quantize": True
        }
        
        # 创建模型
        wan_model = WanAny2V(
            config=cfg,
            checkpoint_dir="ckpts",
            model_filename=[model_filename],  # 单个模型文件
            model_type=CONFIG["model_type"],
            model_def=temp_model_def,
            base_model_type="multitalk",  # 使用原版multitalk
            text_encoder_filename=text_encoder_file,
            quantizeTransformer=False,  # 模型已经量化
            dtype=torch.bfloat16,
            VAE_dtype=torch.float32,
            mixed_precision_transformer=False
        )
        
        print("✅ 原版 MultiTalk 480p 模型加载成功")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        print(traceback.format_exc())
        return False

def generate_video():
    """生成 MultiTalk 视频"""
    global wan_model
    
    if wan_model is None:
        print("❌ 请先加载模型")
        return None
    
    print("开始生成视频...")
    
    try:
        # 检查音频文件
        audio_path1 = CONFIG["audio_file1"] if CONFIG["audio_file1"] and os.path.exists(CONFIG["audio_file1"]) else None
        audio_path2 = CONFIG["audio_file2"] if CONFIG["audio_file2"] and os.path.exists(CONFIG["audio_file2"]) else None
        
        if audio_path1 is None:
            print(f"❌ 音频文件1不存在: {CONFIG['audio_file1']}")
            return None
        
        print(f"✅ 音频文件1: {audio_path1}")
        if audio_path2:
            print(f"✅ 音频文件2: {audio_path2}")
        else:
            print("ℹ️  只使用一个音频文件")
        
        # 设置种子
        torch.manual_seed(CONFIG["seed"])
        
        # 解析说话人位置
        speakers_bboxes, error = parse_speakers_locations(CONFIG["speakers_locations"])
        if error:
            print(f"❌ 说话人位置解析错误: {error}")
            return None
        
        print(f"✅ 说话人位置: {speakers_bboxes}")
        
        # 获取音频嵌入
        print("正在处理音频...")
        full_audio_embs, sum_human_speechs = get_full_audio_embeddings(
            audio_guide1=audio_path1,
            audio_guide2=audio_path2,
            combination_type=CONFIG["audio_combination_type"],
            num_frames=CONFIG["video_length"],
            fps=CONFIG["fps"],
            sr=16000
        )
        
        print(f"✅ 音频嵌入完成，说话人数量: {len(full_audio_embs)}")
        
        # 获取窗口音频嵌入
        audio_proj = get_window_audio_embeddings(
            full_audio_embs,
            audio_start_idx=0,
            clip_length=CONFIG["video_length"],
            vae_scale=4,
            audio_window=5
        )
        
        # 生成目标掩码
        lat_h, lat_w = CONFIG["height"] // 8, CONFIG["width"] // 8
        token_ref_target_masks = get_target_masks(
            HUMAN_NUMBER=len(full_audio_embs),
            lat_h=lat_h,
            lat_w=lat_w,
            src_h=CONFIG["height"],
            src_w=CONFIG["width"],
            bbox=speakers_bboxes
        )
        
        print(f"✅ 目标掩码生成完成: {token_ref_target_masks.shape}")
        
        # 根据VRAM调整参数
        device_mem = torch.cuda.get_device_properties(0).total_memory / 1048576
        print(f"✅ 检测到GPU内存: {device_mem:.0f}MB")
        
        # 使用配置文件设置或自动调整
        if CONFIG.get("VAE_tile_size") is not None:
            vae_tile_size = CONFIG["VAE_tile_size"]
            print(f"✅ 使用配置的VAE tile size: {vae_tile_size}")
        else:
            if device_mem >= 24000:
                vae_tile_size = 0
                print("✅ 自动设置: 高VRAM模式")
            elif device_mem >= 12000:
                vae_tile_size = 256
                print("✅ 自动设置: 中VRAM模式")
            else:
                vae_tile_size = 128
                print("✅ 自动设置: 低VRAM模式")
        
        if CONFIG.get("joint_pass") is not None:
            joint_pass = CONFIG["joint_pass"]
            print(f"✅ 使用配置的joint_pass: {joint_pass}")
        else:
            joint_pass = device_mem >= 12000
            print(f"✅ 自动设置joint_pass: {joint_pass}")
        
        # 创建默认起始图像（MultiTalk需要起始图像）
        print("正在创建起始图像...")
        from PIL import Image
        import numpy as np
        
        # 创建一个简单的黑色起始图像
        start_image = Image.new('RGB', (CONFIG["width"], CONFIG["height"]), color=(0, 0, 0))
        
        # 按照 wgp.py 的方式转换为张量格式 [C, H, W]，范围 [-1, 1]
        image_start = torch.from_numpy(np.array(start_image).astype(np.float32)).div_(127.5).sub_(1.).movedim(-1, 0)
        # image_start = image_start.to(torch.bfloat16)  # 转换为 bfloat16 匹配模型
        print(f"✅ 起始图像张量已创建: {image_start.shape}, dtype: {image_start.dtype}")
        
        # 创建临时目录用于后续清理
        temp_dir = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 重置中断标志
        wan_model._interrupt = False
        
        # 进度回调
        def progress_callback(step, total_steps, latents):
            if step >= 0:
                print(f"  进度: {step+1}/{total_steps}")
        
        # 生成视频
        print(f"开始生成视频: {CONFIG['width']}x{CONFIG['height']}, {CONFIG['video_length']}帧")
        
        # 创建 pre_video_guide (原版multitalk需要此参数)
        pre_video_guide = image_start.unsqueeze(1)  # 从 [3, H, W] 转为 [3, 1, H, W]
        print(f"✅ pre_video_guide 已创建: {pre_video_guide.shape}")
        
        # 按照wgp.py中的方式调用原版multitalk
        samples = wan_model.generate(
            input_prompt=CONFIG["prompt"],
            n_prompt=CONFIG["negative_prompt"],
            image_start=image_start,  # 起始图像张量
            pre_video_guide=pre_video_guide,  # 原版必需参数
            width=CONFIG["width"],
            height=CONFIG["height"],
            frame_num=CONFIG["video_length"],
            batch_size=1,
            seed=CONFIG["seed"],
            denoising_strength=1.0,  # 原版设置
            shift=CONFIG["flow_shift"],
            sample_solver="euler",  # 原版使用euler
            sampling_steps=CONFIG["num_inference_steps"],
            guide_scale=CONFIG["guidance_scale"],
            guide2_scale=CONFIG["embedded_guidance_scale"],
            embedded_guidance_scale=CONFIG["embedded_guidance_scale"],
            audio_cfg_scale=CONFIG["audio_guidance_scale"],
            audio_proj=audio_proj,
            speakers_bboxes=speakers_bboxes,
            # 原版multitalk的关键参数
            model_type="multitalk",
            model_mode=None,
            prefix_frames_count=1,  # 原版使用1
            VAE_tile_size=vae_tile_size,
            joint_pass=joint_pass,
            apg_switch=0,
            cfg_star_switch=0,  # 原版使用0
            cfg_zero_step=-1,  # 原版使用-1
            NAG_scale=1,  # 原版使用1
            NAG_tau=3.5,
            NAG_alpha=0.5,
            enable_RIFLEx=False,  # 原版设置
            slg_start_perc=10,  # 原版设置
            slg_end_perc=90,  # 原版设置
            callback=progress_callback if CONFIG["num_inference_steps"] > 10 else None
        )
        
        # 保存视频
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f"multitalk_{timestamp}.mp4")
        
        print("正在保存视频...")
        
        # 处理视频张量
        if isinstance(samples, torch.Tensor):
            video_frames = samples
        else:
            video_frames = samples[0] if isinstance(samples, list) else samples
        
        # 使用 imageio 保存视频
        import imageio
        writer = imageio.get_writer(output_path, fps=CONFIG["fps"], codec='libx264', quality=8)
        
        # 转换张量格式
        if len(video_frames.shape) == 4:
            if video_frames.shape[0] == 3:  # [C, T, H, W]
                video_frames = video_frames.permute(1, 2, 3, 0)  # -> [T, H, W, C]
            elif video_frames.shape[1] == 3:  # [T, C, H, W]
                video_frames = video_frames.permute(0, 2, 3, 1)  # -> [T, H, W, C]
        
        # 转换为numpy并标准化
        video_frames = video_frames.cpu().numpy()
        if video_frames.dtype != np.uint8:
            video_frames = ((video_frames + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        for frame in video_frames:
            writer.append_data(frame)
        writer.close()
        
        # 合并音频
        if sum_human_speechs is not None:
            print("正在合并音频...")
            # 保存合成音频
            import soundfile as sf
            audio_path = output_path.replace('.mp4', '_audio.wav')
            sf.write(audio_path, sum_human_speechs, 16000)
            
            # 合并视频和音频
            final_output = output_path.replace('.mp4', '_with_audio.mp4')
            import subprocess
            cmd = [
                'ffmpeg', '-y', '-i', output_path, '-i', audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
                final_output
            ]
            subprocess.run(cmd, check=True)
            output_path = final_output
            
            # 清理临时音频文件
            os.remove(audio_path)
        
        print(f"✅ 视频生成完成: {output_path}")
        print(f"✅ 种子: {CONFIG['seed']}")
        
        return output_path
        
    except Exception as e:
        error_msg = f"❌ 生成失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None

def main():
    """主函数"""
    # 应用预设配置
    global CONFIG
    
    print("🎭 MultiTalk 测试脚本")
    print("=" * 50)
    
    # 显示配置
    print("当前配置:")
    for key, value in CONFIG.items():
        if key not in ["negative_prompt"]:  # 跳过长文本
            print(f"  {key}: {value}")
    print("-" * 40)
    
    # 检查设备
    print(f"设备: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    print("=" * 50)
    
    # 加载模型
    if not load_multitalk_model():
        return
    
    print("=" * 50)
    
    # 生成视频
    output_path = generate_video()
    
    if output_path:
        print("=" * 50)
        print(f"🎉 成功！输出文件: {output_path}")
    else:
        print("=" * 50)
        print("❌ 生成失败")

if __name__ == "__main__":
    main()