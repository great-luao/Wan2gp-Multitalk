#!/usr/bin/env python3
"""
MultiTalk 精简交互界面
专注于多说话人视频生成功能
"""

import gradio as gr
import torch
import os
import numpy as np
import traceback
from datetime import datetime

# 导入核心组件
from wan.configs import WAN_CONFIGS  # 修正导入路径
from wan.any2video import WanAny2V
from wan.multitalk.multitalk import (
    get_full_audio_embeddings, 
    get_window_audio_embeddings,
    parse_speakers_locations,
    get_target_masks
)

# 全局变量
wan_model = None
current_model_type = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_multitalk_model(model_type="vace_multitalk_14B"):
    """加载 MultiTalk 模型
    
    Args:
        model_type: 模型类型，默认为"vace_multitalk_14B"（与原版wgp.py一致）
    """
    global wan_model, current_model_type
    
    # 根据原版 vace_multitalk_14B.json 配置设置模型文件名
    if model_type == "vace_multitalk_14B":
        # 使用与原版成功配置一致的 FusioniX 量化模型
        model_filename = "Wan14BT2VFusioniX_quanto_bf16_int8.safetensors"
    else:
        return f"❌ 不支持的模型类型: {model_type}"
    
    # 构建完整的模型文件列表并检查是否存在
    vace_module = "wan2.1_Vace_14B_module_quanto_mbf16_int8.safetensors"
    multitalk_module = "wan2.1_multitalk_14B_quanto_mbf16_int8.safetensors"
    fantasy_module = "fantasy_proj_model.safetensors"
    
    # 设置文本编码器文件名（根据实际文件结构）
    text_encoder_file = "ckpts/umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors"
    
    # 检查所有必需文件
    required_files = [
        f"ckpts/{model_filename}",
        f"ckpts/{vace_module}",
        f"ckpts/{multitalk_module}",
        f"ckpts/{fantasy_module}",
        "ckpts/Wan2.1_VAE.safetensors",
        text_encoder_file
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        return "❌ 以下模型文件不存在:\n" + "\n".join(missing_files) + "\n\n请确保已下载所有模型文件到ckpts目录"
    
    try:
        print(f"正在加载模型: {model_filename}")
        
        # ===== 调试代码：检查基本信息 =====
        print("🔍 DEBUG: multitalk load_multitalk_model")
        print(f"  - model_type = {model_type}")
        print(f"  - current working directory = {os.getcwd()}")
        print(f"  - model_filename = {model_filename}")
        print(f"  - vace_module = {vace_module}")
        print(f"  - multitalk_module = {multitalk_module}")
        print(f"  - fantasy_module = {fantasy_module}")
        print(f"  - text_encoder_file = {text_encoder_file}")
        
        # 使用 i2v 配置，因为 multitalk 需要 i2v 模式
        cfg = WAN_CONFIGS['i2v-14B']
        current_model_type = model_type
        
        # 根据原版配置设置数据类型（模型文件已经是量化版本）
        dtype = torch.bfloat16
        VAE_dtype = torch.float32
        quantizeTransformer = False  # 模型文件已经量化，不需要再次量化
        
        # 使用之前已经设置好的文本编码器文件路径（量化版本）
        
        # 创建模型定义，与原版 vace_multitalk_14B.json 一致
        temp_model_def = {
            "name": "Vace Multitalk FusioniX 14B",
            "architecture": "vace_multitalk_14B", 
            "modules": ["vace_14B", "multitalk"],
            "auto_quantize": True
        }
        
        # 按照原版逻辑：主模型 + 模块列表
        complete_model_list = [
            model_filename,     # 主模型
            vace_module,        # Vace 14B 模块
            multitalk_module,   # MultiTalk 模块  
            fantasy_module      # Fantasy 模块
        ]
        
        # 修复模型文件路径 - 添加checkpoint_dir前缀
        complete_model_list_with_path = []
        for filename in complete_model_list:
            if not filename.startswith("ckpts/"):
                complete_model_list_with_path.append(f"ckpts/{filename}")
            else:
                complete_model_list_with_path.append(filename)

        wan_model = WanAny2V(
            config=cfg,
            checkpoint_dir="ckpts",
            model_filename=complete_model_list_with_path,  # 使用带路径的模型文件列表
            model_type=model_type,
            model_def=temp_model_def,  # 添加模型定义
            base_model_type="vace_multitalk_14B",
            text_encoder_filename=text_encoder_file,  # 使用正确的文本编码器文件名
            quantizeTransformer=quantizeTransformer,  # 根据模式决定是否量化
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            mixed_precision_transformer=False  # 高 VRAM 模式可以禁用混合精度
        )
        
        return "✅ MultiTalk 模型加载成功"
    except Exception as e:
        return f"❌ 模型加载失败: {str(e)}"

def generate_multitalk_video(
    # 基础参数
    prompt,
    negative_prompt,
    # 生成模式
    generation_mode,
    # 图像输入（图生视频模式）
    input_image,
    # 音频参数
    audio_file1,
    audio_file2,
    speakers_locations,
    audio_combination_type,
    # 视频参数
    resolution,
    video_length,
    fps,
    seed,
    num_inference_steps,
    guidance_scale,
    # 高级参数
    flow_shift,
    embedded_guidance_scale,
    audio_guidance_scale
):
    """生成 MultiTalk 视频"""
    global wan_model
    
    if wan_model is None:
        return None, "请先加载模型"
    
    try:
        # 解析分辨率
        width, height = map(int, resolution.split('x'))
        
        # 设置种子
        if seed == -1:
            seed = np.random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        
        # 处理音频输入
        temp_dir = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 使用上传的音频文件
        audio_path1 = audio_file1.name if audio_file1 else None
        audio_path2 = audio_file2.name if audio_file2 else None
        
        # 确定最终要合并的音频文件
        sum_audio_path = audio_path1  # 优先使用第一个音频文件
        
        # 验证音频输入
        if not audio_path1:
            return None, "请至少上传一个音频文件"
        
        # 处理图像输入（如果是图生视频模式）
        image_guide_path = None
        if generation_mode == "图生视频":
            if input_image is None:
                return None, "图生视频模式需要上传图像"
            # 保存上传的图像
            image_guide_path = os.path.join(temp_dir, "input_image.png")
            if hasattr(input_image, 'name'):
                # 如果是文件对象
                import shutil
                shutil.copy(input_image.name, image_guide_path)
            else:
                # 如果是PIL图像
                input_image.save(image_guide_path)
        
        # 解析说话人位置
        speakers_bboxes, error = parse_speakers_locations(speakers_locations)
        if error:
            return None, f"说话人位置解析错误: {error}"
        
        # 获取音频嵌入
        full_audio_embs, sum_human_speechs = get_full_audio_embeddings(
            audio_guide1=audio_path1,
            audio_guide2=audio_path2,
            combination_type=audio_combination_type,
            num_frames=video_length,
            fps=fps,
            sr=16000
        )
        
        # 获取窗口音频嵌入
        audio_proj = get_window_audio_embeddings(
            full_audio_embs,
            audio_start_idx=0,
            clip_length=video_length,
            vae_scale=4,
            audio_window=5
        )
        
        # 生成目标掩码
        lat_h, lat_w = height // 8, width // 8
        token_ref_target_masks = get_target_masks(
            HUMAN_NUMBER=2 if audio_path2 else 1,
            lat_h=lat_h,
            lat_w=lat_w,
            src_h=height,
            src_w=width,
            bbox=speakers_bboxes
        )
        
        # 根据设备 VRAM 自动选择 VAE tile size 和 joint pass
        device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
        if device_mem_capacity >= 24000:  # 24GB+
            VAE_tile_size = 0  # 不使用瓦片化，最快
            joint_pass = True
        elif device_mem_capacity >= 12000:  # 12GB+
            VAE_tile_size = 256
            joint_pass = True
        else:
            VAE_tile_size = 128
            joint_pass = False
        
        # 调用模型生成视频
        print(f"开始生成视频: {width}x{height}, {video_length}帧")
        print(f"设备 VRAM: {device_mem_capacity:.0f}MB, VAE Tile Size: {VAE_tile_size}")
        
        # 重置中断标志
        wan_model._interrupt = False
        
        # 创建回调函数用于显示进度
        def progress_callback(step, total_steps, latents):
            print(f"进度: {step}/{total_steps}")
        
        samples = wan_model.generate(
            input_prompt=prompt,
            n_prompt=negative_prompt,
            width=width,
            height=height,
            frame_num=video_length,
            batch_size=1,
            seed=seed,
            sampling_steps=num_inference_steps,
            guide_scale=guidance_scale,
            shift=flow_shift,
            guide2_scale=embedded_guidance_scale,
            audio_cfg_scale=audio_guidance_scale,
            audio_proj=audio_proj,
            speakers_bboxes=speakers_bboxes,
            token_ref_target_masks=token_ref_target_masks,  # 添加正确的参数名
            input_video=image_guide_path if generation_mode == "图生视频" else None,
            model_type="multitalk",
            sample_solver="unipc",
            VAE_tile_size=VAE_tile_size,
            joint_pass=joint_pass,
            callback=progress_callback if num_inference_steps > 10 else None
        )
        
        # 保存视频
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"multitalk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        
        # samples已经是解码后的视频张量，形状为 [C, T, H, W] 或 [T, C, H, W]
        if isinstance(samples, torch.Tensor):
            video_frames = samples
        else:
            # 如果返回的是列表，取第一个
            video_frames = samples[0] if isinstance(samples, list) else samples
        
        # 使用 imageio 保存视频
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
        
        # 转换张量格式：[C, T, H, W] -> [T, H, W, C]
        if len(video_frames.shape) == 4:
            if video_frames.shape[0] == 3:  # [C, T, H, W]
                video_frames = video_frames.permute(1, 2, 3, 0)  # -> [T, H, W, C]
            elif video_frames.shape[1] == 3:  # [T, C, H, W]
                video_frames = video_frames.permute(0, 2, 3, 1)  # -> [T, H, W, C]
        
        # 转换为numpy并标准化到0-255
        video_frames = video_frames.cpu().numpy()
        if video_frames.dtype != np.uint8:
            # 从[-1,1]范围转换到[0,255]
            video_frames = ((video_frames + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        for frame in video_frames:
            writer.append_data(frame)
        
        writer.close()
        
        # 如果有音频，合并音频和视频
        if sum_audio_path and os.path.exists(sum_audio_path):
            final_output = output_path.replace('.mp4', '_with_audio.mp4')
            import subprocess
            cmd = [
                'ffmpeg', '-y', '-i', output_path, '-i', sum_audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
                final_output
            ]
            subprocess.run(cmd, check=True)
            output_path = final_output
        
        # 清理临时文件
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        
        return output_path, f"✅ 视频生成成功！种子: {seed}"
        
    except Exception as e:
        error_msg = f"❌ 生成失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="MultiTalk 精简版", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎭 MultiTalk 精简版
        
        专注于多说话人视频生成的精简交互界面
        """)
        
        # 状态信息
        status_text = gr.Textbox(label="状态", value="请先加载模型", interactive=False)
        
        with gr.Row():
            # 左侧：输入参数
            with gr.Column(scale=1):
                # 模型加载
                with gr.Group():
                    gr.Markdown("### 📦 模型设置")
                    model_type_choice = gr.Dropdown(
                        choices=["vace_multitalk_14B"],
                        value="vace_multitalk_14B",
                        label="模型类型"
                    )
                    load_btn = gr.Button("加载模型", variant="primary")
                    
                    # 显示当前将使用的模型文件
                    model_info = gr.Textbox(
                        label="模型信息",
                        value="主模型: Wan14BT2VFusioniX_quanto_bf16_int8.safetensors\nVace模块: wan2.1_Vace_14B_module_quanto_mbf16_int8.safetensors\nMultiTalk模块: wan2.1_multitalk_14B_quanto_mbf16_int8.safetensors\nFantasy模块: fantasy_proj_model.safetensors\nVAE: Wan2.1_VAE.safetensors\n文本编码器: umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors\n模式: Vace+MultiTalk 量化模式",
                        interactive=False,
                        lines=7
                    )
                
                # 基础参数
                with gr.Group():
                    gr.Markdown("### 🎬 基础参数")
                    generation_mode = gr.Radio(
                        choices=["文生视频", "图生视频"],
                        value="文生视频",
                        label="生成模式"
                    )
                    prompt = gr.Textbox(
                        label="提示词",
                        placeholder="描述你想要生成的场景...",
                        lines=3
                    )
                    negative_prompt = gr.Textbox(
                        label="负面提示词",
                        placeholder="不想要出现的内容...",
                        lines=2
                    )
                    
                    # 图像输入（图生视频模式）
                    with gr.Group(visible=False) as image_input_group:
                        gr.Markdown("上传一张图像作为视频生成的起始帧")
                        input_image = gr.Image(
                            label="输入图像",
                            type="pil"
                        )
                
                # 音频设置
                with gr.Group():
                    gr.Markdown("### 🎤 音频设置")
                    
                    audio_file1 = gr.File(label="说话人1音频", file_types=["audio"])
                    audio_file2 = gr.File(label="说话人2音频（可选）", file_types=["audio"])
                    
                    speakers_locations = gr.Textbox(
                        label="说话人位置 (示例: '25:75' 表示单人位置，'25:75 40:80' 表示两人位置)",
                        value="25:75",
                        placeholder="单人: '25:75' 或 两人: '25:75 40:80' (左:右百分比)"
                    )
                    
                    audio_combination_type = gr.Radio(
                        choices=["add", "para"],
                        value="add",
                        label="音频组合方式 (add: 顺序播放, para: 并行播放)"
                    )
                
                # 视频参数
                with gr.Group():
                    gr.Markdown("### 🎥 视频参数")
                    resolution = gr.Dropdown(
                        choices=["1280x720", "1024x576", "768x432", "512x288"],
                        value="768x432",
                        label="分辨率"
                    )
                    video_length = gr.Slider(
                        minimum=25,
                        maximum=129,
                        value=49,
                        step=8,
                        label="视频长度（帧）"
                    )
                    fps = gr.Slider(
                        minimum=8,
                        maximum=30,
                        value=25,
                        step=1,
                        label="帧率 (FPS)"
                    )
                
                # 生成参数
                with gr.Accordion("高级参数", open=False):
                    seed = gr.Number(value=-1, label="种子（-1为随机）", precision=0)
                    num_inference_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=30,
                        step=1,
                        label="推理步数"
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.0,
                        step=0.5,
                        label="引导强度"
                    )
                    flow_shift = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=5.0,
                        step=0.5,
                        label="Flow Shift"
                    )
                    embedded_guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=5.0,
                        step=0.5,
                        label="嵌入引导强度"
                    )
                    audio_guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.5,
                        label="音频引导强度"
                    )
                    
                    # VRAM 优化提示
                    gr.Markdown("""
                    #### 💡 性能优化建议
                    - **8GB VRAM**: 使用 512x288 分辨率，49 帧
                    - **12GB VRAM**: 使用 768x432 分辨率，81 帧
                    - **24GB+ VRAM**: 启用高 VRAM 模式，使用 1280x720，129 帧
                    """)
            
            # 右侧：输出区域
            with gr.Column(scale=1):
                gr.Markdown("### 📹 输出")
                output_video = gr.Video(label="生成的视频")
                generate_btn = gr.Button("🚀 生成视频", variant="primary", size="lg")
                
                # 示例
                gr.Markdown("### 💡 使用示例")
                gr.Examples(
                    examples=[
                        [
                            "文生视频",
                            "Two people having a conversation in a modern office",
                            "blurry, low quality",
                            "25:75",
                            "add"
                        ],
                        [
                            "图生视频",
                            "Interview scene with two speakers at a news desk",
                            "cartoon, animated",
                            "30:70",
                            "add"
                        ]
                    ],
                    inputs=[generation_mode, prompt, negative_prompt, speakers_locations, audio_combination_type]
                )
        
        # 事件处理
        def toggle_generation_mode(mode):
            return gr.update(visible=(mode == "图生视频"))
        
        generation_mode.change(
            toggle_generation_mode,
            inputs=[generation_mode],
            outputs=[image_input_group]
        )
        
        load_btn.click(
            load_multitalk_model,
            inputs=[model_type_choice],
            outputs=[status_text]
        )
        
        generate_btn.click(
            generate_multitalk_video,
            inputs=[
                prompt, negative_prompt,
                generation_mode, input_image,
                audio_file1, audio_file2,
                speakers_locations, audio_combination_type,
                resolution, video_length, fps,
                seed, num_inference_steps, guidance_scale,
                flow_shift, embedded_guidance_scale, audio_guidance_scale
            ],
            outputs=[output_video, status_text]
        )
    
    return app

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MultiTalk 精简版")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--server-port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="是否创建公共链接")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()
    
    # 创建并启动应用
    app = create_ui()
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=not args.no_browser
    )