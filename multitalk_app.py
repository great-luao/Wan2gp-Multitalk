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
    get_target_masks,
    process_tts_single,
    process_tts_multi
)

# 检查是否需要导入 Kokoro（如果使用 TTS）
try:
    from wan.multitalk.kokoro import KPipeline
except ImportError:
    print("警告: Kokoro TTS 模块未找到，TTS 功能可能不可用")
    KPipeline = None

# 全局变量
wan_model = None
current_model_type = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_multitalk_model(model_path="ckpts/multitalk-wan2gp-14B.pth", high_vram_mode=False):
    """加载 MultiTalk 模型
    
    Args:
        model_path: 模型文件路径
        high_vram_mode: 是否使用高 VRAM 模式（优化速度和质量）
    """
    global wan_model, current_model_type
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        return f"❌ 模型文件不存在: {model_path}"
    
    try:
        print(f"正在加载模型: {model_path}")
        print(f"高 VRAM 模式: {'开启' if high_vram_mode else '关闭'}")
        # 使用 i2v 配置，因为 multitalk 需要 i2v 模式
        cfg = WAN_CONFIGS['i2v-14B']
        current_model_type = "multitalk"
        
        # 根据 VRAM 模式选择数据类型
        if high_vram_mode:
            # 高 VRAM 模式：使用更高精度，不量化
            dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
            VAE_dtype = torch.float32
            quantize = False
        else:
            # 标准模式：使用量化以节省 VRAM
            dtype = torch.bfloat16
            VAE_dtype = torch.float32
            quantize = True
        
        # 初始化模型
        # 如果 model_path 是单个文件，需要转换为列表
        if isinstance(model_path, str):
            model_filename = [model_path]
        else:
            model_filename = model_path
            
        wan_model = WanAny2V(
            config=cfg,
            checkpoint_dir="ckpts",
            model_filename=model_filename,  # 需要是列表格式
            model_type="multitalk",
            base_model_type="wan_i2v_14B",
            text_encoder_filename=None,  # 使用默认
            quantizeTransformer=quantize,  # 根据模式决定是否量化
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
    # 音频参数
    audio_source_type,
    audio_file1,
    audio_file2,
    tts_text,
    tts_voice1,
    tts_voice2,
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
        
        if audio_source_type == "上传音频":
            # 使用上传的音频文件
            audio_path1 = audio_file1.name if audio_file1 else None
            audio_path2 = audio_file2.name if audio_file2 else None
            sum_audio_path = None
        elif audio_source_type == "TTS生成":
            # 使用 TTS 生成音频
            if not tts_text:
                return None, "请输入 TTS 文本"
            
            # 检查是否为多说话人模式
            if "(s1)" in tts_text and "(s2)" in tts_text:
                # 多说话人 TTS
                if not tts_voice1 or not tts_voice2:
                    return None, "多说话人模式需要提供两个语音文件"
                if KPipeline is None:
                    return None, "TTS 功能不可用，请安装 Kokoro 模块"
                audio1, audio2, sum_audio_path = process_tts_multi(
                    tts_text, temp_dir, tts_voice1.name, tts_voice2.name
                )
                audio_path1 = f"{temp_dir}/s1.wav"
                audio_path2 = f"{temp_dir}/s2.wav"
            else:
                # 单说话人 TTS
                if not tts_voice1:
                    return None, "请提供语音文件"
                if KPipeline is None:
                    return None, "TTS 功能不可用，请安装 Kokoro 模块"
                audio1, audio_path1 = process_tts_single(
                    tts_text, temp_dir, tts_voice1.name
                )
                audio_path2 = None
                sum_audio_path = audio_path1
        else:
            return None, "无效的音频源类型"
        
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
        
        # 根据设备 VRAM 自动选择 VAE tile size
        device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
        if device_mem_capacity >= 24000:  # 24GB+
            VAE_tile_size = 0  # 不使用瓦片化，最快
        elif device_mem_capacity >= 12000:  # 12GB+
            VAE_tile_size = 256
        else:
            VAE_tile_size = 128
        
        # 调用模型生成视频
        print(f"开始生成视频: {width}x{height}, {video_length}帧")
        print(f"设备 VRAM: {device_mem_capacity:.0f}MB, VAE Tile Size: {VAE_tile_size}")
        
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
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            embedded_guidance_scale=embedded_guidance_scale,
            audio_cfg_scale=audio_guidance_scale,
            audio_proj=audio_proj,
            speakers_bboxes=speakers_bboxes,
            token_ref_target_masks=token_ref_target_masks,  # 添加目标掩码
            model_type="multitalk",
            sample_solver="unipc",
            VAE_tile_size=VAE_tile_size,  # 添加 VAE 瓦片化参数
            joint_pass=device_mem_capacity >= 16000,  # 16GB+ 启用 joint pass 优化
            callback=progress_callback if num_inference_steps > 10 else None
        )
        
        # 保存视频
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"multitalk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        
        # 解码并保存视频
        video_frames = wan_model.decode_video(samples)
        
        # 使用 imageio 保存视频
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
        
        for frame in video_frames:
            # 确保帧是正确的格式
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            if len(frame.shape) == 3 and frame.shape[0] in [3, 4]:
                frame = frame.transpose(1, 2, 0)
            writer.append_data(frame)
        
        writer.close()
        
        # 如果有音频，合并音频和视频
        if sum_audio_path:
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
                    model_path = gr.Textbox(
                        label="模型路径",
                        value="ckpts/multitalk-wan2gp-14B.pth",
                        placeholder="输入模型文件路径"
                    )
                    high_vram_mode = gr.Checkbox(
                        label="高 VRAM 模式（24GB+）",
                        value=False,
                        info="禁用量化，使用更高精度，提升速度和质量"
                    )
                    load_btn = gr.Button("加载模型", variant="primary")
                
                # 基础参数
                with gr.Group():
                    gr.Markdown("### 🎬 基础参数")
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
                
                # 音频设置
                with gr.Group():
                    gr.Markdown("### 🎤 音频设置")
                    
                    audio_source_type = gr.Radio(
                        choices=["上传音频", "TTS生成"],
                        value="上传音频",
                        label="音频源类型"
                    )
                    
                    # 上传音频选项
                    with gr.Group(visible=True) as upload_audio_group:
                        audio_file1 = gr.File(label="说话人1音频", file_types=["audio"])
                        audio_file2 = gr.File(label="说话人2音频（可选）", file_types=["audio"])
                    
                    # TTS 选项
                    with gr.Group(visible=False) as tts_group:
                        tts_text = gr.Textbox(
                            label="TTS文本",
                            placeholder="单说话人: 直接输入文本\n多说话人: (s1)说话人1的话 (s2)说话人2的话",
                            lines=3
                        )
                        tts_voice1 = gr.File(label="说话人1语音模板", file_types=[".pt"])
                        tts_voice2 = gr.File(label="说话人2语音模板（多说话人时需要）", file_types=[".pt"])
                    
                    speakers_locations = gr.Textbox(
                        label="说话人位置",
                        value="25:75",
                        placeholder="格式: 左:右 或 左:上:右:下（百分比）",
                        info="示例: '25:75' 表示两个说话人分别在屏幕 25% 和 75% 的位置"
                    )
                    
                    audio_combination_type = gr.Radio(
                        choices=["add", "para"],
                        value="add",
                        label="音频组合方式",
                        info="add: 顺序播放, para: 并行播放"
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
                            "Two people having a conversation in a modern office",
                            "blurry, low quality",
                            "上传音频",
                            "25:75",
                            "add"
                        ],
                        [
                            "Interview scene with two speakers at a news desk",
                            "cartoon, animated",
                            "TTS生成",
                            "30:70",
                            "add"
                        ]
                    ],
                    inputs=[prompt, negative_prompt, audio_source_type, speakers_locations, audio_combination_type]
                )
        
        # 事件处理
        def toggle_audio_input(audio_type):
            return (
                gr.update(visible=(audio_type == "上传音频")),
                gr.update(visible=(audio_type == "TTS生成"))
            )
        
        audio_source_type.change(
            toggle_audio_input,
            inputs=[audio_source_type],
            outputs=[upload_audio_group, tts_group]
        )
        
        load_btn.click(
            load_multitalk_model,
            inputs=[model_path, high_vram_mode],
            outputs=[status_text]
        )
        
        generate_btn.click(
            generate_multitalk_video,
            inputs=[
                prompt, negative_prompt,
                audio_source_type, audio_file1, audio_file2,
                tts_text, tts_voice1, tts_voice2,
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