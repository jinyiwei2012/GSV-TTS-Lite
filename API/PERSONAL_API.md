# GSV-TTS 个人化应用 API

简单、功能全的 TTS API，支持流式和批量两种推理模式。

## 快速开始

```bash
cd API
python personal_api.py
```

服务启动后访问 http://localhost:8000/docs 查看交互式文档。

## API 端点

### 1. 流式推理 `/tts/stream`

使用 SSE (Server-Sent Events) 实时推送音频片段，适用于实时对话、长文本生成等需要低延迟响应的场景。

**请求参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| text | string | 是 | - | 要合成的文本 |
| speaker_audio | string | 是 | - | 说话人参考音频路径或URL |
| prompt_audio | string | 是 | - | 提示音频路径或URL |
| prompt_text | string | 否 | - | 提示音频文本，为空时自动ASR识别 |
| text_language | string | 否 | "auto" | 目标文本语言: auto/ja/zh/en，混语文本建议手动指定 |
| prompt_language | string | 否 | "auto" | 提示音频文本语言: auto/ja/zh/en |
| is_cut_text | bool | 否 | true | 是否按标点切分文本 |
| cut_minlen | int | 否 | 10 | 文本切分最小长度 |
| cut_mute | float | 否 | 0.3 | 切分后的静音时长(秒) |
| stream_mode | string | 否 | "token" | 流式模式: token 或 sentence |
| stream_chunk | int | 否 | 25 | token模式下每次生成的token数 |
| overlap_len | int | 否 | 5 | 重叠长度，用于平滑拼接 |
| boost_first_chunk | bool | 否 | true | 是否加速首个chunk生成 |
| top_k | int | 否 | 15 | GPT采样top_k |
| top_p | float | 否 | 1.0 | GPT采样top_p |
| temperature | float | 否 | 1.0 | GPT采样温度 |
| repetition_penalty | float | 否 | 1.35 | 重复惩罚 |
| noise_scale | float | 否 | 0.5 | 噪声强度 |
| speed | float | 否 | 1.0 | 语速 |

**返回格式 (SSE)：**

```
event: audio
data: {"audio": "<base64>", "sample_rate": 32000, "duration": 0.5, "subtitles": [...], "text": "..."}

event: done
data: {"total_duration": 5.2}

event: error
data: {"error": "错误信息"}
```

**调用示例：**

```python
import httpx
import json
import base64

async def stream_tts():
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", "http://localhost:8000/tts/stream", json={
            "text": "你好，这是一段测试文本。今天天气真不错。",
            "speaker_audio": "examples/AnAn.ogg",
            "prompt_audio": "examples/AnAn.ogg",
            "prompt_text": "ちが……ちがう。レイア、貴様は間違っている。"
        }) as response:
            async for line in response.aiter_lines():
                if line.startswith("event: audio"):
                    continue
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "audio" in data:
                        audio_bytes = base64.b64decode(data["audio"])
                        print(f"收到音频片段: {data['duration']:.2f}秒")
```

### 2. 批量推理 `/tts/batched`

一次请求生成多个音频，适用于批量生成、离线处理等不需要实时响应的场景。

**请求参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| texts | [string] | 是 | - | 要合成的文本列表 |
| speaker_audio | string | 是 | - | 说话人参考音频路径或URL |
| prompt_audio | string | 是 | - | 提示音频路径或URL |
| prompt_text | string | 否 | - | 提示音频文本，为空时自动ASR识别 |
| text_languages | string/[string] | 否 | "auto" | 目标文本语言: auto/ja/zh/en，支持逐句列表 |
| prompt_languages | string/[string] | 否 | "auto" | 提示音频文本语言: auto/ja/zh/en，支持逐句列表 |
| is_cut_text | bool | 否 | true | 是否按标点切分文本 |
| cut_minlen | int | 否 | 10 | 文本切分最小长度 |
| cut_mute | float | 否 | 0.3 | 切分后的静音时长(秒) |
| return_subtitles | bool | 否 | false | 是否返回字幕时间戳 |
| top_k | int | 否 | 15 | GPT采样top_k |
| top_p | float | 否 | 1.0 | GPT采样top_p |
| temperature | float | 否 | 1.0 | GPT采样温度 |
| repetition_penalty | float | 否 | 1.35 | 重复惩罚 |
| noise_scale | float | 否 | 0.5 | 噪声强度 |
| speed | float | 否 | 1.0 | 语速 |

**返回格式 (JSON)：**

```json
{
    "success": true,
    "count": 2,
    "filenames": ["tts_abc12345.wav", "tts_def67890.wav"],
    "prompt_text_used": "ちが……ちがう。レイア、貴様は間違っている。",
    "subtitles": [
        [{"text": "你好", "start_s": 0.0, "end_s": 0.3}],
        [{"text": "世界", "start_s": 0.0, "end_s": 0.4}]
    ]
}
```

**调用示例：**

```python
import httpx

response = httpx.post("http://localhost:8000/tts/batched", json={
    "texts": ["第一段文本。", "第二段文本。"],
    "speaker_audio": "examples/AnAn.ogg",
    "prompt_audio": "examples/AnAn.ogg",
    "prompt_text": "ちが……ちがう。レイア、貴様は間違っている。",
    "return_subtitles": True
}, timeout=60.0)

result = response.json()
print(f"生成了 {result['count']} 个音频文件")
for filename in result["filenames"]:
    print(f"文件: {filename}")
```

### 3. 获取音频 `/audio/{filename}`

获取生成的音频文件。

**示例：**
```
GET /audio/tts_abc12345.wav
```

### 4. 多角色推理 (Multi-Speaker) 🆕

在同一服务中加载多个微调角色，共享 GPT + SoVITS 模型骨干，每个角色仅注入 ~5-15% 专属权重，显存节省 50-75%。

**端点列表：**

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/multi-speaker/init` | 初始化多角色引擎 |
| POST | `/multi-speaker/add` | 添加角色 |
| POST | `/multi-speaker/remove` | 移除角色 |
| GET  | `/multi-speaker/list` | 列出已加载角色 |
| POST | `/multi-speaker/infer` | 单角色推理 |
| POST | `/multi-speaker/batch` | 多角色批量推理 |
| POST | `/multi-speaker/stream` | 单角色流式推理 (SSE) |

**使用流程：**

**1. 初始化引擎**
```bash
curl -X POST "http://localhost:8000/multi-speaker/init" \
  -H "Content-Type: application/json" \
  -d '{"use_bert": true, "use_flash_attn": false}'
```

**2. 添加角色**（`speaker_audio` / `prompt_audio` 支持本地路径或 URL）
```bash
curl -X POST "http://localhost:8000/multi-speaker/add" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "alice",
    "gpt_model_path": "models/alice_gpt.ckpt",
    "sovits_model_path": "models/alice_sovits.pth",
    "speaker_audio": "audio/alice_ref.wav",
    "prompt_audio": "audio/alice_prompt.ogg",
    "prompt_text": "こんにちは、アリスです。"
  }'
```
> 响应返回 `mode`：`shared`（共享骨干）或 `full_model_degraded`（架构不兼容时自动降级为完整模型）。

**3. 查看角色**
```bash
curl "http://localhost:8000/multi-speaker/list"
```

**4. 单角色推理**（支持语言参数与按次 prompt 覆盖）
```bash
curl -X POST "http://localhost:8000/multi-speaker/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "speaker": "alice",
    "text": "こんにちは！",
    "text_language": "ja",
    "prompt_language": "ja",
    "prompt_audio_path": "audio/other_style.ogg",
    "prompt_audio_text": "別のスタイルのテキスト。"
  }'
```

**5. 批量推理**（每条可独立指定语言）
```bash
curl -X POST "http://localhost:8000/multi-speaker/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "speaker_texts": [
      {"speaker": "alice", "text": "こんにちは", "text_language": "ja"},
      {"speaker": "bob",   "text": "你好", "text_language": "zh"}
    ]
  }'
```

**6. 流式推理 (SSE)** — Token 级流式输出，低延迟实时反馈
```bash
curl -N -X POST "http://localhost:8000/multi-speaker/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "speaker": "alice",
    "text": "こんにちは！長いテキストでも低遅延で読み上げます。",
    "text_language": "ja",
    "stream_chunk": 25,
    "overlap_len": 5
  }'
```
返回 SSE 事件：
```
event: audio
data: {"audio": "<base64>", "sample_rate": 32000, "duration": 0.5, "text": "..."}

event: done
data: {"total_duration": 5.2}

event: error
data: {"error": "错误信息"}
```

## 功能特性

### 外链音频URL支持

所有音频参数支持HTTP/HTTPS URL，API会自动下载：

```json
{
    "speaker_audio": "https://example.com/speaker.wav",
    "prompt_audio": "https://example.com/prompt.mp3"
}
```

### ASR自动识别

当 `prompt_text` 为空时，自动使用ASR模型识别提示音频的文本内容。

可通过环境变量控制：
```bash
USE_ASR=true python personal_api.py   # 启用ASR (默认)
USE_ASR=false python personal_api.py  # 禁用ASR
```

### 流式模式选择

- **token模式**：按token数量切分，延迟更低，适合实时对话
- **sentence模式**：按句子切分，音频更连贯，适合长文本朗读

### 多角色 (Multi-Speaker)

- 共享骨干 + 角色专属权重（~5-15%），显存节省 50-75%
- 角色模型架构不兼容时自动降级为完整模型加载
- 支持按次调用覆盖角色默认的风格参考音频（`prompt_audio_path` / `prompt_audio_text`）

## 两种模式对比

| 特性 | Stream | Batched |
|------|--------|---------|
| 适用场景 | 实时对话、长文本 | 批量生成、离线处理 |
| 响应方式 | SSE实时推送 | 一次性返回 |
| 首字延迟 | 低 | 高 |
| GPU利用率 | 中 | 高 |
| 音频返回 | base64编码 | 文件 |

## 环境要求

- Python 3.10+
- CUDA 11.x+ (推荐)
- 依赖见 requirements.txt
