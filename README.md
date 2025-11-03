# WAS LM Studio Easy-Query

A custom node pack that integrates LM Studio's Python SDK with ComfyUI for local LLM/VLM workflows. It provides nodes for model configuration, flexible per-request options, text+vision querying, image captioning, and chat with persistent or temporary conversations.

<img width="2441" height="1166" alt="image" src="https://github.com/user-attachments/assets/783173e3-f48a-4662-98a8-28ad264ec9fd" />

## Requirements
Install the Python dependencies used by these nodes with the requirements.txt or simply install required modules:

```
pip install lmstudio numpy Pillow
```

## Installation
Open manager and search Manager for `WAS LMStudio Easy-Query`, or:

1. Place this folder in `ComfyUI/custom_nodes/ComfyUI_LMStudio_EasyQuery`.
2. Start or restart ComfyUI. On startup, this pack clears:
   - `temp_convos/` (temporary conversations)

## Configuration

Edit `lmstudio_config.json` in the node pack directory to customize default settings:

```json
{
  "temperature": 0.15,
  "max_tokens": 768,
  "seed": 0,
  "unload_after_use": true,
  "default_model": "qwen/qwen2.5-vl-3b",
  "image_max_sizes": [256, 512, 1024, 2048],
  "default_image_max_size": 1024,
  "allowed_root_directories": ["/mnt/k", "/mnt/m"]
}
```

### Configuration Options

| Option | Type | Description |
| --- | --- | --- |
| `temperature` | float | Default sampling temperature (0.0-2.0). Lower = more deterministic |
| `max_tokens` | int | Default maximum tokens to generate per response |
| `seed` | int | Default random seed (0 = disabled) |
| `unload_after_use` | bool | Automatically unload model after queries to free VRAM |
| `default_model` | string | Model ID to select by default in the Model node dropdown |
| `image_max_sizes` | array | Available image size options in the Model node dropdown |
| `default_image_max_size` | int | Default maximum edge size for image resizing |
| `allowed_root_directories` | array | **Required for dataset nodes.** Whitelist of directories that can be accessed |

### Adding Allowed Root Directories

For security, the **WAS Load Image Directory** node requires directories to be explicitly whitelisted. To add directories:

1. Open `lmstudio_config.json`
2. Add your dataset paths to the `allowed_root_directories` array:

**Windows example:**
```json
"allowed_root_directories": ["k:/datasets", "c:/data/images", "d:/training"]
```

**Linux/Mac example:**
```json
"allowed_root_directories": ["/home/user/datasets", "/mnt/storage/images"]
```

**Notes:**
- Use forward slashes (`/`) even on Windows
- Subdirectories are automatically allowed (e.g., `k:/datasets` allows `k:/datasets/anime`)
- Paths are case-sensitive on Linux/Mac
- Restart ComfyUI after editing the config

## Node Overview

| Node | Type | Key Inputs | Outputs | Notes |
| --- | --- | --- | --- | --- |
| **LM Studio Model** | Config | model, manual_model_id, unload_after_use, temperature, max_tokens, seed, image_max_size | LMSTUDIO_MODEL | Produces model config consumed by other nodes. Fetches available models from LM Studio SDK |
| **LM Studio Options** | Options | temperature, max_tokens, seed, top_p, top_k, frequency_penalty, presence_penalty, repeat_penalty, stop | LMSTUDIO_OPTIONS | Per-request overrides; take precedence over Model values |
| **LM Studio Query** | Request | model, mode (one-by-one/batch), system_prompt, user_prompt, images?, options? | responses (list) | Text/VLM single-turn queries. Images resized to image_max_size before encoding |
| **LM Studio Easy-Caption** | Caption | model, images, mode (one-by-one/batch), task_name, user_prompt?, options? | captions (list) | Uses preset tasks from /tasks/*.txt files; same image handling as Query |
| **LM Studio Chat** | Chat | model, conversation_choice, conversation_name, mode, system_prompt, user_prompt, images?, options?, temp_convo | responses (list), queries (list), conversation_name (string) | Persistent conversations stored as JSON; temp_convo stores in temp_convos/ |
| **WAS Load Image Directory** | Dataset | directory_path, recursive, extensions, dataset_output_path, copy_images, force_aspect, max_size, resize_mode | LMSTUDIO_DATASET_IMAGES | Loads images from directory for batch captioning. Requires allowed_root_directories in config |
| **LM Studio Easy-Caption Dataset** | Dataset | model, dataset, task_name, user_prompt?, options? | captions (list), written_caption_paths (list), result (string) | Batch captions entire dataset, writes .txt files alongside images |

## Tips
- Use a VLM model (e.g., qwen2-vl variants) for image inputs.
- Tooltips in the UI describe all fields; Options override Model settings per request.
- Enable `unload_after_use` in the Model node to free VRAM after each query.
- Create custom caption tasks by adding .txt files to the `/tasks` directory.
- For dataset captioning, configure `allowed_root_directories` in `lmstudio_config.json` for security.

## Troubleshooting
- **Responses look text-only with images:**
  - Verify a VLM is selected in the Model node
  - Ensure images are wired into `images` input
- **Conversation selection errors:**
  - Keep `conversation_choice` on "New Conversation" and type a name, or pick an existing name from the dropdown
- **Dataset loading fails:**
  - Add your dataset directory to `allowed_root_directories` in `lmstudio_config.json`
  - Example: `"allowed_root_directories": ["k:/datasets", "c:/data"]`
- **Model not found:**
  - Use `manual_model_id` field if models don't appear in dropdown
  - Ensure LM Studio is running and models are downloaded
- If the SDK logs websocket shutdown messages on exit, they are informational during normal shutdown.

## License
MIT (see repository terms).
