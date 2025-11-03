# WAS LM Studio Easy-Query

A custom node pack that integrates LM Studio's Python SDK with ComfyUI for local LLM/VLM workflows. It provides nodes for model configuration, flexible per-request options, text+vision querying, image captioning, and chat with persistent or temporary conversations.

## Requirements
Install the Python dependencies used by these nodes with the requirements.txt or simply install required modules:

```
pip install lmstudio numpy Pillow
```

## Installation
Open manager and search Manager for `WAS LM Studio Easy-Query`, or:

1. Place this folder in `ComfyUI/custom_nodes/comfyui_lmstudio_nodes`.
2. Start or restart ComfyUI. On startup, this pack clears:
   - `temp_convos/` (temporary conversations)

## Node Overview

| Node | Type | Key Inputs | Outputs | Notes |
| --- | --- | --- | --- | --- |
| LM Studio Model | Config | base_url, model_id, temperature, max_tokens, seed, timeout, image_max_size, unload | LMSTUDIO_MODEL | Produces model config consumed by other nodes |
| LM Studio Options | Options | temperature, max_tokens, seed, top_p, top_k, frequency_penalty, presence_penalty, repeat_penalty, stop | LMSTUDIO_OPTIONS | Per-request overrides; take precedence over Model values |
| LM Studio Query | Request | model, mode (one-by-one/batch), system_prompt, user_prompt, images?, options? | responses (list) | Text/VLM single-turn. Images sent via temp PNG paths to `lms.prepare_image(path)` |
| LM Studio Easy-Caption | Caption | model, images, mode (one-by-one/batch), task_name, user_prompt?, options? | captions (list) | Preset tasks; same image handling as Query |
| LM Studio Chat | Chat | model, conversation_choice, conversation_name, mode, system_prompt, user_prompt, images?, options?, temp_convo | responses (list), queries (list), conversation_name (string) | "New Conversation" sentinel; temp_convo stores in temp_convos |

## Tips
- Use a VLM model (e.g., qwen2-vl variants) for image inputs.
- Tooltips in the UI describe all fields; Options override Model settings per request.
- Enable `unload` in the Model node if you want to free VRAM after each node run.

## Troubleshooting
- Responses look text-only with images:
  - Verify a VLM is selected in the Model node
  - Ensure images are wired into `images` input
- Conversation selection errors:
  - Keep `conversation_choice` on "New Conversation" and type a name, or pick an existing name from the dropdown
- If the SDK logs websocket shutdown messages on exit, they are informational during normal shutdown.

## License
MIT (see repository terms).
