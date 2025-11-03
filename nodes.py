import os
import json
import io
import uuid

import numpy as np
import lmstudio as lms

from typing import Any, Dict, List, Tuple, Optional
from PIL import Image
import shutil

import folder_paths
from comfy.utils import ProgressBar

def read_config() -> Dict[str, Any]:
    """Read configuration from lmstudio_config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "lmstudio_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def read_tasks() -> Dict[str, str]:
    """Read task files from the tasks directory."""
    tasks_dir = os.path.join(os.path.dirname(__file__), "tasks")
    tasks = {}
    if os.path.exists(tasks_dir):
        for filename in os.listdir(tasks_dir):
            if filename.endswith(".txt"):
                task_name = filename[:-4]  # Drop .txt extension
                task_path = os.path.join(tasks_dir, filename)
                with open(task_path, "r", encoding="utf-8") as f:
                    tasks[task_name] = f.read().strip()
    return tasks

def fetch_models() -> Optional[List[str]]:
    """Fetch available models from LM Studio using the official SDK."""
    try:
        downloaded = lms.list_downloaded_models()
        keys: List[str] = []
        for m in downloaded:
            key = getattr(m, "model_key", None) or getattr(m, "key", None)
            keys.append(key if isinstance(key, str) else str(m))
        return keys or None
    except Exception as e:
        print(f"Error fetching models: {e}")
        return None

def image_tensor_to_png_bytes(image_tensor, max_edge: int = 1024) -> bytes:
    """Convert an image tensor to PNG bytes for lmstudio.prepare_image."""
    t = image_tensor
    if hasattr(t, "dim"):
        if t.dim() == 4:
            t = t[0] if t.shape[0] > 1 else t.squeeze(0)
        arr = t.detach().cpu().numpy()
    else:
        arr = np.asarray(t)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim == 3 and arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(arr)
    if max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _temp_images_dir() -> str:
    try:
        d = folder_paths.get_temp_directory()
    except Exception:
        d = os.path.join(os.path.dirname(__file__), "temp_images")
        os.makedirs(d, exist_ok=True)
    return d

def image_tensor_to_temp_png_path(image_tensor, max_edge: int = 1024) -> str:
    data = image_tensor_to_png_bytes(image_tensor, max_edge=max_edge)
    d = _temp_images_dir()
    p = os.path.join(d, f"{uuid.uuid4().hex}.png")
    with open(p, "wb") as f:
        f.write(data)
    return p

def listify(x):
    """Convert input to list if it's not already."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def string_list(strings: List[str]) -> Tuple[List[str]]:
    """Return strings as a tuple containing a list (for ComfyUI output)."""
    return (strings,)

def _map_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        if k == "max_tokens":
            out["maxTokens"] = v
        elif k == "top_p":
            out["topP"] = v
        elif k == "top_k":
            out["topK"] = v
        elif k == "frequency_penalty":
            out["frequencyPenalty"] = v
        elif k == "presence_penalty":
            out["presencePenalty"] = v
        elif k == "repeat_penalty":
            out["repeatPenalty"] = v
        else:
            out[k] = v
    return out

class WASLMStudioModel:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = read_config()
        models = fetch_models() or ["<no models found>"]

        default_model = cfg.get("default_model") or (models[0] if models else "")
        temperature_default = float(cfg.get("temperature", 0.2))
        max_tokens_default = int(cfg.get("max_tokens", 512))
        seed_default = int(cfg.get("seed", 0))
        unload_default = bool(cfg.get("unload_after_use", True))

        size_choices = [str(s) for s in cfg.get("image_max_sizes", [256, 512, 1024, 2048])]
        size_default = str(cfg.get("default_image_max_size", 1024))
        if size_default not in size_choices:
            size_default = size_choices[0]

        return {
            "required": {
                "model": (
                    list(models),
                    {
                        "default": default_model if default_model in models else models[0],
                        "tooltip": "Model key discovered from the LM Studio SDK. Pick a listed model or use manual_model_id.",
                    },
                ),
                "manual_model_id": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "e.g. qwen/qwen2.5-vl-3b",
                        "tooltip": "Optional manual model identifier if your LM Studio instance is not returning it via /models or runs on a different base URL.",
                    },
                ),
                "unload_after_use": (
                    "BOOLEAN",
                    {
                        "default": unload_default,
                        "tooltip": "Unload the model after queries to free up memory. Uses LM Studio SDK for proper model management.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": temperature_default,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature. Higher is more random; lower is more deterministic.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": max_tokens_default,
                        "min": 1,
                        "max": 32768,
                        "tooltip": "Maximum new tokens to generate for the assistant reply.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": seed_default,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Optional seed for deterministic sampling if supported by LM Studio. Use 0 to disable.",
                    },
                ),
                "image_max_size": (
                    list(size_choices),
                    {
                        "default": size_default,
                        "tooltip": "Maximum edge size for input images (keeps aspect ratio). Images larger than this are downscaled before encoding and sending to LM Studio.",
                    },
                ),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("LMSTUDIO_MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "LM Studio"
    FUNCTION = "load_model"

    def load_model(
        self,
        model: str,
        manual_model_id: str,
        unload_after_use: bool,
        temperature: float,
        max_tokens: int,
        seed: int,
        image_max_size: str,
    ):
        cfg = read_config()

        chosen_id = manual_model_id.strip() if manual_model_id.strip() else model
        if chosen_id == "<no models found>":
            chosen_id = manual_model_id.strip() or ""

        selected_max = int(image_max_size) if str(image_max_size).isdigit() else int(cfg.get("default_image_max_size", 1024))

        try:
            _handle = lms.llm(chosen_id)
            print(f"Loaded (or attached to) model: {chosen_id}")
        except Exception as e:
            print(f"Warning: Could not load model {chosen_id}: {e}")

        result = {
            "model_id": chosen_id,
            "unload": bool(unload_after_use),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "seed": int(seed) if seed != 0 else None,
            "image_max_size": int(selected_max),
        }
        return (result,)

class WASLMStudioQuery:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "LMSTUDIO_MODEL",
                    {
                        "tooltip": "LM Studio model settings produced by the LM Studio Model node. Contains model_id, temperature, max_tokens, seed, and image_max_size.",
                    },
                ),
                "mode": (
                    ["one-by-one", "batch"],
                    {
                        "default": "one-by-one",
                        "tooltip": "Batch sends all images in a single request; one-by-one sends one request per image using the same prompts.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional system instructions for the model.",
                        "tooltip": "System role content that sets the assistant's behavior for this request. If blank, no system message is added.",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "default": "Describe the image.",
                        "multiline": True,
                        "tooltip": "User message sent to the model. Works alone for text-only or together with provided images for vision models.",
                    },
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Optional IMAGE input. Provide one or more images. Resized to image_max_size before being sent.",
                    },
                ),
                "options": (
                    "LMSTUDIO_OPTIONS",
                    {
                        "tooltip": "Per-request overrides (temperature, max_tokens, seed, top_p, top_k, penalties, stop). These take precedence over values from the Model node.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("responses",)
    CATEGORY = "LM Studio"
    FUNCTION = "run_query"
    OUTPUT_IS_LIST = (True,)

    def run_query(
        self,
        model: Dict[str, Any],
        mode: str,
        system_prompt: str,
        user_prompt: str,
        images=None,
        options: Optional[Dict[str, Any]] = None,
    ):
        model_id = model.get("model_id", "")
        temperature = float(model.get("temperature", 0.2))
        max_tokens = int(model.get("max_tokens", 512))
        seed = model.get("seed", None)
        max_edge = int(model.get("image_max_size", 1024))
        unload = model.get("unload", False)

        model_handle = None
        
        responses_out: List[str] = ["Error: Failed to process request"]
        tmp_img_paths: List[str] = []
        try:
            imgs = listify(images)
            model_handle = lms.llm(model_id) if model_id else lms.llm()
            if imgs:
                if mode == "one-by-one":
                    out: List[str] = []
                    pbar = ProgressBar(len(imgs))
                    for idx, img in enumerate(imgs):
                        path = image_tensor_to_temp_png_path(img, max_edge=max_edge)
                        tmp_img_paths.append(path)
                        img_handle = lms.prepare_image(path)
                        chat = lms.Chat(system_prompt) if system_prompt else lms.Chat()
                        chat.add_user_message(user_prompt or "", images=[img_handle])
                        params: Dict[str, Any] = {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        }
                        if seed is not None:
                            params["seed"] = seed
                        if options:
                            try:
                                for k, v in options.items():
                                    params[k] = v
                            except Exception:
                                pass
                        pred = model_handle.respond(chat, config=_map_params(params))
                        out.append(str(pred))
                        pbar.update_absolute(idx + 1)
                    responses_out = out
                else:
                    img_handles = []
                    for i in imgs:
                        path = image_tensor_to_temp_png_path(i, max_edge=max_edge)
                        tmp_img_paths.append(path)
                        img_handles.append(lms.prepare_image(path))
                    chat = lms.Chat(system_prompt) if system_prompt else lms.Chat()
                    chat.add_user_message(user_prompt or "", images=img_handles)
                    params: Dict[str, Any] = {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if seed is not None:
                        params["seed"] = seed
                    if options:
                        try:
                            for k, v in options.items():
                                params[k] = v
                        except Exception:
                            pass
                    pred = model_handle.respond(chat, config=_map_params(params))
                    responses_out = [str(pred)]
            else:
                chat = lms.Chat(system_prompt) if system_prompt else lms.Chat()
                chat.add_user_message(user_prompt or "")
                params: Dict[str, Any] = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if seed is not None:
                    params["seed"] = seed
                if options:
                    try:
                        for k, v in options.items():
                            params[k] = v
                    except Exception:
                        pass
                pred = model_handle.respond(chat, config=_map_params(params))
                responses_out = [str(pred)]

        except Exception as e:
            print(f"Error in run_query: {e}")
            responses_out = [f"Error: {str(e)}"]
        finally:
            if unload and model_handle is not None:
                try:
                    model_handle.unload()
                    print(f"Unloaded model: {model_id}")
                except Exception as e:
                    print(f"Warning: Could not unload model {model_id}: {e}")
            try:
                if model_handle is not None:
                    del model_handle
            except Exception:
                pass
            try:
                for p in tmp_img_paths:
                    if isinstance(p, str) and os.path.isfile(p):
                        os.remove(p)
            except Exception:
                pass

        return string_list(responses_out)


class WASLMStudioOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Sampling temperature. Higher = more random; lower = more deterministic."},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 32768, "tooltip": "Maximum number of new tokens to generate for the response."},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**31 - 1, "tooltip": "Optional random seed for reproducible sampling. Use 0 to disable and let the model choose."},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling: consider tokens with cumulative probability up to top_p. Set to 1.0 to disable."},
                ),
                "top_k": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2048, "tooltip": "Top-K sampling: only consider the top_k most likely tokens. Set to 0 to disable."},
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Penalize tokens proportionally to how often they have appeared. Helps reduce repetition."},
                ),
                "presence_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Penalize tokens if they have appeared at all. Encourages introducing new topics."},
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Generic repetition penalty (model/engine specific). 1.0 means no penalty."},
                ),
            },
            "optional": {
                "stop": (
                    "STRING",
                    {"default": "", "placeholder": ",,", "tooltip": "Comma-separated stop strings. Generation will stop when any is encountered. Leave blank for none."},
                ),
            },
        }

    RETURN_TYPES = ("LMSTUDIO_OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "LM Studio"
    FUNCTION = "make_options"

    def make_options(
        self,
        temperature: float,
        max_tokens: int,
        seed: int,
        top_p: float,
        top_k: int,
        frequency_penalty: float,
        presence_penalty: float,
        repeat_penalty: float,
        stop: str = "",
    ):
        params: Dict[str, Any] = {
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "seed": int(seed) if seed != 0 else None,
            "top_p": float(top_p),
            "top_k": int(top_k),
            "frequency_penalty": float(frequency_penalty),
            "presence_penalty": float(presence_penalty),
            "repeat_penalty": float(repeat_penalty),
        }
        if stop.strip():
            params["stop"] = [s for s in stop.split(",") if s]
        return (params,)


class WASLMStudioChat:
    @staticmethod
    def convo_dir(temp: bool = False) -> str:
        folder = "temp_convos" if temp else "conversations"
        d = os.path.join(os.path.dirname(__file__), folder)
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def list_conversation(cls, temp: bool = False) -> List[str]:
        d = cls.convo_dir(temp)
        out: List[str] = []
        try:
            for f in os.listdir(d):
                if f.lower().endswith(".json"):
                    out.append(os.path.splitext(f)[0])
        except Exception:
            pass
        return out

    @classmethod
    def list_all_conversations(cls) -> List[str]:
        a = set(cls.list_conversation(False))
        b = set(cls.list_conversation(True))
        names = sorted(a.union(b))
        return names

    @classmethod
    def INPUT_TYPES(cls):
        names = cls.list_conversation(False)
        choices = ["New Conversation"] + [n for n in names if n != "New Conversation"]
        return {
            "required": {
                "model": (
                    "LMSTUDIO_MODEL",
                    {
                        "tooltip": "Model settings including model_id, temperature, max_tokens, seed, and image_max_size.",
                    },
                ),
                "conversation_choice": (
                    list(choices),
                    {
                        "default": "New Conversation",
                        "tooltip": "Pick an existing conversation or select 'New Conversation' to start a new one. Provide 'conversation_name' to name it, or leave blank to auto-generate.",
                    },
                ),
                "conversation_name": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "optional-new-conversation-name",
                        "tooltip": "If provided, creates/uses this conversation. If left blank, uses the dropdown selection.",
                    },
                ),
                "mode": (
                    ["one-by-one", "batch"],
                    {
                        "default": "one-by-one",
                        "tooltip": "Batch sends all images in a single request; one-by-one sends one request per image.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional system instructions for this conversation (used when creating).",
                        "tooltip": "System instructions for the assistant's behavior. If the conversation has no prior messages, this will be set as the initial system message.",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "User message to append and send.",
                        "tooltip": "User message appended to the conversation and sent to the model on this run.",
                    },
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Optional images to include with the user message. Resized to image_max_size.",
                    },
                ),
                "options": (
                    "LMSTUDIO_OPTIONS",
                    {
                        "tooltip": "Per-request overrides (temperature, max_tokens, seed, top_p, top_k, penalties, stop). These take precedence over values from the Model node.",
                    },
                ),
                "temp_convo": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Store/load conversation in a temporary workspace (cleared on startup). New conversations use temp when enabled.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("responses", "queries", "conversation_name")
    CATEGORY = "LM Studio"
    FUNCTION = "chat"
    OUTPUT_IS_LIST = (True, True, False)

    @classmethod
    def history_path(cls, name: str, temp: bool = False) -> str:
        return os.path.join(cls.convo_dir(temp), f"{name}.json")

    @classmethod
    def load_history(cls, name: str, temp: bool = False) -> Dict[str, Any]:
        path = cls.history_path(name, temp=temp)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"messages": []}
        return {"messages": []}

    @classmethod
    def save_history(cls, name: str, data: Dict[str, Any], temp: bool = False) -> None:
        path = cls.history_path(name, temp=temp)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: could not save conversation '{name}': {e}")

    def chat(
        self,
        model: Dict[str, Any],
        conversation_choice: str,
        conversation_name: str,
        mode: str,
        system_prompt: str,
        user_prompt: str,
        images=None,
        options: Optional[Dict[str, Any]] = None,
        temp_convo: bool = False,
    ):
        model_id = model.get("model_id", "")
        temperature = float(model.get("temperature", 0.2))
        max_tokens = int(model.get("max_tokens", 512))
        seed = model.get("seed", None)
        max_edge = int(model.get("image_max_size", 1024))
        unload = model.get("unload", False)

        name = conversation_name.strip() or conversation_choice
        if not name or name in ("<none>", "<None>", "New Conversation"):
            name = f"chat-{__import__('time').strftime('%Y%m%d-%H%M%S')}"

        persist_exists = os.path.exists(self.__class__.history_path(name, temp=False))
        temp_exists = os.path.exists(self.__class__.history_path(name, temp=True))
        store_temp = temp_exists or (not persist_exists and temp_convo)

        history = self.__class__.load_history(name, temp=store_temp)
        msgs = history.get("messages", [])
        if not msgs and system_prompt:
            msgs.append({"role": "system", "content": system_prompt})

        responses_out: List[str] = ["Error: Failed to process request"]
        tmp_img_paths: List[str] = []
        try:
            imgs = listify(images)
            model_handle = lms.llm(model_id) if model_id else lms.llm()
            if imgs:
                if mode == "one-by-one":
                    out: List[str] = []
                    pbar = ProgressBar(len(imgs))
                    for idx, img in enumerate(imgs):
                        path = image_tensor_to_temp_png_path(img, max_edge=max_edge)
                        tmp_img_paths.append(path)
                        img_handle = lms.prepare_image(path)
                        chat = lms.Chat.from_history({"messages": msgs}) if msgs else (lms.Chat(system_prompt) if system_prompt else lms.Chat())
                        chat.add_user_message(user_prompt or "", images=[img_handle])
                        params: Dict[str, Any] = {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        }
                        if seed is not None:
                            params["seed"] = seed
                        if options:
                            try:
                                for k, v in options.items():
                                    params[k] = v
                            except Exception:
                                pass
                        pred = model_handle.respond(chat, config=_map_params(params))
                        out.append(str(pred))
                        msgs.append({"role": "user", "content": user_prompt or ""})
                        msgs.append({"role": "assistant", "content": str(pred)})
                        pbar.update_absolute(idx + 1)
                    responses_out = out
                else:
                    img_handles = []
                    for i in imgs:
                        path = image_tensor_to_temp_png_path(i, max_edge=max_edge)
                        tmp_img_paths.append(path)
                        img_handles.append(lms.prepare_image(path))
                    chat = lms.Chat.from_history({"messages": msgs}) if msgs else (lms.Chat(system_prompt) if system_prompt else lms.Chat())
                    chat.add_user_message(user_prompt or "", images=img_handles)
                    params: Dict[str, Any] = {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if seed is not None:
                        params["seed"] = seed
                    if options:
                        try:
                            for k, v in options.items():
                                params[k] = v
                        except Exception:
                            pass
                    pred = model_handle.respond(chat, config=_map_params(params))
                    responses_out = [str(pred)]
                    msgs.append({"role": "user", "content": user_prompt or ""})
                    msgs.append({"role": "assistant", "content": str(pred)})
            else:
                chat = lms.Chat.from_history({"messages": msgs}) if msgs else (lms.Chat(system_prompt) if system_prompt else lms.Chat())
                chat.add_user_message(user_prompt or "")
                params: Dict[str, Any] = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if seed is not None:
                    params["seed"] = seed
                if options:
                    try:
                        for k, v in options.items():
                            params[k] = v
                    except Exception:
                        pass
                pred = model_handle.respond(chat, config=_map_params(params))
                responses_out = [str(pred)]
                msgs.append({"role": "user", "content": user_prompt or ""})
                msgs.append({"role": "assistant", "content": str(pred)})
        except Exception as e:
            print(f"Error in chat: {e}")
            responses_out = [f"Error: {str(e)}"]
        finally:
            try:
                self.__class__.save_history(name, {"messages": msgs}, temp=store_temp)
            except Exception as e:
                print(f"Warning: Could not save conversation history: {e}")
            
            if unload and model_handle is not None:
                try:
                    model_handle.unload()
                    print(f"Unloaded model: {model_id}")
                except Exception as e:
                    print(f"Warning: Could not unload model {model_id}: {e}")
            try:
                if model_handle is not None:
                    del model_handle
            except Exception:
                pass
            try:
                for p in tmp_img_paths:
                    if isinstance(p, str) and os.path.isfile(p):
                        os.remove(p)
            except Exception:
                pass

        responses_out = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
        queries_out = [m.get("content", "") for m in msgs if m.get("role") == "user"]
        return (responses_out, queries_out, name)


class WASLMStudioCaption:
    cached_tasks: Dict[str, str] = read_tasks()

    @classmethod
    def INPUT_TYPES(cls):
        cls.cached_tasks = read_tasks()
        names = list(cls.cached_tasks.keys()) if cls.cached_tasks else ["Photorealism Caption", "Anime Caption", "Tags"]
        if not cls.cached_tasks:
            cls.cached_tasks = {
                "Photorealism Caption": "You are a professional photo captioner. Describe the image in natural, concise prose with attention to lighting, lens characteristics, composition, and realistic details.",
                "Anime Caption": "You are an anime scene captioner. Describe the image with anime art cues, character design elements, expressions, and stylistic references.",
                "Tags": "Return a comma-separated list of concise, search-friendly tags describing the image content, style, and attributes. No full sentences.",
            }
            names = list(cls.cached_tasks.keys())
        return {
            "required": {
                "model": (
                    "LMSTUDIO_MODEL",
                    {
                        "tooltip": "LM Studio model settings produced by the LM Studio Model node. Contains model_id, temperature, max_tokens, seed, and image_max_size.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "IMAGE input to caption. One or more images are accepted. Each will be resized to image_max_size before encoding.",
                    },
                ),
                "mode": (
                    ["one-by-one", "batch"],
                    {
                        "default": "one-by-one",
                        "tooltip": "Batch sends all images in a single request; one-by-one sends one request per image using the same task and user prompt.",
                    },
                ),
                "task_name": (
                    list(names),
                    {
                        "default": names[0],
                        "tooltip": "Built-in task preset name loaded from /tasks/*.txt. The file content is used as the system prompt.",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Extra directions or context.",
                        "tooltip": "Optional user instructions appended to the task's system prompt. Leave blank to use only the task preset.",
                    },
                ),
            },
            "optional": {
                "options": (
                    "LMSTUDIO_OPTIONS",
                    {
                        "tooltip": "Per-request overrides (temperature, max_tokens, seed, top_p, top_k, penalties, stop). These take precedence over values from the Model node.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("captions",)
    CATEGORY = "LM Studio"
    FUNCTION = "generate_captions"
    OUTPUT_IS_LIST = (True,)

    def generate_captions(
        self,
        model: Dict[str, Any],
        images,
        mode: str,
        task_name: str,
        user_prompt: str,
        options: Optional[Dict[str, Any]] = None,
    ):
        model_id = model.get("model_id", "")
        temperature = float(model.get("temperature", 0.2))
        max_tokens = int(model.get("max_tokens", 512))
        seed = model.get("seed", None)
        max_edge = int(model.get("image_max_size", 1024))
        unload = model.get("unload", False)

        system_text = self.__class__.cached_tasks.get(task_name, task_name)
        imgs = listify(images)

        model_handle = None
        result = string_list(["Error: Failed to process request"])
        
        try:
            model_handle = lms.llm(model_id) if model_id else lms.llm()
            if mode == "one-by-one":
                out: List[str] = []
                tmp_img_paths: List[str] = []
                pbar = ProgressBar(len(imgs))
                for idx, img in enumerate(imgs):
                    path = image_tensor_to_temp_png_path(img, max_edge=max_edge)
                    tmp_img_paths.append(path)
                    img_handle = lms.prepare_image(path)
                    chat = lms.Chat(system_text) if system_text else lms.Chat()
                    chat.add_user_message(user_prompt or "", images=[img_handle])
                    params: Dict[str, Any] = {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if seed is not None:
                        params["seed"] = seed
                    if options:
                        try:
                            for k, v in options.items():
                                params[k] = v
                        except Exception:
                            pass
                    pred = model_handle.respond(chat, config=_map_params(params))
                    out.append(str(pred))
                    pbar.update_absolute(idx + 1)
                result = string_list(out)
            else:
                img_handles = []
                tmp_img_paths: List[str] = []
                for i in imgs:
                    path = image_tensor_to_temp_png_path(i, max_edge=max_edge)
                    tmp_img_paths.append(path)
                    img_handles.append(lms.prepare_image(path))
                chat = lms.Chat(system_text) if system_text else lms.Chat()
                chat.add_user_message(user_prompt or "", images=img_handles)
                params: Dict[str, Any] = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if seed is not None:
                    params["seed"] = seed
                if options:
                    try:
                        for k, v in options.items():
                            params[k] = v
                    except Exception:
                        pass
                pred = model_handle.respond(chat, config=_map_params(params))
                result = string_list([str(pred)])

        except Exception as e:
            print(f"Error in generate_captions: {e}")
            result = string_list([f"Error: {str(e)}"])
        finally:
            if unload and model_handle is not None:
                try:
                    model_handle.unload()
                    print(f"Unloaded model: {model_id}")
                except Exception as e:
                    print(f"Warning: Could not unload model {model_id}: {e}")
            try:
                if model_handle is not None:
                    del model_handle
            except Exception:
                pass

        return result


def _is_under_allowed_root(path: str, allowed_roots: List[str]) -> bool:
    try:
        if not allowed_roots:
            return False
        p = os.path.realpath(path)
        for root in allowed_roots:
            if not root:
                continue
            r = os.path.realpath(root)
            try:
                if os.path.commonpath([p, r]) == r:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


class WASLoadImageDirectory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "k:/datasets/images",
                        "tooltip": "Absolute path to a directory containing images. Must be under an allowed root.",
                    },
                ),
                "recursive": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Recurse into subdirectories."},
                ),
                "extensions": (
                    "STRING",
                    {
                        "default": ".png,.jpg,.jpeg,.webp,.bmp",
                        "tooltip": "Comma-separated list of image file extensions to include.",
                    },
                ),
                "dataset_output_path": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "k:/datasets/captions_out",
                        "tooltip": "Optional output directory for captions (and images if copy is enabled). Must be under an allowed root if provided.",
                    },
                ),
                "copy_images": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled and an output path is provided that differs from the image directory, copy each image alongside its caption file.",
                    },
                ),
                "force_aspect": (
                    ["none", "1:1", "4:3", "3:2", "16:9", "21:9", "3:4", "2:3", "9:16"],
                    {
                        "default": "none",
                        "tooltip": "Force images to a specific aspect ratio. 'none' keeps original aspect.",
                    },
                ),
                "max_size": (
                    ["512", "768", "1024", "1280", "2048"],
                    {
                        "default": "1024",
                        "tooltip": "Maximum resolution (longest edge) for resized images.",
                    },
                ),
                "resize_mode": (
                    ["none", "crop_center", "stretch", "pad", "fit"],
                    {
                        "default": "none",
                        "tooltip": "Resize/crop method: 'none' (no resize), 'crop_center' (crop to aspect), 'stretch' (distort to fit), 'pad' (letterbox), 'fit' (scale to fit).",
                    },
                ),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("LMSTUDIO_DATASET_IMAGES",)
    RETURN_NAMES = ("dataset",)
    CATEGORY = "LM Studio"
    FUNCTION = "load_dir"

    def load_dir(self, directory_path: str, recursive: bool, extensions: str, dataset_output_path: str, copy_images: bool, force_aspect: str, max_size: str, resize_mode: str):
        cfg = read_config()
        allowed_roots = cfg.get("allowed_root_directories", [])
        if not _is_under_allowed_root(directory_path, allowed_roots):
            raise ValueError("Directory is not under any allowed_root_directories. Update lmstudio_config.json.")

        out_dir = dataset_output_path.strip()
        if out_dir:
            if not _is_under_allowed_root(out_dir, allowed_roots):
                raise ValueError("dataset_output_path is not under any allowed_root_directories. Update lmstudio_config.json.")
            os.makedirs(out_dir, exist_ok=True)

        exts = {e.strip().lower() if e.strip().startswith(".") else ("." + e.strip().lower()) for e in extensions.split(",") if e.strip()}
        if not exts:
            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

        dataset: Dict[str, Dict[str, Any]] = {
            "__output_path": out_dir,
            "__copy_images": bool(copy_images),
            "__force_aspect": force_aspect,
            "__max_size": int(max_size) if max_size.isdigit() else 1024,
            "__resize_mode": resize_mode,
        }
        if recursive:
            walker = os.walk(directory_path)
            for root, _, files in walker:
                for f in files:
                    if os.path.splitext(f)[1].lower() in exts:
                        p = os.path.join(root, f)
                        dataset[p] = {"path": p, "filename": f}
        else:
            for f in os.listdir(directory_path):
                p = os.path.join(directory_path, f)
                if os.path.isfile(p) and os.path.splitext(f)[1].lower() in exts:
                    dataset[p] = {"path": p, "filename": f}

        return (dataset,)


class WASLMStudioCaptionDataset:
    @classmethod
    def INPUT_TYPES(cls):
        WASLMStudioCaption.cached_tasks = read_tasks()
        names = list(WASLMStudioCaption.cached_tasks.keys()) if WASLMStudioCaption.cached_tasks else ["Photorealism Caption", "Anime Caption", "Tags"]
        return {
            "required": {
                "model": (
                    "LMSTUDIO_MODEL",
                    {"tooltip": "Model settings from LM Studio Model node."},
                ),
                "dataset": (
                    "LMSTUDIO_DATASET_IMAGES",
                    {"tooltip": "Dictionary produced by WASLoadImageDirectory mapping file paths to metadata."},
                ),
                "task_name": (
                    list(names),
                    {"default": names[0] if names else "Photorealism Caption", "tooltip": "Caption preset (system prompt)."},
                ),
                "user_prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Optional extra instructions."},
                ),
            },
            "optional": {
                "options": (
                    "LMSTUDIO_OPTIONS",
                    {"tooltip": "Override generation options."},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("captions", "written_caption_paths", "result")
    CATEGORY = "LM Studio"
    FUNCTION = "caption_dataset"
    OUTPUT_IS_LIST = (True, True, False)

    def caption_dataset(self, model: Dict[str, Any], dataset: Dict[str, Any], task_name: str, user_prompt: str, options: Optional[Dict[str, Any]] = None):
        model_id = model.get("model_id", "")
        temperature = float(model.get("temperature", 0.2))
        max_tokens = int(model.get("max_tokens", 512))
        seed = model.get("seed", None)
        unload = model.get("unload", False)

        system_text = WASLMStudioCaption.cached_tasks.get(task_name, task_name)
        meta_output = ""
        meta_copy = False
        meta_force_aspect = "none"
        meta_max_size = 1024
        meta_resize_mode = "none"
        try:
            meta_output = str(dataset.get("__output_path", "") or "").strip()
            meta_copy = bool(dataset.get("__copy_images", False))
            meta_force_aspect = str(dataset.get("__force_aspect", "none"))
            meta_max_size = int(dataset.get("__max_size", 1024))
            meta_resize_mode = str(dataset.get("__resize_mode", "none"))
        except Exception:
            meta_output = ""
            meta_copy = False
            meta_force_aspect = "none"
            meta_max_size = 1024
            meta_resize_mode = "none"

        written: List[str] = []
        captions: List[str] = []
        failed_count = 0
        skipped_count = 0
        model_handle = None
        try:
            model_handle = lms.llm(model_id) if model_id else lms.llm()
            image_paths = [k for k in dataset.keys() if not str(k).startswith("__")]
            pbar = ProgressBar(len(image_paths))
            for idx, p in enumerate(image_paths):
                try:
                    if not os.path.isfile(p):
                        skipped_count += 1
                        pbar.update_absolute(idx + 1)
                        continue
                    img_handle = lms.prepare_image(p)
                    chat = lms.Chat(system_text) if system_text else lms.Chat()
                    chat.add_user_message(user_prompt or "", images=[img_handle])
                    params: Dict[str, Any] = {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if seed is not None:
                        params["seed"] = seed
                    if options:
                        try:
                            for k, v in options.items():
                                params[k] = v
                        except Exception:
                            pass
                    pred = model_handle.respond(chat, config=_map_params(params))
                    text = str(pred)
                    captions.append(text.strip())
                    # Determine output directory
                    src_dir = os.path.dirname(p)
                    out_dir = meta_output if meta_output else src_dir
                    os.makedirs(out_dir, exist_ok=True)
                    base = os.path.splitext(os.path.basename(p))[0]
                    txt_path = os.path.join(out_dir, base + ".txt")
                    # Copy image if requested and output differs from source
                    if meta_copy and meta_output and os.path.realpath(out_dir) != os.path.realpath(src_dir):
                        dst_img = os.path.join(out_dir, os.path.basename(p))
                        if not os.path.exists(dst_img):
                            shutil.copy2(p, dst_img)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text.strip())
                    written.append(txt_path)
                    pbar.update_absolute(idx + 1)
                except Exception as ie:
                    print(f"Caption failed for {p}: {ie}")
                    failed_count += 1
                    pbar.update_absolute(idx + 1)
                    continue
        except Exception as e:
            print(f"Error in caption_dataset: {e}")
        finally:
            if unload and model_handle is not None:
                try:
                    model_handle.unload()
                except Exception:
                    pass

        # Build result summary
        total_images = len(image_paths)
        success_count = len(written)
        result_lines = [
            f"Dataset Captioning Complete",
            f"="*50,
            f"Total Images: {total_images}",
            f"Successfully Captioned: {success_count}",
            f"Failed: {failed_count}",
            f"Skipped: {skipped_count}",
            f"",
            f"Settings:",
            f"  Task: {task_name}",
            f"  Model: {model_id}",
            f"  Temperature: {temperature}",
            f"  Max Tokens: {max_tokens}",
            f"  Force Aspect: {meta_force_aspect}",
            f"  Max Size: {meta_max_size}",
            f"  Resize Mode: {meta_resize_mode}",
            f"  Copy Images: {meta_copy}",
        ]
        if meta_output:
            result_lines.append(f"  Output Path: {meta_output}")
        result_summary = "\n".join(result_lines)

        return (captions, written, result_summary)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WASLMStudioModel": WASLMStudioModel,
    "WASLMStudioQuery": WASLMStudioQuery,
    "WASLMStudioCaption": WASLMStudioCaption,
    "WASLMStudioChat": WASLMStudioChat,
    "WASLMStudioOptions": WASLMStudioOptions,
    "WASLoadImageDirectory": WASLoadImageDirectory,
    "WASLMStudioCaptionDataset": WASLMStudioCaptionDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASLMStudioModel": "LM Studio Model",
    "WASLMStudioQuery": "LM Studio Query",
    "WASLMStudioCaption": "LM Studio Easy-Caption",
    "WASLMStudioChat": "LM Studio Chat",
    "WASLMStudioOptions": "LM Studio Options",
    "WASLoadImageDirectory": "WAS Load Image Directory",
    "WASLMStudioCaptionDataset": "LM Studio Easy-Caption Dataset",
}