"""
Memory Reward Function for verl-based RL Training.

This module implements the reward computation for ADRPO training,
combining format validation, task reward (QA accuracy), and length penalty.

Reward Structure:
- Format check: Invalid format → reward = 0 (validity gate)
- Task reward: QA accuracy (0-1.0)
- Length penalty: Penalize deviation from expert length
- Final: task_reward * (1 - lambda * length_penalty)

Integration with verl:
- compute_score_batch: Main entry point for verl's BatchRewardManager
- Supports parallel processing of multiple rollouts
- Compatible with verl's n_resp_per_prompt > 1

Usage:
    In verl config, set:
    reward_model:
        custom_reward_function: training.reward_server.reward_function.compute_score_batch
"""

import json
import os
import sys
import time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 动态添加项目根目录到 sys.path，确保 verl 动态加载时能找到模块
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from .format_checker import FormatChecker
    from .attribution import compute_attribution_weights, set_attribution_weights, apply_verl_patch
except Exception:
    from training.reward_server.format_checker import FormatChecker
    from training.reward_server.attribution import (
        compute_attribution_weights,
        set_attribution_weights,
        apply_verl_patch,
    )

_VERL_PATCH_ATTEMPTED = False

def _ensure_verl_patch_applied() -> None:
    """Apply verl attribution patch lazily to avoid import-time torch loading."""
    global _VERL_PATCH_ATTEMPTED

    if _VERL_PATCH_ATTEMPTED:
        return

    _VERL_PATCH_ATTEMPTED = True
    apply_verl_patch()

# Default configuration (matches paper Appendix)
DEFAULT_REWARD_CONFIG = {
    # Task reward settings
    "task_reward_mode": "api",  # "api" | "inproc" | "simulate" | "none"
    "memory_api_url": "http://localhost:8765",
    "reward_max_concurrent": 4,

    # ── In-process mode (verl worker 内直接调外部 LLM API，零 HTTP) ─────────
    # Used when task_reward_mode == "inproc". Default values target an
    # OpenAI-compatible endpoint with deepseek-v4-pro for answer
    # generation, deepseek-v4-flash for LLM-judge, and text-embedding-v4 for retrieval.
    "inproc_api_key": "",   # filled from LLM_API_KEY env var
    "inproc_base_url": "",
    "inproc_answer_model": "deepseek-v4-pro",
    "inproc_judge_model": "deepseek-v4-flash",
    "inproc_embedding_model": "text-embedding-v4",
    "inproc_qa_per_session": 5,
    "inproc_max_concurrent": 8,
    
    # Length penalty settings (Section 3.4.1 in paper)
    "enable_length_penalty": True,
    "length_penalty_weight": 0.3,  # lambda (reduced from 0.8 for CMI+QA exploration)
    
    # Core memory thresholds (absolute value scheme)
    "core_length_start_penalty": 150,  # theta_min: start penalty
    "core_length_max_penalty": 400,    # theta_max: full penalty
    
    # Other memory thresholds (ratio scheme)
    "other_length_upper_tolerance": 1.3,  # gamma_u
    "other_length_lower_tolerance": 0.5,  # gamma_l
    "other_length_upper_max_ratio": 2.0,  # full penalty upper
    "other_length_lower_min_ratio": 0.2,  # full penalty lower
    "other_length_min_diff_tokens": 200,  # minimum difference to trigger penalty

    # Minimum content threshold (prevent response collapse)
    "min_content_tokens_threshold": 100,  # below this → full length penalty for non-core agents

    # Fallback for non-expert training (e.g. CMI mode without expert trajectories)
    # Budget is generous: episodic agents legitimately need ~50 ops × 80 tokens/op = 4000 tokens
    # for information-dense sessions. Hard cap at 6000 prevents unbounded growth.
    "other_length_no_expert_budget": 4000,  # free token budget per agent when no expert
    "other_length_no_expert_max": 6000,     # full penalty threshold when no expert
    # Lower bound for CMI mode: penalize too-short responses that lose information
    "other_length_no_expert_min_budget": 150,  # below this → linearly increasing penalty
    "other_length_no_expert_min_floor": 30,    # below this → full penalty (1.0)
    
    # Attribution settings (Section 3.4.2 in paper)
    "enable_attribution": True,
    "attribution_alpha": 4.0,  # alpha in paper
    
    # Sparse reward ablation
    "sparse_reward_enabled": False,
    "sparse_reward_drop_rate": 0.5,
    "sparse_reward_default": 0.0,
    
    # API settings
    "api_max_retries": 5,
    "api_timeout_seconds": 80,
    "api_retry_sleep_seconds": 5,
    
    # Reward logging
    "reward_log_dir": None,
}
_REWARD_CONFIG = None
_REWARD_CONFIG_MTIME = None
_REWARD_LOG_FILE = None
_SPARSE_REWARD_RNG = None
_TOKENIZER = None  # 用于计算token长度的tokenizer（懒加载）

def _resolve_tokenizer_path(model_path: str) -> str:
    """Resolve tokenizer path via ModelScope to avoid HuggingFace Hub access."""
    expanded_model_path = os.path.expanduser(model_path)
    if os.path.exists(expanded_model_path):
        return expanded_model_path

    if model_path.startswith(("http://", "https://")):
        raise ValueError(f"Tokenizer path must be a local path or ModelScope model id, got: {model_path}")

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Loading tokenizer from ModelScope requires `modelscope`. "
            "Install it or set tokenizer_path to a local directory."
        ) from exc

    cache_dir = os.getenv("MEMBUILDER_MODELSCOPE_CACHE") or os.getenv("MODELSCOPE_CACHE")
    if cache_dir:
        return snapshot_download(model_path, cache_dir=cache_dir)
    return snapshot_download(model_path)

def _get_tokenizer():
    """懒加载tokenizer，用于计算token长度"""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            config = _load_reward_config()
            model_path = os.getenv("MEMBUILDER_TOKENIZER_PATH") or config.get('tokenizer_path', 'Qwen/Qwen3-4B-Instruct-2507')
            local_model_path = _resolve_tokenizer_path(model_path)
            _TOKENIZER = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
            print(f"📏 Tokenizer加载成功: {model_path} -> {local_model_path}")
        except Exception as e:
            print(f"⚠️ Tokenizer加载失败，将使用字符数估算: {e}")
            _TOKENIZER = "fallback"
    return _TOKENIZER


def _count_tokens(text: str) -> int:
    """计算文本的token数量"""
    tokenizer = _get_tokenizer()
    if tokenizer == "fallback":
        return int(len(text) / 3.5)
    return len(tokenizer.encode(text))


def _load_reward_config() -> Dict[str, Any]:
    global _REWARD_CONFIG, _REWARD_CONFIG_MTIME

    # 配置文件路径（写死，避免Ray多机环境下环境变量传递问题）
    # 优先使用internal版本，不存在则fallback到公开版本
    internal_path = Path(__file__).parent / "reward_config_internal.json"
    public_path = Path(__file__).parent / "reward_config.json"
    config_path = internal_path if internal_path.exists() else public_path

    try:
        mtime = config_path.stat().st_mtime if config_path.exists() else None
    except Exception:
        mtime = None

    if (
        _REWARD_CONFIG is not None
        and _REWARD_CONFIG_MTIME is not None
        and mtime is not None
        and mtime <= _REWARD_CONFIG_MTIME
    ):
        return _REWARD_CONFIG

    cfg = DEFAULT_REWARD_CONFIG.copy()

    loaded: Dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                tmp = json.load(f)
            if isinstance(tmp, dict):
                loaded = tmp
                cfg.update(loaded)
        except Exception:
            loaded = {}

    # Legacy key mapping (legacy overrides defaults unless explicit new key provided in file)
    if "enable_expert_length_penalty" in loaded and "enable_length_penalty" not in loaded:
        cfg["enable_length_penalty"] = bool(loaded.get("enable_expert_length_penalty"))
    if "expert_length_penalty_weight" in loaded and "length_penalty_weight" not in loaded:
        cfg["length_penalty_weight"] = float(loaded.get("expert_length_penalty_weight"))

    if "enable_attribution_weighting" in loaded and "enable_attribution" not in loaded:
        cfg["enable_attribution"] = bool(loaded.get("enable_attribution_weighting"))
    if "attribution_boost_factor" in loaded and "attribution_alpha" not in loaded:
        cfg["attribution_alpha"] = float(loaded.get("attribution_boost_factor"))

    if "enable_sparse_reward_ablation" in loaded and "sparse_reward_enabled" not in loaded:
        cfg["sparse_reward_enabled"] = bool(loaded.get("enable_sparse_reward_ablation"))
    if "sparse_reward_default_reward" in loaded and "sparse_reward_default" not in loaded:
        cfg["sparse_reward_default"] = float(loaded.get("sparse_reward_default_reward"))
    if "sparse_default_reward" in loaded and "sparse_reward_default" not in loaded:
        cfg["sparse_reward_default"] = float(loaded.get("sparse_default_reward"))

    env_reward_max_concurrent = os.environ.get("REWARD_MAX_CONCURRENT")
    if env_reward_max_concurrent is not None:
        try:
            cfg["reward_max_concurrent"] = int(env_reward_max_concurrent)
        except Exception:
            pass

    # 环境变量优先级高于配置文件
    env_memory_api_url = os.environ.get("MEMBUILDER_MEMORY_API_URL") or os.environ.get("MEMBUILDER_REWARD_SERVER_URL")
    if env_memory_api_url:
        cfg["memory_api_url"] = str(env_memory_api_url)

    # task_reward_mode 环境变量覆盖（remote worker 内无法访问 localhost:8765，
    # 需要通过 MEMBUILDER_TASK_REWARD_MODE=simulate 一键退化）
    env_task_mode = os.environ.get("MEMBUILDER_TASK_REWARD_MODE")
    if env_task_mode:
        cfg["task_reward_mode"] = str(env_task_mode).strip().lower()

    # reward_mode：选择 task reward 的具体公式
    #   "membuilder_qa" — 默认。MemBuilder 论文的多 QA judge reward
    #                     （走 _compute_task_reward_inproc / _api / _simplified）。
    #   "qa"            — 纯 QA reward（与 membuilder_qa 等价的别名），明确语义为
    #                     "完全跳过 CMI 计算"，用于 QA-only 消融。efficiency 优于
    #                     cmi_plus_qa+CMI_WEIGHT=0（后者仍会走 CMI 计算流程）。
    #   "cmi"           — 条件互信息 reward（CMI-Mem, Our Method），
    #                     走 _compute_task_reward_cmi，无需 QA 评测。
    #   "cmi_plus_qa"   — CMI + QA 组合 reward（alpha * CMI + (1-alpha) * QA）。
    env_reward_mode = os.environ.get("MEMBUILDER_REWARD_MODE")
    if env_reward_mode:
        cfg["reward_mode"] = str(env_reward_mode).strip().lower()
    cfg.setdefault("reward_mode", "membuilder_qa")

    # inproc 模式专用环境变量（LLM 凭证 + 模型名）
    # 这些只在 task_reward_mode == "inproc" 时被使用；环境变量优先级高于配置文件。
    for env_key, cfg_key in [
        ("LLM_API_KEY",                      "inproc_api_key"),
        ("LLM_BASE_URL",                     "inproc_base_url"),
        ("MEMBUILDER_INPROC_ANSWER_MODEL",    "inproc_answer_model"),
        ("MEMBUILDER_INPROC_JUDGE_MODEL",     "inproc_judge_model"),
        ("MEMBUILDER_INPROC_EMBEDDING_MODEL", "inproc_embedding_model"),
    ]:
        val = os.environ.get(env_key)
        if val:
            cfg[cfg_key] = str(val).strip()

    # 🔍 强制打印配置（首次加载时）
    if _REWARD_CONFIG is None:
        print(f"\n🔧 [CONFIG] Reward配置加载:")
        print(f"   task_reward_mode: {cfg.get('task_reward_mode')}")
        print(f"   memory_api_url: {cfg.get('memory_api_url')}")
        print(f"   env MEMBUILDER_MEMORY_API_URL: {os.environ.get('MEMBUILDER_MEMORY_API_URL', 'NOT SET')}")
        print(f"   config_path: {config_path}")

    _REWARD_CONFIG = cfg
    _REWARD_CONFIG_MTIME = mtime
    return cfg


def _init_reward_log_file(log_dir: Optional[str]) -> Optional[str]:
    global _REWARD_LOG_FILE
    if not log_dir:
        return None
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        return None
    if _REWARD_LOG_FILE is None:
        try:
            import datetime

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            _REWARD_LOG_FILE = os.path.join(log_dir, f"reward_{ts}.jsonl")
        except Exception:
            return None
    return _REWARD_LOG_FILE


def _append_reward_log(path: str, records: List[Dict[str, Any]]) -> None:
    if not path or not records:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        return


def _to_python_type(obj: Any) -> Any:
    """将 numpy 类型转换为 Python 原生类型"""
    import numpy as np
    if obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_type(v) for v in obj]
    return obj


def _extract_ground_truth_config(reward_model: Any) -> Dict[str, Any]:
    """从 reward_model 提取配置，正确处理 numpy 类型和多种数据格式。

    支持的输入格式：
    1. dict（4-agent 分组数据）: {"conversation_id": ..., "ground_truth": [...], ...}
    2. str - JSON 序列化的 list/dict: '[{"agent_type": "core", ...}]' → 反序列化后提取
    3. str - Memory QA 简单答案: "$185" → {"ground_truth": "$185", "style": "rule"}
    4. list: [dict, ...] → 取第一个元素
    5. numpy 类型: 自动转换
    """
    import numpy as np
    
    rm = reward_model
    if hasattr(rm, "tolist"):
        rm = rm.tolist()
    if isinstance(rm, list):
        rm = rm[0] if len(rm) > 0 else {}
    if isinstance(rm, str):
        # 先尝试 JSON 反序列化（处理 parquet 中序列化的复杂结构）
        try:
            parsed = json.loads(rm)
            if isinstance(parsed, list):
                # 4-agent ground_truth: list of dicts with agent_type, expert_output, etc.
                if parsed and isinstance(parsed[0], dict) and "agent_type" in parsed[0]:
                    # 从第一个元素（current agent）提取配置，保留完整的 ground_truth 列表
                    first = parsed[0]
                    return {
                        "conversation_id": first.get("conversation_id", "unknown"),
                        "session_index": first.get("session_index", -1),
                        "agent_type": first.get("agent_type", "unknown"),
                        "expert_output": first.get("expert_output", {}),
                        "ground_truth": parsed,
                        "qa_questions": first.get("qa_questions", []),
                        "state_before_path": first.get("state_before_path", ""),
                        "new_session_messages": first.get("new_session_messages", []),
                        "style": "custom",
                    }
                rm = parsed[0] if parsed else {}
            elif isinstance(parsed, dict):
                rm = parsed
            else:
                # 简单值（不太可能）
                return {"ground_truth": rm, "style": "rule"}
        except (json.JSONDecodeError, TypeError):
            # 不是 JSON，当作 Memory QA 简单答案
            return {"ground_truth": rm, "style": "rule"}
    if not isinstance(rm, dict):
        return {}

    # 处理嵌套的 JSON 字符串字段（从 parquet 序列化来的）
    for key in ["ground_truth", "expert_output", "qa_questions"]:
        val = rm.get(key)
        if isinstance(val, str):
            try:
                rm[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass

    # 处理 ground_truth（可能是 numpy.ndarray）
    gt = rm.get("ground_truth")
    if isinstance(gt, np.ndarray):
        gt = gt.tolist()
        rm["ground_truth"] = gt
    if isinstance(gt, list) and len(gt) > 0 and isinstance(gt[0], dict):
        # 返回整个 rm，但确保 qa_questions 等字段被正确转换
        pass  # 不再返回 gt[0]，因为顶层 rm 已有完整信息

    # 转换所有 numpy 类型为 Python 原生类型
    return _to_python_type(rm)


def compute_memory_reward(
    model_outputs: Dict[str, str],
    qa_pairs: List[Dict],
    state_before: Dict,
    expert_lengths: Optional[Dict] = None,
    answer_func=None,
    judge_func=None,
    config: Optional[Dict] = None
) -> Tuple[float, Dict]:
    """
    Compute reward for a single rollout.
    
    Args:
        model_outputs: Dict mapping agent type to raw model output string
                      e.g., {'core': '{"operation": "APPEND", ...}', ...}
        qa_pairs: List of QA dicts for task reward
        state_before: Memory state before this session
        expert_lengths: Optional expert memory lengths for length penalty
        answer_func: Function(question, memory_bank) -> (answer, retrieved_memories)
        judge_func: Function(question, gold, generated) -> 0 or 1
        config: Optional config overrides
    
    Returns:
        Tuple of (reward, info_dict)
    """
    cfg = {**DEFAULT_REWARD_CONFIG, **(config or {})}
    format_checker = FormatChecker()
    
    info = {
        "format_valid": True,
        "task_reward": 0.0,
        "length_penalty": 0.0,
        "retrieval_counts": {},
        "attribution_weights": {}
    }
    
    # Step 1: Format validation
    parsed_outputs = {}
    for agent_type in ['core', 'episodic', 'semantic', 'procedural']:
        output = model_outputs.get(agent_type, "")
        is_valid, parsed = format_checker.check_format(output, agent_type)
        
        if not is_valid:
            info["format_valid"] = False
            info["invalid_agent"] = agent_type
            return 0.0, info
        
        parsed_outputs[agent_type] = parsed
    
    # Step 2: Build memory bank from parsed outputs
    memory_bank = _build_memory_bank(parsed_outputs, state_before)
    
    # Step 3: Compute task reward (QA accuracy)
    if qa_pairs and answer_func and judge_func:
        correct = 0
        retrieval_counts = {"episodic": 0, "semantic": 0, "procedural": 0}
        
        for qa in qa_pairs:
            question = qa["question"]
            gold_answer = qa["answer"]
            
            generated_answer, retrieved = answer_func(question, memory_bank)
            
            # Count retrievals by type
            for mem in retrieved:
                for mem_type in retrieval_counts:
                    if f"[{mem_type.upper()}]" in mem:
                        retrieval_counts[mem_type] += 1
                        break
            
            correct += judge_func(question, gold_answer, generated_answer)
        
        info["task_reward"] = correct / len(qa_pairs) if qa_pairs else 0.0
        info["retrieval_counts"] = retrieval_counts
        info["correct"] = correct
        info["total"] = len(qa_pairs)
    
    # Step 4: Compute length penalty
    if cfg["enable_length_penalty"] and expert_lengths:
        info["length_penalty"] = _compute_length_penalty(
            memory_bank, expert_lengths, cfg
        )
    
    # Step 5: Compute attribution weights
    if cfg["enable_attribution"]:
        info["attribution_weights"] = compute_attribution_weights(
            info.get("retrieval_counts", {}),
            alpha=cfg["attribution_alpha"]
        )
    
    # Step 6: Final reward
    task_r = info["task_reward"]
    len_pen = info["length_penalty"]
    lambda_l = cfg["length_penalty_weight"]
    
    reward = task_r * (1.0 - lambda_l * len_pen)
    
    return reward, info


def _build_memory_bank(parsed_outputs: Dict, state_before: Dict) -> Dict:
    """Build memory bank from parsed outputs and previous state."""
    memory_bank = {
        "core": state_before.get("core_memory", ""),
        "episodic": list(state_before.get("episodic", [])),
        "semantic": list(state_before.get("semantic", [])),
        "procedural": list(state_before.get("procedural", []))
    }
    
    # Apply core operation
    core_parsed = parsed_outputs.get("core", {})
    if core_parsed:
        op = core_parsed.get("operation", "").upper()
        if op == "APPEND":
            content = core_parsed.get("content", "")
            if content:
                memory_bank["core"] = memory_bank["core"] + "\n" + content
        elif op == "REPLACE":
            old = core_parsed.get("old_text", "")
            new = core_parsed.get("new_text", "")
            if old and new:
                memory_bank["core"] = memory_bank["core"].replace(old, new)
        elif op == "REWRITE":
            content = core_parsed.get("content", "")
            if content:
                memory_bank["core"] = content
    
    # Apply memory operations
    for agent_type in ["episodic", "semantic", "procedural"]:
        parsed = parsed_outputs.get(agent_type, {})
        operations = parsed.get("operations", [])
        
        prefix = f"[{agent_type.upper()}]"
        
        for op in operations:
            action = op.get("action", "").upper()
            
            if action == "ADD":
                memory = op.get("memory", "")
                if memory:
                    if not memory.startswith(prefix):
                        memory = f"{prefix} {memory}"
                    memory_bank[agent_type].append(memory)
            
            elif action in ["UPDATE", "MERGE"]:
                new_memory = op.get("new_memory", "")
                if new_memory:
                    if not new_memory.startswith(prefix):
                        new_memory = f"{prefix} {new_memory}"
                    memory_bank[agent_type].append(new_memory)
    
    return memory_bank


def _compute_length_penalty(
    memory_bank: Dict,
    expert_lengths: Dict,
    config: Dict
) -> float:
    """Compute length penalty based on deviation from expert lengths."""
    penalties = []
    
    # Core memory penalty
    core_len = len(memory_bank.get("core", ""))
    expert_core = expert_lengths.get("core", 0)
    
    # Penalize token increase beyond thresholds
    increase = max(0, core_len - expert_core)
    
    theta_min = config["core_length_start_penalty"]
    theta_max = config["core_length_max_penalty"]
    
    if increase > theta_max:
        penalties.append(1.0)
    elif increase > theta_min:
        penalties.append((increase - theta_min) / (theta_max - theta_min))
    else:
        penalties.append(0.0)
    
    # Other memory penalty
    gamma_l = config["other_length_lower_tolerance"]
    gamma_u = config["other_length_upper_tolerance"]
    
    for mem_type in ["episodic", "semantic", "procedural"]:
        current = sum(len(m) for m in memory_bank.get(mem_type, []))
        expert = expert_lengths.get(mem_type, current)
        
        if expert == 0:
            penalties.append(0.0)
            continue
        
        ratio = current / expert
        
        if gamma_l <= ratio <= gamma_u:
            penalties.append(0.0)
        elif ratio > gamma_u:
            pen = min(1.0, (ratio - gamma_u) / (2.0 - gamma_u))
            penalties.append(pen)
        else:
            pen = min(1.0, (gamma_l - ratio) / gamma_l)
            penalties.append(pen)
    
    return sum(penalties) / len(penalties) if penalties else 0.0


# ============================================================================
# Batch Version: Main entry for verl's BatchRewardManager
# ============================================================================

def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[Dict[str, Any]],
    **kwargs
) -> List[float]:
    """
    Main batch reward computation for verl's BatchRewardManager.
    
    This function processes all responses in a batch, groups them by session,
    and computes rewards with format validation, task reward, and length penalty.
    
    Args:
        data_sources: Data source identifiers (all should be "memory_management")
        solution_strs: Model outputs (N*4 responses for N sessions × 4 agents)
        ground_truths: Ground truth data with QA pairs and expert outputs
        extra_infos: Additional info (unused)
        **kwargs: Additional parameters
    
    Returns:
        List[float]: Rewards in same order as inputs
    """
    _ensure_verl_patch_applied()

    start_time = time.time()
    n_samples = len(solution_strs)
    print(f"\n{'='*60}")
    print(f"🚀 Batch Reward计算开始: {n_samples}个samples")
    print(f"{'='*60}")
    
    # 调试模式
    DEBUG_REWARD = os.environ.get("DEBUG_REWARD", "0") == "1"

    config_path_override = kwargs.get("config_path") or kwargs.get("reward_config_path")
    if config_path_override:
        try:
            os.environ["MEMBUILDER_REWARD_CONFIG_PATH"] = str(config_path_override)
        except Exception:
            pass
        global _REWARD_CONFIG, _REWARD_CONFIG_MTIME
        _REWARD_CONFIG = None
        _REWARD_CONFIG_MTIME = None

    # Load config (supports hot reload via mtime cache inside _load_reward_config)
    config = _load_reward_config()

    api_url_override = kwargs.get("memory_api_url") or kwargs.get("reward_server_url")
    if api_url_override:
        config["memory_api_url"] = str(api_url_override)
    
    if DEBUG_REWARD:
        print(f"[DEBUG] task_reward_mode: {config.get('task_reward_mode')}")
        print(f"[DEBUG] memory_api_url: {config.get('memory_api_url')}")
    
    # Group by session
    from collections import defaultdict
    session_groups = defaultdict(dict)
    occurrence_count = defaultdict(int)
    
    for i, (response, reward_model) in enumerate(zip(solution_strs, ground_truths)):
        if reward_model is None:
            continue

        rm = _extract_ground_truth_config(reward_model)
        if not rm:
            continue

        conv_id = rm.get("conversation_id", "unknown")
        sess_idx = rm.get("session_index", -1)
        agent_type = rm.get("agent_type", "unknown")
        base_session_id = f"{conv_id}_sess{sess_idx}"
        
        # Handle multiple rollouts per session
        key = (base_session_id, agent_type)
        rollout_idx = occurrence_count[key]
        occurrence_count[key] += 1
        
        session_group_id = f"{base_session_id}_r{rollout_idx}"
        
        session_groups[session_group_id][agent_type] = {
            'index': i,
            'response': response,
            'ground_truth': rm
        }
    
    # 检测n_resp_per_prompt（与老代码一致）
    if occurrence_count:
        n_resp_detected = max(occurrence_count.values())
        print(f"📊 检测到 n_resp_per_prompt={n_resp_detected}")
    
    print(f"📦 分组完成: {len(session_groups)}个session-rollout组")
    for sid, agents in list(session_groups.items())[:5]:  # 只打印前5个
        print(f"   {sid}: {list(agents.keys())}")
    if len(session_groups) > 5:
        print(f"   ... 共{len(session_groups)}组")
    
    # 🔍 强制打印第一个sample的response（调试用）
    first_sid = list(session_groups.keys())[0] if session_groups else None
    if first_sid:
        first_agents = session_groups[first_sid]
        print(f"\n🔍 [调试] 第一个session {first_sid} 的response预览:")
        for agent_type, data in first_agents.items():
            resp_preview = data['response'][:300].replace('\n', '\\n')
            print(f"   {agent_type}: {resp_preview}...")
    
    # Initialize rewards
    rewards = [0.0] * n_samples
    format_checker = FormatChecker()
    weights_for_batch = {i: 1.0 for i in range(n_samples)}

    if not session_groups:
        print("No valid session groups formed from ground_truths; returning zero rewards")
        return rewards
    
    # Process each session
    def process_session(session_id: str, agents_data: Dict) -> Dict[str, Dict]:
        """Process single session, return {agent_type: reward_info}"""
        session_rewards = {}
        expected_agents = ['core', 'episodic', 'semantic', 'procedural']
        available_agents = list(agents_data.keys())
        is_complete = all(a in available_agents for a in expected_agents)
        
        # 与老代码一致：打印session处理日志
        print(f"  📍 处理session: {session_id} (agents: {available_agents}, complete: {is_complete})")
        
        # Format validation and parsing
        parsed_responses = {}
        format_results = {}
        is_first_rollout = session_id.endswith('_r0')

        # Detect extracted mode from ground_truth
        sample_gt_for_mode = agents_data.get("core", {}).get("ground_truth", {})
        is_extracted_mode = bool(sample_gt_for_mode.get("extracted_mode", False))
        extracted_candidates = sample_gt_for_mode.get("extracted_candidates", {})

        for agent_type, data in agents_data.items():
            response = data['response']
            is_valid, parsed = format_checker.check_format(
                response, agent_type, extracted_mode=is_extracted_mode)
            
            # In extracted mode, match model-generated text content back to
            # pre-extracted candidates. This ensures the reward only credits
            # content that originates from the external extraction model.
            if is_valid and parsed and is_extracted_mode:
                agent_candidates = extracted_candidates.get(agent_type, [])
                parsed = _match_to_extracted_candidates(parsed, agent_type, agent_candidates)

            parsed_responses[agent_type] = parsed
            format_results[agent_type] = is_valid
            
            # 🔍 与老代码一致：打印第一个rollout的解析结果
            if is_first_rollout:
                if parsed:
                    inner = parsed.get(agent_type, parsed)
                    ops = inner.get('operations', [])
                    print(f"    🔍 {agent_type}: format={'✓' if is_valid else '✗'}, ops_count={len(ops)}, parsed={str(parsed)[:150]}...")
                else:
                    print(f"    🔍 {agent_type}: format={'✓' if is_valid else '✗'}, 解析失败")
            
            if DEBUG_REWARD:
                print(f"[DEBUG] {session_id}/{agent_type}: format_valid={is_valid}, response_len={len(response)}")
        
        # Compute task reward if complete
        task_reward = 0.0
        dominant_agent = None
        sparse_dropped = False
        _session_cmi_reward = None
        _session_qa_reward = None
 
        sparse_enabled = bool(config.get("sparse_reward_enabled", False))
        sparse_drop_rate = float(config.get("sparse_reward_drop_rate", 0.0))
        sparse_default_reward = float(config.get("sparse_reward_default", 1.0))
        
        if is_complete and any(parsed_responses.values()):
            if sparse_enabled and sparse_drop_rate > 0:
                global _SPARSE_REWARD_RNG
                if _SPARSE_REWARD_RNG is None:
                    import random
 
                    seed = int(config.get("sparse_reward_seed", 42))
                    _SPARSE_REWARD_RNG = random.Random(seed)
                if _SPARSE_REWARD_RNG.random() < sparse_drop_rate:
                    task_reward = sparse_default_reward
                    dominant_agent = None
                    sparse_dropped = True
 
            sample_gt = agents_data.get("core", {}).get("ground_truth", {})
            qa_questions = sample_gt.get("qa_questions", [])
            state_before_path = sample_gt.get("state_before_path", "")
            
            # 🔍 强制调试：打印第一个 session 的关键信息
            if session_id.endswith("_r0") and "sess0" in session_id:
                print(f"\n🔍 [FORCE DEBUG] Session: {session_id}")
                print(f"   sample_gt keys: {list(sample_gt.keys())}")
                print(f"   qa_questions type: {type(qa_questions)}, len: {len(qa_questions) if hasattr(qa_questions, '__len__') else 'N/A'}")
                print(f"   state_before_path: {state_before_path}")
                print(f"   task_reward_mode: {config['task_reward_mode']}")
                print(f"   memory_api_url: {config.get('memory_api_url', 'NOT SET')}")
                print(f"   is_complete: {is_complete}, sparse_dropped: {sparse_dropped}")
                print(f"   any_parsed: {any(parsed_responses.values())}")
                # 检查条件
                cond1 = not sparse_dropped
                cond2 = config["task_reward_mode"] == "api"
                cond3 = bool(qa_questions) if not hasattr(qa_questions, '__len__') else len(qa_questions) > 0
                reward_mode_for_debug = str(config.get("reward_mode", "membuilder_qa")).strip().lower()
                cond4 = config["task_reward_mode"] == "inproc" and reward_mode_for_debug == "cmi"
                print(f"   条件检查: not_sparse={cond1}, mode_api={cond2}, has_qa={cond3}, mode_inproc_cmi={cond4}")
                print(f"   → 是否调用API: {cond1 and cond2 and cond3}")
                print(f"   → 是否计算CMI: {cond1 and cond4}")
            
            if DEBUG_REWARD:
                print(f"[DEBUG] {session_id}: is_complete={is_complete}, qa_count={len(qa_questions)}, mode={config['task_reward_mode']}")
                print(f"[DEBUG] {session_id}: any_parsed={any(parsed_responses.values())}, sparse_dropped={sparse_dropped}")

            qa_list = list(qa_questions) if hasattr(qa_questions, '__iter__') and not isinstance(qa_questions, (str, dict)) else []
            task_reward_mode = config["task_reward_mode"].lower()
            reward_mode = str(config.get("reward_mode", "membuilder_qa")).strip().lower()
            # cmi 和 cmi_plus_qa 不强制要求预置 QA（cmi_plus_qa 在无 QA 时退化为纯 CMI）
            qa_required = reward_mode not in ("cmi", "cmi_plus_qa")
            has_required_qa = (not qa_required) or len(qa_list) > 0
            
            if (not sparse_dropped) and task_reward_mode == "api" and len(qa_list) > 0:
                # 调用Memory API Server计算真实Task Reward（与老代码一致）
                try:
                    task_result = _compute_task_reward_api(
                        parsed_responses, sample_gt, session_id, config
                    )
                    task_reward = float(task_result.get("task_reward", 0.0))
                    dominant_agent = task_result.get("dominant_agent")
                    print(f"    ✅ Task Reward (API): {task_reward:.3f}")
                except Exception as e:
                    print(f"    ❌ API调用失败: {str(e)[:50]}, reward=0 (no fallback)")
                    task_reward = 0.0
            elif (not sparse_dropped) and task_reward_mode == "simulate":
                task_reward = _compute_task_reward_simplified(parsed_responses)
                print(f"    🔄 Task Reward (模拟): {task_reward:.3f}")
            elif (not sparse_dropped) and task_reward_mode == "inproc" and has_required_qa:
                # In-process: build memory + retrieve + answer + judge in worker process,
                # LLM via OpenAI-compatible endpoint (default deepseek-v4-pro).
                #
                # reward_mode 分发：
                #   "cmi"           → CMI reward（条件互信息）
                #   "cmi_plus_qa"   → CMI + QA 组合 reward
                #   "qa"            → 纯 QA reward（显式跳过 CMI，与默认 else 等价但语义明确）
                #   其他（默认）    → MemBuilder 原版 multi-QA judge reward
                try:
                    if reward_mode == "cmi":
                        task_result = _compute_task_reward_cmi(
                            parsed_responses, sample_gt, session_id, config
                        )
                        task_reward = float(task_result.get("task_reward", 0.0))
                        dominant_agent = task_result.get("dominant_agent")
                    elif reward_mode == "cmi_plus_qa":
                        # Combined mode: alpha * CMI + (1-alpha) * QA
                        # QA component uses ALL available QA pairs and averages
                        # their scores for a stable, low-variance reward signal.
                        try:
                            from .cmi_config import load_cmi_config as _load_cmi_cfg
                        except ImportError:
                            from training.reward_server.cmi_config import load_cmi_config as _load_cmi_cfg
                        _cmi_cfg = _load_cmi_cfg()
                        _alpha = float(_cmi_cfg.get("cmi_weight", 0.5))
                        cmi_result = _compute_task_reward_cmi(
                            parsed_responses, sample_gt, session_id, config
                        )
                        _cmi_score = float(cmi_result.get("task_reward", 0.0))
                        if len(qa_list) > 0:
                            # Use ALL QA pairs for stable reward signal
                            qa_result = _compute_task_reward_inproc(
                                parsed_responses, sample_gt, session_id, config
                            )
                            qa_score = float(qa_result.get("task_reward", 0.0))
                            dominant_agent = qa_result.get("dominant_agent")
                        else:
                            raise RuntimeError(
                                f"[FATAL] cmi_plus_qa mode requires QA questions but session "
                                f"{session_id} has none. All sessions must have pre-generated "
                                f"QA pairs in the training data. Re-run prepare_rl_data.py "
                                f"with --add-qa to fix."
                            )
                        task_reward = _alpha * _cmi_score + (1 - _alpha) * qa_score
                        # Store decomposed rewards for per-session tracking
                        _session_cmi_reward = _cmi_score
                        _session_qa_reward = qa_score
                        print(f"    [OK] Task Reward (CMI+QA, alpha={_alpha:.2f}, n_qa={len(qa_list)}): {task_reward:.3f} (cmi={_cmi_score:.3f}, qa={qa_score:.3f})")
                    elif reward_mode == "qa":
                        # 纯 QA reward：完全跳过 CMI 计算（无 embedding/logprob/kernel_mi 调用）
                        task_result = _compute_task_reward_inproc(
                            parsed_responses, sample_gt, session_id, config
                        )
                        task_reward = float(task_result.get("task_reward", 0.0))
                        dominant_agent = task_result.get("dominant_agent")
                        print(f"    [OK] Task Reward (pure QA, no CMI): {task_reward:.3f}")
                    else:
                        task_result = _compute_task_reward_inproc(
                            parsed_responses, sample_gt, session_id, config
                        )
                        task_reward = float(task_result.get("task_reward", 0.0))
                        dominant_agent = task_result.get("dominant_agent")
                        print(f"    [OK] Task Reward (inproc LLM): {task_reward:.3f}")
                except Exception as e:
                    print(f"    [ERR] inproc failed ({type(e).__name__}: {str(e)[:80]}), reward=0 (no fallback)")
                    task_reward = 0.0
            elif task_reward_mode == "none":
                task_reward = 0.0
        else:
            # 不完整的session、所有agent都解析失败、或被sparse drop（与老代码一致）
            task_reward_mode = config["task_reward_mode"].lower()
            if task_reward_mode != "none" and not sparse_dropped:
                if not is_complete:
                    print(f"    ⚠️ Session不完整，Task Reward=0")
                elif not any(parsed_responses.values()):
                    print(f"    ⚠️ 所有agent解析失败，Task Reward=0")
        
        # Compute final rewards for each agent
        for agent_type, data in agents_data.items():
            format_pass = format_results.get(agent_type, False)
            length_penalty = 0.0
            
            if format_pass:
                total_reward = task_reward
                
                # Length penalty (optional)
                if config['enable_length_penalty'] and total_reward > 0:
                    length_penalty = _compute_agent_length_penalty(
                        data['response'], 
                        data['ground_truth'], 
                        agent_type, 
                        config
                    )
                    total_reward = total_reward * (1.0 - config['length_penalty_weight'] * length_penalty)
            else:
                total_reward = 0.0
            
            # Attribution weight
            attribution_weight = 1.0
            if config.get("enable_attribution") and agent_type != "core":
                if dominant_agent and agent_type == dominant_agent:
                    attribution_weight = float(config.get("attribution_alpha", 1.0))
            
            session_rewards[agent_type] = {
                'index': data['index'],
                'reward': total_reward,
                'format_pass': format_pass,
                'task_reward': task_reward if format_pass else 0.0,
                'length_penalty': length_penalty,
                'attribution_weight': attribution_weight,
                'dominant_agent': dominant_agent,
                'sparse_dropped': sparse_dropped,
                'cmi_reward': _session_cmi_reward,
                'qa_reward': _session_qa_reward,
                'agent_type': agent_type,
            }
        
        return session_rewards
    
    # Process sessions in parallel（与老代码一致）
    try:
        max_concurrent = int(config.get('reward_max_concurrent', 1) or 1)  # 默认串行，避免rate limit
    except Exception:
        max_concurrent = 1
    max_concurrent = max(1, max_concurrent)
    print(f"\n🔄 开始处理 {len(session_groups)} 个sessions (并发数: {max_concurrent})...")
    
    all_attribution_weights = {}
    all_length_penalties = []
    all_task_rewards = {}
    sparse_dropped_count = 0
    reward_records: List[Dict[str, Any]] = []
    
    with ThreadPoolExecutor(max_workers=min(max_concurrent, len(session_groups))) as executor:
        futures = {
            executor.submit(process_session, sid, agents): sid
            for sid, agents in session_groups.items()
        }
        
        for future in as_completed(futures):
            session_id = futures[future]
            try:
                session_rewards = future.result()
                for agent_type, reward_data in session_rewards.items():
                    idx = reward_data['index']
                    rewards[idx] = reward_data['reward']
                    format_status = "✓" if reward_data.get('format_pass', False) else "✗"
                    
                    # 收集长度惩罚
                    lp = reward_data.get('length_penalty')
                    if lp is not None:
                        all_length_penalties.append(lp)
                    
                    # 收集归因权重
                    attr_weight = reward_data.get('attribution_weight')
                    if attr_weight is not None:
                        all_attribution_weights[idx] = float(attr_weight)
                    
                    # 收集纯task reward
                    task_r = reward_data.get('task_reward', 0.0)
                    all_task_rewards[idx] = task_r
                    
                    # 统计sparse drop（只统计一次per session，用core agent）
                    if agent_type == 'core' and reward_data.get('sparse_dropped', False):
                        sparse_dropped_count += 1
                    
                    # 与老代码一致：详细的每个agent reward输出
                    lp_str = f", 📏{lp:.2f}" if lp is not None else ""
                    attr_str = f", 🎯{attr_weight:.1f}" if attr_weight is not None and attr_weight != 1.0 else ""
                    sparse_str = " [SPARSE]" if reward_data.get('sparse_dropped', False) else ""
                    print(f"    ✓ {session_id}/{agent_type}: {reward_data['reward']:.3f} "
                          f"(format={format_status}, task={task_r:.3f}{lp_str}{attr_str}{sparse_str})")

                    rm = None
                    try:
                        rm = session_groups.get(session_id, {}).get(agent_type, {}).get('ground_truth', {})
                    except Exception:
                        rm = {}

                    reward_records.append(
                        {
                            "index": idx,
                            "session_group_id": session_id,
                            "conversation_id": rm.get("conversation_id"),
                            "session_index": rm.get("session_index"),
                            "agent_type": rm.get("agent_type", agent_type),
                            "reward": reward_data.get("reward"),
                            "task_reward": task_r,
                            "length_penalty": lp,
                            "format_pass": reward_data.get("format_pass"),
                            "attribution_weight": attr_weight,
                            "dominant_agent": reward_data.get("dominant_agent"),
                            "sparse_dropped": reward_data.get("sparse_dropped"),
                            "cmi_reward": reward_data.get("cmi_reward"),
                            "qa_reward": reward_data.get("qa_reward"),
                        }
                    )
            except Exception as e:
                print(f"    ❌ Session {session_id} 处理失败: {str(e)[:100]}")

    # ── Cross-session duplicate penalty ────────────────────────────────────
    # Skip in extracted mode: candidates are per-session, cross-session
    # duplication is impossible when model only selects from pre-extracted items.
    _any_extracted = any(
        agents.get("core", {}).get("ground_truth", {}).get("extracted_mode", False)
        for agents in session_groups.values()
    )
    if not _any_extracted:
        try:
            from .cmi_config import load_cmi_config as _load_dedup_cfg
        except ImportError:
            from training.reward_server.cmi_config import load_cmi_config as _load_dedup_cfg
        _dedup_cfg = _load_dedup_cfg()
        if _dedup_cfg.get("cross_session_dedup_enabled", True):
            rewards = _apply_cross_session_dedup_penalty(
                rewards, solution_strs, ground_truths, session_groups, config,
                penalty_weight=float(_dedup_cfg.get("cross_session_dedup_weight", 0.5)),
            )

    # Set attribution weights for verl patch
    if all_attribution_weights and config.get("enable_attribution"):
        weights_for_batch.update(all_attribution_weights)
        set_attribution_weights(weights_for_batch)
        boosted_count = sum(1 for w in all_attribution_weights.values() if w > 1.0)
        print(f"    🎯 归因权重已设置: {len(all_attribution_weights)}个样本, {boosted_count}个提升")
        
    # 收集统计信息（与老代码一致）
    elapsed = time.time() - start_time
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    
    # 长度惩罚统计
    all_length_penalties = [r.get('length_penalty', 0) for r in reward_records if r.get('length_penalty') is not None]
    if all_length_penalties:
        avg_lp = sum(all_length_penalties) / len(all_length_penalties)
        nonzero_lps = [lp for lp in all_length_penalties if lp > 0]
        nonzero_rate = len(nonzero_lps) / len(all_length_penalties) * 100
        avg_nonzero_lp = sum(nonzero_lps) / len(nonzero_lps) if nonzero_lps else 0.0
    else:
        avg_lp, nonzero_rate, avg_nonzero_lp = 0.0, 0.0, 0.0
    
    # Task reward 统计
    task_reward_values = [r.get('task_reward', 0) for r in reward_records if r.get('task_reward') is not None]
    task_reward_mean = sum(task_reward_values) / len(task_reward_values) if task_reward_values else 0.0
    
    # 稀疏奖励统计
    sparse_dropped_count = sum(1 for r in reward_records if r.get('sparse_dropped'))
    sparse_enabled = config.get('sparse_reward_enabled', False)
    sparse_drop_rate = config.get('sparse_reward_drop_rate', 0.0)

    # 保存详细日志（与老代码一致）
    log_path = _init_reward_log_file(config.get("reward_log_dir"))
    if log_path:
        _save_reward_details(
            save_path=log_path,
            solution_strs=solution_strs,
            ground_truths=ground_truths,
            rewards=rewards,
            task_rewards={r['index']: r.get('task_reward', 0) for r in reward_records if 'index' in r},
            session_groups=session_groups,
            elapsed=elapsed,
            sparse_dropped_count=sparse_dropped_count,
            sparse_config={'enabled': sparse_enabled, 'drop_rate': sparse_drop_rate}
        )
    
    # 详细统计输出（与老代码一致）
    print(f"\n{'='*60}")
    print(f"✅ Batch Reward计算完成")
    print(f"   总samples: {n_samples}")
    print(f"   平均reward: {avg_reward:.3f}")
    print(f"   平均Task Reward (QA准确率): {task_reward_mean:.3f}")
    if sparse_enabled:
        actual_drop_rate = sparse_dropped_count / len(session_groups) * 100 if session_groups else 0
        print(f"   🎲 稀疏奖励: {sparse_dropped_count}/{len(session_groups)}个session被drop "
              f"(配置{sparse_drop_rate:.0%}, 实际{actual_drop_rate:.1f}%)")
    if all_length_penalties:
        print(f"   📏 长度惩罚: 平均{avg_lp:.3f}, 触发率{nonzero_rate:.1f}%, 触发时平均{avg_nonzero_lp:.3f}")
    if all_attribution_weights:
        boosted_count = sum(1 for w in all_attribution_weights.values() if w > 1.0)
        print(f"   🎯 归因权重: {boosted_count}/{len(all_attribution_weights)}个提升")
    print(f"   耗时: {elapsed:.2f}秒")
    print(f"{'='*60}\n")

    # Build per-sample dicts so BatchRewardManager propagates decomposed
    # reward signals into reward_extra_info for ml_tracker logging.
    # Index reward_records by sample index for O(1) lookup.
    record_by_index = {r['index']: r for r in reward_records if 'index' in r}
    
    # Map agent_type to integer for downstream grouping (avoids string in tensor batch)
    _AGENT_TYPE_TO_ID = {"core": 0, "episodic": 1, "semantic": 2, "procedural": 3}
    
    results = []
    for i in range(n_samples):
        rec = record_by_index.get(i, {})
        task_reward = float(rec.get("task_reward", 0.0))
        cmi_reward = rec.get("cmi_reward")
        qa_reward = rec.get("qa_reward")
        agent_type = rec.get("agent_type", "unknown")
        final_reward = float(rewards[i])
        results.append({
            "score": final_reward,
            "acc": task_reward,
            "reward/task_reward": task_reward,
            "reward/format_pass": float(rec.get("format_pass", False)),
            "reward/length_penalty": float(rec.get("length_penalty", 0.0) or 0.0),
            "reward/attribution_weight": float(rec.get("attribution_weight", 1.0) or 1.0),
            "reward/sparse_dropped": float(rec.get("sparse_dropped", False)),
            # Decomposed reward signals for monitoring
            "reward/cmi_reward": float(cmi_reward) if cmi_reward is not None else 0.0,
            "reward/qa_reward": float(qa_reward) if qa_reward is not None else 0.0,
            "reward/total_reward": final_reward,
            # Agent type id for per-agent grouping in metrics
            "reward/agent_type_id": float(_AGENT_TYPE_TO_ID.get(agent_type, -1)),
        })

    return results


def _compute_task_reward_api(
    parsed_responses: Dict,
    ground_truth: Dict,
    session_id: str,
    config: Dict
) -> Dict[str, Any]:
    """Compute task reward by calling Memory API Server."""
    import requests
    
    DEBUG_REWARD = os.environ.get("DEBUG_REWARD", "0") == "1"
    
    api_url = config.get('memory_api_url', 'http://localhost:8765')
    qa_questions = ground_truth.get('qa_questions', [])[:5]  # Limit to 5
    
    if DEBUG_REWARD:
        print(f"[DEBUG] _compute_task_reward_api called for {session_id}")
        print(f"[DEBUG]   api_url: {api_url}")
        print(f"[DEBUG]   qa_questions: {len(qa_questions)}")
        print(f"[DEBUG]   state_before_path: {ground_truth.get('state_before_path', '')}")
    
    if not qa_questions:
        if DEBUG_REWARD:
            print(f"[DEBUG]   -> No QA questions, reward=0 (no fallback)")
        return {"task_reward": 0.0, "dominant_agent": None}
    
    # Prepare request
    memory_actions = []
    for agent_type in ['core', 'episodic', 'semantic', 'procedural']:
        parsed = parsed_responses.get(agent_type)
        if parsed:
            action_data = parsed.get(agent_type, parsed)
        else:
            action_data = {'operations': []} if agent_type != 'core' else {'operation': 'SKIP'}
        memory_actions.append({'agent_type': agent_type, 'action': action_data})
        if DEBUG_REWARD:
            if agent_type == 'core':
                print(f"[DEBUG]   {agent_type}: op={action_data.get('operation', 'N/A')}")
            else:
                ops = action_data.get('operations', [])
                print(f"[DEBUG]   {agent_type}: {len(ops)} ops")
    
    request_data = {
        'session_group_id': session_id,
        'memory_actions': memory_actions,
        'qa_pairs': [{'question': q['question'], 'answer': q['answer']} for q in qa_questions],
        'state_before_path': ground_truth.get('state_before_path', '')
    }
    
    max_retries = int(config.get("api_max_retries", 5))
    timeout_s = float(config.get("api_timeout_seconds", 80))
    retry_sleep = float(config.get("api_retry_sleep_seconds", 5))  # 与老代码一致：5秒
    
    if DEBUG_REWARD:
        print(f"[DEBUG]   Sending POST to {api_url}/build_and_qa ...")
 
    # 与老代码一致：range(max_retries + 1) = 最多 6 次尝试
    for attempt in range(max_retries + 1):
        try:
            if attempt == 0:
                print(f"      调用API: {len(request_data['memory_actions'])} agents, {len(request_data['qa_pairs'])} questions")
            else:
                print(f"      重试API ({attempt}/{max_retries})...")
            
            response = requests.post(f"{api_url}/build_and_qa", json=request_data, timeout=timeout_s)
            
            if DEBUG_REWARD:
                print(f"[DEBUG]   Attempt {attempt+1}: status={response.status_code}")
            
            if response.status_code != 200:
                if attempt < max_retries:
                    time.sleep(retry_sleep)
                    continue
                print(f"      ⚠️  API返回错误: {response.status_code}, reward=0 (no fallback)")
                return {"task_reward": 0.0, "dominant_agent": None}
            
            result = response.json()
            
            if not result.get('success'):
                error_msg = result.get('error', 'unknown')[:50]
                print(f"      ⚠️  API返回失败: {error_msg}, reward=0 (no fallback)")
                return {"task_reward": 0.0, "dominant_agent": None}
            
            # 提取评分结果（与老代码一致）
            avg_score = result.get('avg_score', 0.0)
            task_reward = result.get('task_reward', avg_score)
            dominant_agent = result.get('dominant_agent')
            processing_time = result.get('processing_time', 0)
            
            # 详细日志输出（与老代码一致）
            dominant_str = f", 主导={dominant_agent}" if dominant_agent else ""
            print(f"      ✅ API成功: 准确率{avg_score:.2%}, task_reward={task_reward:.3f}{dominant_str}, 耗时{processing_time:.1f}s")
            
            return {"task_reward": float(task_reward), "dominant_agent": dominant_agent}
                
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(retry_sleep)
                continue
            print(f"      ⚠️  API调用超时, reward=0 (no fallback)")
            return {"task_reward": 0.0, "dominant_agent": None}
        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_sleep)
                continue
            if DEBUG_REWARD:
                print(f"[DEBUG]   Attempt {attempt+1} failed: {str(e)[:100]}")
    
    print(f"      ⚠️  所有重试失败, reward=0 (no fallback)")
    return {"task_reward": 0.0, "dominant_agent": None}


# ============================================================================
# In-process reward mode (no HTTP server, direct LLM API calls)
# ============================================================================

_INPROC_ANSWER_CLIENT = None
_INPROC_JUDGE_CLIENT = None

def _get_inproc_answer_client(config: Dict):
    """Lazy-init OpenAI-compatible client for answer generation (inproc mode)."""
    global _INPROC_ANSWER_CLIENT
    if _INPROC_ANSWER_CLIENT is None:
        from llm_client import LLMClient
        _INPROC_ANSWER_CLIENT = LLMClient(
            api_key=config.get("inproc_api_key") or os.environ.get("LLM_API_KEY", ""),
            base_url=config["inproc_base_url"],
            model=config["inproc_answer_model"],
            embedding_model=config["inproc_embedding_model"],
        )
    return _INPROC_ANSWER_CLIENT

def _get_inproc_judge_client(config: Dict):
    """Lazy-init OpenAI-compatible client for LLM-judge (inproc mode)."""
    global _INPROC_JUDGE_CLIENT
    if _INPROC_JUDGE_CLIENT is None:
        from llm_client import LLMClient
        _INPROC_JUDGE_CLIENT = LLMClient(
            api_key=config.get("inproc_api_key") or os.environ.get("LLM_API_KEY", ""),
            base_url=config["inproc_base_url"],
            model=config["inproc_judge_model"],
            embedding_model=config["inproc_embedding_model"],
        )
    return _INPROC_JUDGE_CLIENT

def _compute_task_reward_inproc(
    parsed_responses: Dict,
    ground_truth: Dict,
    session_id: str,
    config: Dict,
) -> Dict[str, Any]:
    """In-process task reward computation.

    Builds a MemorySystem in-place from parsed memory actions, then runs
    retrieve + answer + LLM-judge for each QA. All LLM calls go through the
    OpenAI-compatible endpoint configured in `inproc_*` keys (default:
    `deepseek-v4-pro` + `text-embedding-v4`).

    No HTTP reward-server dependency, suitable for remote workers.
    """
    from pathlib import Path

    answer_client = _get_inproc_answer_client(config)
    judge_client = _get_inproc_judge_client(config)

    from training.reward_server.cmi_reward import get_cluster_embedding_func
    from memory_system import MemorySystem
    from eval.llm_judge import evaluate_answer
    from training.reward_server.server import _apply_memory_actions_list

    embedding_func = get_cluster_embedding_func()
    memory = MemorySystem(llm_client=answer_client, embedding_func=embedding_func)

    # Load state_before snapshot (so retrieval has prior-session context)
    state_before_path = ground_truth.get("state_before_path", "") or ""
    if state_before_path:
        if state_before_path.startswith("/"):
            candidates = [state_before_path]
        else:
            project_root = Path(__file__).resolve().parent.parent.parent
            candidates = [str(project_root / state_before_path)]
        for c in candidates:
            if Path(c).exists():
                try:
                    memory.load(c)
                    break
                except Exception:
                    pass

    # Apply parsed memory actions for the four agents
    actions_list = []
    for agent_type in ["core", "episodic", "semantic", "procedural"]:
        parsed = parsed_responses.get(agent_type)
        if parsed:
            action_data = parsed.get(agent_type, parsed)
        else:
            action_data = (
                {"operation": "SKIP"} if agent_type == "core" else {"operations": []}
            )
        actions_list.append({"agent_type": agent_type, "action": action_data})
    _apply_memory_actions_list(memory, actions_list)

    # Retrieve + answer + judge for each QA
    qa_per_session = int(config.get("inproc_qa_per_session", 5))
    qa_list = list(ground_truth.get("qa_questions", []))[:qa_per_session]
    if not qa_list:
        return {"task_reward": 0.0, "dominant_agent": None}

    # Lazy-import top-k from config (default 10, matches server.py)
    try:
        from config import QA_ANSWERING_TOP_K as _TOP_K
    except Exception:
        _TOP_K = 10

    # --- Memory-Guided RAG: load chunk data if configured ---
    try:
        from config import RAG_CHUNK_TYPES
    except Exception:
        RAG_CHUNK_TYPES = []

    session_chunks = []
    memory_chunk_mapping = {}
    if RAG_CHUNK_TYPES:
        import json as _json
        _sc = ground_truth.get("session_chunks", "[]")
        if isinstance(_sc, str):
            try:
                session_chunks = _json.loads(_sc)
            except Exception:
                session_chunks = []
        elif isinstance(_sc, list):
            session_chunks = _sc
        _mcm = ground_truth.get("memory_chunk_mapping", "{}")
        if isinstance(_mcm, str):
            try:
                memory_chunk_mapping = _json.loads(_mcm)
            except Exception:
                memory_chunk_mapping = {}
        elif isinstance(_mcm, dict):
            memory_chunk_mapping = _mcm

    correct = 0
    retrieval_counts = {"episodic": 0, "semantic": 0, "procedural": 0}
    judge_model = config["inproc_judge_model"]

    for qa in qa_list:
        question = qa.get("question", "")
        gold = qa.get("answer", "")
        try:
            search_result = memory.search(question, user_id=session_id, top_k=_TOP_K)
            retrieved = search_result.get("results", []) if isinstance(search_result, dict) else []
        except Exception:
            retrieved = []

        # Track retrieval counts and collect chunk indices for RAG
        rag_chunk_indices = set()
        for r in retrieved:
            mem = r.get("memory", "") if isinstance(r, dict) else str(r)
            for t in retrieval_counts:
                if f"[{t.upper()}]" in mem:
                    retrieval_counts[t] += 1
                    # If this type is in RAG_CHUNK_TYPES, find its chunk
                    if t in RAG_CHUNK_TYPES and session_chunks:
                        # Find item index by matching text in the memory store
                        _type_mapping = memory_chunk_mapping.get(t, [])
                        mem_text = mem.replace(f"[{t.upper()}] ", "")
                        # Try to find which extracted item this memory matches
                        _candidates = ground_truth.get("extracted_candidates", {})
                        if isinstance(_candidates, str):
                            try:
                                _candidates = _json.loads(_candidates)
                            except Exception:
                                _candidates = {}
                        _type_items = _candidates.get(t, [])
                        for item_idx, item_text in enumerate(_type_items):
                            if item_idx < len(_type_mapping) and str(item_text).strip() in mem_text:
                                chunk_idx = _type_mapping[item_idx]
                                if isinstance(chunk_idx, int) and 0 <= chunk_idx < len(session_chunks):
                                    rag_chunk_indices.add(chunk_idx)
                                break
                    break

        # Build chunk context string for RAG
        chunk_context = None
        if rag_chunk_indices and session_chunks:
            chunk_texts = []
            for ci in sorted(rag_chunk_indices):
                chunk = session_chunks[ci]
                text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                if text:
                    chunk_texts.append(f"[Conversation Segment {ci + 1}]\n{text}")
            if chunk_texts:
                chunk_context = "\n\n".join(chunk_texts)

        try:
            generated = memory.generate_answer(
                question, user_id=session_id, chunk_context=chunk_context
            )
        except Exception as e:
            generated = ""

        try:
            score = evaluate_answer(
                question, gold, generated,
                llm_client=judge_client, judge_model=judge_model,
            )
        except Exception:
            score = 0
        correct += int(score or 0)

    dominant = (
        max(retrieval_counts, key=retrieval_counts.get)
        if any(retrieval_counts.values())
        else None
    )
    return {
        "task_reward": correct / len(qa_list),
        "dominant_agent": dominant,
    }


def _compute_task_reward_cmi(
    parsed_responses: Dict,
    sample_gt: Dict,
    session_id: str,
    config: Dict,
) -> Dict[str, Any]:
    """Session-level CMI (Conditional Mutual Information) reward.

    Computes information gain from memory_before → memory_after conditioned
    on the current session context. Returns a scalar reward shared by all
    4 agent samples in the session (same interface as _compute_task_reward_inproc).

    Three computation methods are available (configured via CMI_METHOD env var):
        - "logprob"   : LLM log-probability difference
        - "embedding" : Embedding cosine similarity (fastest)
        - "entropy"   : Perplexity-based entropy reduction

    Two reward shaping strategies (configured via CMI_SHAPING env var):
        - "threshold" : Fixed interval [low, high]
        - "gaussian"  : Bell curve centered at optimal CMI

    Args:
        parsed_responses: Per-agent parsed actions ({'core': {...}, ...})
        sample_gt: Ground truth dict with state_before_path, new_session_messages
        session_id: Session identifier for logging
        config: Main reward configuration dict

    Returns:
        Dict[str, Any]: {"task_reward": float, "dominant_agent": None, ...}
    """
    from pathlib import Path
    try:
        from .cmi_reward import compute_cmi_reward
    except ImportError:
        from training.reward_server.cmi_reward import compute_cmi_reward

    # Load memory state before this session
    memory_state_before = {"core_memory": "", "memories": []}
    state_before_path = sample_gt.get("state_before_path", "") or ""

    if state_before_path:
        project_root = Path(__file__).resolve().parent.parent.parent
        if state_before_path.startswith("/"):
            candidates = [state_before_path]
        else:
            candidates = [str(project_root / state_before_path)]

        for c in candidates:
            p = Path(c)
            if p.exists():
                try:
                    # Load core memory
                    core_path = p / "core_memory.json"
                    if core_path.exists():
                        import json as _json
                        with open(core_path, "r", encoding="utf-8") as f:
                            core_data = _json.load(f)
                        memory_state_before["core_memory"] = core_data.get("human", "")

                    # Load vector store memories
                    payload_path = p / "vector_store" / "payload.json"
                    if payload_path.exists():
                        import json as _json
                        with open(payload_path, "r", encoding="utf-8") as f:
                            payload = _json.load(f)
                        memory_state_before["memories"] = payload.get("memories", [])
                    break
                except Exception as e:
                    if os.environ.get("CMI_DEBUG") == "1":
                        print(f"    [CMI] Failed to load state from {c}: {e}")

    # Get session messages for context
    session_messages = sample_gt.get("new_session_messages", [])
    if not session_messages:
        # Try to extract from ground_truth structure
        gt_list = sample_gt.get("ground_truth", [])
        if isinstance(gt_list, list) and gt_list:
            first_gt = gt_list[0] if isinstance(gt_list[0], dict) else {}
            session_messages = first_gt.get("new_session_messages", [])

    # Compute CMI reward
    # Pass extracted_mode flag so CMI anti-hacking penalties can be skipped
    cmi_config = dict(config)
    if sample_gt.get("extracted_mode", False):
        cmi_config["extracted_mode"] = True
    result = compute_cmi_reward(
        parsed_responses=parsed_responses,
        memory_state_before=memory_state_before,
        session_messages=session_messages,
        config=cmi_config,
    )
    task_reward = float(result.get("task_reward", 0.0))
    cmi_raw = result.get("cmi_raw", 0.0)
    cmi_method = result.get("cmi_method", "unknown")
    print(f"    [OK] Task Reward (CMI/{cmi_method}): {task_reward:.3f} "
          f"(raw_cmi={cmi_raw:.4f})")
    return result


def _compute_task_reward_simplified(parsed_responses: Dict) -> float:
    """Simplified task reward based on memory count (fallback)."""
    total_memories = 0
    
    for agent_type, parsed in parsed_responses.items():
        if not parsed or agent_type == 'core':
            continue
        
        agent_data = parsed.get(agent_type, parsed)
        operations = agent_data.get('operations', [])
        
        for op in operations:
            action = op.get('action', '').upper()
            if action in ['ADD', 'UPDATE', 'MERGE'] and (op.get('memory') or op.get('new_memory')):
                total_memories += 1
    
    # Each memory gives 0.05, max 0.8
    return min(0.8, total_memories * 0.05)


def _compute_agent_length_penalty(
    response: str,
    ground_truth: Dict,
    agent_type: str,
    config: Dict
) -> float:
    """Compute length penalty for a single agent."""
    # Extract content from response
    model_tokens = _count_content_tokens(response, agent_type)
    
    # ── Short response penalty (lower bound) ────────────────────────────────
    # Prevent response collapse: if content tokens are below a minimum
    # threshold, apply a penalty that scales linearly from 0 to 1.
    # This ensures the model cannot exploit "empty valid JSON" as a shortcut.
    min_content_tokens = int(config.get('min_content_tokens_threshold', 10))
    if model_tokens < min_content_tokens and agent_type != 'core':
        # Full penalty for completely empty content, linear ramp for near-empty
        if model_tokens <= 0:
            return 1.0
        return 1.0 - (model_tokens / min_content_tokens)

    if agent_type == 'core':
        theta_min = config['core_length_start_penalty']
        theta_max = config['core_length_max_penalty']
        
        if model_tokens <= theta_min:
            return 0.0
        elif model_tokens >= theta_max:
            return 1.0
        else:
            return (model_tokens - theta_min) / (theta_max - theta_min)
    else:
        # Ratio scheme: compare to expert (only for membuilder_qa mode)
        reward_mode = str(config.get("reward_mode", "membuilder_qa")).strip().lower()
        if reward_mode in ("cmi", "cmi_plus_qa"):
            # CMI modes never use expert trajectories for length penalty;
            # force fallback to absolute token budget regardless of data content.
            expert_tokens = 0
        else:
            expert_output = ground_truth.get('expert_output', {})
            expert_memories = expert_output.get('memories_added', [])
            expert_tokens = sum(len(str(m)) for m in expert_memories) if expert_memories else 0
        
        if expert_tokens == 0:
            # No expert data (e.g. CMI training without expert trajectories).
            # Fall back to absolute token thresholds to prevent both unbounded
            # growth AND response collapse (the latter causes memory loss).
            fallback_budget = int(config.get('other_length_no_expert_budget', 4000))
            fallback_max = int(config.get('other_length_no_expert_max', 6000))
            # Lower bound: penalize too-short responses that lose information
            min_budget = int(config.get('other_length_no_expert_min_budget', 150))
            min_floor = int(config.get('other_length_no_expert_min_floor', 30))
            if model_tokens <= min_floor:
                return 1.0  # Full penalty for near-empty responses
            elif model_tokens < min_budget:
                # Linear ramp from 1.0 (at min_floor) to 0.0 (at min_budget)
                return (min_budget - model_tokens) / (min_budget - min_floor)
            elif model_tokens <= fallback_budget:
                return 0.0
            elif model_tokens >= fallback_max:
                return 1.0
            else:
                return (model_tokens - fallback_budget) / (fallback_max - fallback_budget)
 
        min_diff = int(config.get('other_length_min_diff_tokens', 200))
        if abs(model_tokens - expert_tokens) < min_diff:
            return 0.0
        
        ratio = model_tokens / expert_tokens
        gamma_l = config['other_length_lower_tolerance']
        gamma_u = config['other_length_upper_tolerance']
        
        if gamma_l <= ratio <= gamma_u:
            return 0.0
        elif ratio > gamma_u:
            return min(1.0, (ratio - gamma_u) / (config['other_length_upper_max_ratio'] - gamma_u))
        else:
            return min(1.0, (gamma_l - ratio) / (gamma_l - config['other_length_lower_min_ratio']))


def _count_content_tokens(response: str, agent_type: str) -> int:
    """Count tokens in model response content."""
    import re
    
    # Remove <think> tags
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    
    # Remove markdown code blocks
    response = re.sub(r'^```(?:json)?\s*|\s*```$', '', response).strip()
    
    try:
        parsed = json.loads(response)
        
        if agent_type == 'core':
            content = parsed.get('content', '') or ''
            return len(content) // 4  # Approximate tokens
        else:
            agent_data = parsed.get(agent_type, parsed)
            operations = agent_data.get('operations', [])
            
            total = 0
            for op in operations:
                mem = op.get('memory', '') or op.get('new_memory', '')
                total += len(str(mem))
            return total // 4
    except:
        return len(response) // 4


def _match_to_extracted_candidates(parsed, agent_type, candidates):
    """Match model-generated text content to pre-extracted candidates.

    In extracted mode, the model outputs standard text format (content/memory
    fields) rather than item_id references. This function matches each piece
    of model-generated text to the most similar pre-extracted candidate using
    substring matching and fuzzy similarity, then replaces the model text with
    the canonical candidate text.

    This ensures:
    1. Reward only credits content originating from external extraction
    2. Model can use natural text output format (no SFT needed for item_id)
    3. Downstream CMI/QA reward sees clean, canonical memory text

    Args:
        parsed: Parsed model output dict (standard format from format_checker).
        agent_type: One of core/episodic/semantic/procedural.
        candidates: List of extracted memory item strings.

    Returns:
        Parsed dict with model text replaced by best-matching candidate text.
        Operations whose text doesn't match any candidate are kept as-is
        (the CMI reward will naturally score them based on information gain).
    """
    if not candidates:
        return parsed

    def _best_match(text):
        """Find the best matching candidate for a given text.

        Returns (matched_candidate_text, similarity_score).
        Uses a tiered matching strategy:
          1. Exact substring match (candidate in text or text in candidate)
          2. Token overlap ratio (Jaccard-like)
        """
        if not text or not text.strip():
            return text, 0.0

        text_stripped = text.strip()
        text_lower = text_stripped.lower()

        best_candidate = text_stripped
        best_score = 0.0

        for candidate in candidates:
            candidate_stripped = candidate.strip()
            if not candidate_stripped:
                continue
            candidate_lower = candidate_stripped.lower()

            # Tier 1: exact substring match
            if candidate_lower in text_lower or text_lower in candidate_lower:
                # Prefer the longer match
                score = min(len(candidate_stripped), len(text_stripped)) / max(len(candidate_stripped), len(text_stripped), 1)
                score = max(score, 0.8)  # substring match gets at least 0.8
                if score > best_score:
                    best_score = score
                    best_candidate = candidate_stripped
                continue

            # Tier 2: token overlap (Jaccard similarity)
            text_tokens = set(text_lower.split())
            cand_tokens = set(candidate_lower.split())
            if text_tokens and cand_tokens:
                intersection = text_tokens & cand_tokens
                union = text_tokens | cand_tokens
                jaccard = len(intersection) / len(union) if union else 0.0
                if jaccard > best_score:
                    best_score = jaccard
                    best_candidate = candidate_stripped

        return best_candidate, best_score

    # Matching threshold: below this, keep model's original text
    MATCH_THRESHOLD = 0.3

    if agent_type == "core":
        operation = parsed.get("operation", "").upper()

        if operation == "SKIP":
            return parsed

        if operation == "APPEND":
            content = parsed.get("content", "")
            matched, score = _best_match(content)
            if score >= MATCH_THRESHOLD:
                return {"operation": "APPEND", "content": matched}
            return parsed

        elif operation == "REPLACE":
            new_text = parsed.get("new_text", "")
            matched, score = _best_match(new_text)
            result = {
                "operation": "REPLACE",
                "old_text": parsed.get("old_text", ""),
                "new_text": matched if score >= MATCH_THRESHOLD else new_text,
            }
            return result

        elif operation == "REWRITE":
            content = parsed.get("content", "")
            matched, score = _best_match(content)
            if score >= MATCH_THRESHOLD:
                return {"operation": "REWRITE", "content": matched}
            return parsed

        return parsed

    # Memory agents (episodic/semantic/procedural)
    operations = parsed.get("operations", [])
    matched_ops = []

    for op in operations:
        action = op.get("action", "").upper()

        if action == "ADD":
            memory = op.get("memory", "")
            matched, score = _best_match(memory)
            matched_ops.append({
                "action": "ADD",
                "memory": matched if score >= MATCH_THRESHOLD else memory,
            })

        elif action == "UPDATE":
            new_memory = op.get("new_memory", "")
            matched, score = _best_match(new_memory)
            matched_ops.append({
                "action": "UPDATE",
                "old_memory": op.get("old_memory", ""),
                "new_memory": matched if score >= MATCH_THRESHOLD else new_memory,
            })

        elif action == "MERGE":
            new_memory = op.get("new_memory", "")
            matched, score = _best_match(new_memory)
            matched_ops.append({
                "action": "MERGE",
                "old_memories": op.get("old_memories", []),
                "new_memory": matched if score >= MATCH_THRESHOLD else new_memory,
            })

        elif action == "SKIP":
            matched_ops.append({
                "action": "SKIP",
                "reason": op.get("reason", "skipped by manager"),
            })

        else:
            matched_ops.append(op)

    return {"operations": matched_ops}


def _apply_cross_session_dedup_penalty(
    rewards: List[float],
    solution_strs: List[str],
    ground_truths: List[Any],
    session_groups: Dict,
    config: Dict,
    penalty_weight: float = 0.5,
) -> List[float]:
    """Apply cross-session duplicate penalty within each conversation.

    Groups sessions by conversation_id, extracts ADD memories from each
    session's model output, and penalizes sessions whose ADD memories
    duplicate those from other sessions in the same conversation.

    The penalty is multiplicative:
        reward *= (1 - penalty_weight * cross_dup_ratio)
    where cross_dup_ratio = fraction of this session's ADDs that appear
    in other sessions of the same conversation.

    Only applies to non-core agents (episodic/semantic/procedural) since
    core memory is a single document, not ADD-based.

    Args:
        rewards: Current rewards list (modified in-place and returned).
        solution_strs: Raw model outputs.
        ground_truths: Ground truth configs.
        session_groups: Session grouping from compute_score_batch.
        config: Reward config dict.
        penalty_weight: How strongly to penalize (0=off, 1=full zero-out).

    Returns:
        Modified rewards list.
    """
    import re
    from collections import defaultdict

    # Group session_groups by (conversation_id, rollout_idx)
    # session_group_id format: "{conv_id}_sess{idx}_r{rollout}"
    conv_rollout_sessions: Dict[str, List[tuple]] = defaultdict(list)
    for session_group_id, agents_data in session_groups.items():
        # Extract conv_id from session_group_id
        # Format: "conv_id_sessN_rM"
        parts = session_group_id.rsplit("_r", 1)
        if len(parts) != 2:
            continue
        base_id = parts[0]  # "conv_id_sessN"
        rollout_idx = parts[1]
        # Extract conv_id (everything before last _sessN)
        sess_parts = base_id.rsplit("_sess", 1)
        if len(sess_parts) != 2:
            continue
        conv_id = sess_parts[0]
        group_key = f"{conv_id}_r{rollout_idx}"
        conv_rollout_sessions[group_key].append((session_group_id, agents_data))

    if not conv_rollout_sessions:
        return rewards

    # For each conversation+rollout, collect ADD memories per session
    total_penalized = 0
    total_sessions_checked = 0

    for group_key, session_list in conv_rollout_sessions.items():
        if len(session_list) <= 1:
            continue  # Need at least 2 sessions to detect cross-session dups

        # Extract ADD memories from each session's responses
        # session_memories[session_group_id] = list of memory text fingerprints
        session_memories: Dict[str, List[str]] = {}
        session_indices: Dict[str, List[int]] = {}  # session_group_id → sample indices

        for session_group_id, agents_data in session_list:
            memories = []
            indices = []
            for agent_type, data in agents_data.items():
                if agent_type == "core":
                    continue  # Core is APPEND/REPLACE, not ADD-based
                indices.append(data['index'])
                response = data['response']
                # Extract ADD memories from raw response JSON
                extracted = _extract_add_memories_from_response(response, agent_type)
                memories.extend(extracted)
            session_memories[session_group_id] = memories
            session_indices[session_group_id] = indices

        # Build global memory counter across all sessions in this conv+rollout
        all_memories = []
        for mems in session_memories.values():
            all_memories.extend(mems)

        if not all_memories:
            continue

        from collections import Counter
        global_counter = Counter(all_memories)
        # Memories appearing in 2+ sessions are cross-session duplicates
        cross_dups = {mem for mem, count in global_counter.items() if count > 1}

        if not cross_dups:
            continue

        # Apply penalty to each session based on its duplicate ratio
        for session_group_id, mems in session_memories.items():
            if not mems:
                continue
            total_sessions_checked += 1
            dup_count = sum(1 for m in mems if m in cross_dups)
            dup_ratio = dup_count / len(mems)

            if dup_ratio <= 0:
                continue

            total_penalized += 1
            penalty = penalty_weight * dup_ratio
            multiplier = max(0.0, 1.0 - penalty)

            # Apply to all sample indices of this session
            for idx in session_indices.get(session_group_id, []):
                if idx < len(rewards):
                    rewards[idx] *= multiplier

    if total_penalized > 0:
        print(f"    🔁 跨session去重: {total_penalized}/{total_sessions_checked}个session被惩罚 "
              f"(weight={penalty_weight:.2f})")

    return rewards


def _extract_add_memories_from_response(response: str, agent_type: str) -> List[str]:
    """Extract ADD memory fingerprints from a raw model response string.

    Parses the JSON response and extracts memory text from ADD operations.
    Returns first-80-chars of each ADD memory as fingerprint for dedup.
    """
    import re

    # Strip <think> tags
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()

    # Strip markdown code blocks
    response = re.sub(r'^```(?:json)?\s*|\s*```$', '', response).strip()

    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return []

    # Navigate to operations list
    agent_data = parsed.get(agent_type, parsed) if isinstance(parsed, dict) else {}
    if not isinstance(agent_data, dict):
        return []
    operations = agent_data.get("operations", [])
    if not isinstance(operations, list):
        return []

    fingerprints = []
    for op in operations:
        if not isinstance(op, dict):
            continue
        action = str(op.get("action", "")).upper()
        if action == "ADD":
            memory = str(op.get("memory", "")).strip()
            if memory:
                # Use first 80 chars as fingerprint (robust to minor suffix diffs)
                fingerprints.append(memory[:80].lower())

    return fingerprints


def _save_reward_details(
    save_path: str,
    solution_strs: List[str],
    ground_truths: List[Any],
    rewards: List[float],
    task_rewards: Dict[int, float],
    session_groups: Dict,
    elapsed: float,
    sparse_dropped_count: int = 0,
    sparse_config: Dict = None
):
    """
    保存详细的reward信息到JSON文件（与老代码一致）
    """
    import datetime
    import numpy as np
    
    # 计算额外统计信息
    response_lengths = [_count_tokens(r) for r in solution_strs]
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    
    # 按 agent 类型统计
    agent_stats = {'core': [], 'episodic': [], 'semantic': [], 'procedural': []}
    format_pass_count = 0
    
    for i, (response, gt, reward) in enumerate(zip(solution_strs, ground_truths, rewards)):
        gt_dict = _extract_ground_truth_config(gt) if gt else {}
        agent_type = gt_dict.get('agent_type', 'unknown')
        if agent_type in agent_stats:
            agent_stats[agent_type].append({
                'reward': reward,
                'length': _count_tokens(response)
            })
        if reward > 0:
            format_pass_count += 1
    
    # 计算标准差
    reward_std = float(np.std(rewards)) if len(rewards) > 1 else 0.0
    length_std = float(np.std(response_lengths)) if len(response_lengths) > 1 else 0.0
    
    # 计算纯task reward统计
    task_reward_values = list(task_rewards.values()) if task_rewards else []
    task_reward_mean = sum(task_reward_values) / len(task_reward_values) if task_reward_values else 0.0
    task_reward_std = float(np.std(task_reward_values)) if len(task_reward_values) > 1 else 0.0
    
    # 构建详细记录
    batch_record = {
        'timestamp': datetime.datetime.now().isoformat(),
        'n_samples': len(solution_strs),
        'n_sessions': len(session_groups),
        'elapsed_seconds': elapsed,
        'avg_reward': sum(rewards) / len(rewards) if rewards else 0.0,
        'rewards_summary': {
            'min': min(rewards) if rewards else 0.0,
            'max': max(rewards) if rewards else 0.0,
            'mean': sum(rewards) / len(rewards) if rewards else 0.0,
            'std': reward_std,
        },
        'task_rewards_summary': {
            'min': min(task_reward_values) if task_reward_values else 0.0,
            'max': max(task_reward_values) if task_reward_values else 0.0,
            'mean': task_reward_mean,
            'std': task_reward_std,
            'accuracy': task_reward_mean,
        },
        'length_stats': {
            'avg_response_length': avg_response_length,
            'min_response_length': min(response_lengths) if response_lengths else 0,
            'max_response_length': max(response_lengths) if response_lengths else 0,
            'std_response_length': length_std,
        },
        'format_stats': {
            'format_pass_count': format_pass_count,
            'format_pass_rate': format_pass_count / len(rewards) if rewards else 0.0,
        },
        'sparse_reward_stats': {
            'enabled': sparse_config.get('enabled', False) if sparse_config else False,
            'drop_rate_config': sparse_config.get('drop_rate', 0.0) if sparse_config else 0.0,
            'dropped_sessions': sparse_dropped_count,
            'total_sessions': len(session_groups),
            'actual_drop_rate': sparse_dropped_count / len(session_groups) if session_groups else 0.0,
        },
        'agent_stats': {
            agent_type: {
                'count': len(stats),
                'avg_reward': sum(s['reward'] for s in stats) / len(stats) if stats else 0.0,
                'std_reward': float(np.std([s['reward'] for s in stats])) if len(stats) > 1 else 0.0,
                'avg_length': sum(s['length'] for s in stats) / len(stats) if stats else 0.0,
                'std_length': float(np.std([s['length'] for s in stats])) if len(stats) > 1 else 0.0,
                'format_pass_rate': sum(1 for s in stats if s['reward'] > 0) / len(stats) if stats else 0.0,
            }
            for agent_type, stats in agent_stats.items()
        },
        'samples': []
    }
    
    # 添加每个sample的详细信息
    for i, (response, gt, reward) in enumerate(zip(solution_strs, ground_truths, rewards)):
        gt_dict = _extract_ground_truth_config(gt) if gt else {}
        
        sample_info = {
            'index': i,
            'session_group_id': f"{gt_dict.get('conversation_id', 'unknown')}_sess{gt_dict.get('session_index', -1)}",
            'agent_type': gt_dict.get('agent_type', 'unknown'),
            'reward': reward,
            'task_reward': task_rewards.get(i, 0.0),
            'response_preview': response[:500] + '...' if len(response) > 500 else response,
        }
        batch_record['samples'].append(sample_info)
    
    # 追加写入JSONL文件
    try:
        with open(save_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(batch_record, ensure_ascii=False) + '\n')
        print(f"📝 Reward详情已保存到: {save_path}")
    except Exception as e:
        print(f"⚠️ 保存reward详情失败: {e}")
