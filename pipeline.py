import os
import json
import yaml
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from torch.utils.data import Dataset
except Exception:
    Dataset = None

from pydantic import BaseModel, ValidationError

try:
    # optional HF tokenizer / model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        import torch
    except Exception:
        torch = None
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


class VerificationResult(BaseModel):
    is_sufficient: bool
    sufficiency_score: float
    missing_evidence_type: str
    predicted_utility_loss: float
    diagnostic_reasoning: str


class SimpleTokenizer:
    """Fallback whitespace tokenizer for token counting."""

    def __init__(self):
        pass

    def encode(self, text: str) -> List[str]:
        if not text:
            return []
        return text.split()

    def decode(self, tokens: List[str]) -> str:
        return ' '.join(tokens)

    def __len__(self):
        return 0


class ModelWrapper:
    """Abstracted model wrapper. By default does simple echo / heuristic.

    Replace generate() with actual calls to local Qwen3-8B model in production.
    """

    def __init__(self, model_name_or_path: str, tokenizer_name: Optional[str] = None, load_model: bool = True):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = None
        self.model = None
        if HF_AVAILABLE and tokenizer_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            except Exception:
                self.tokenizer = None
        if HF_AVAILABLE and not self.tokenizer and model_name_or_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
            except Exception:
                self.tokenizer = None
        if not self.tokenizer:
            self.tokenizer = SimpleTokenizer()

        if HF_AVAILABLE and load_model and torch is not None:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                )
                self.model.eval()
            except Exception:
                self.model = None

        if load_model and self.model is None:
            raise RuntimeError(
                "Model loading failed. Ensure model path is correct and torch/transformers are installed."
            )

    def count_tokens(self, text: str) -> int:
        if hasattr(self.tokenizer, 'encode'):
            return len(self.tokenizer.encode(text))
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not hasattr(self.tokenizer, 'encode'):
            return ' '.join(text.split()[:max_tokens])
        tokens = self.tokenizer.encode(text)
        tokens = tokens[:max_tokens]
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(tokens)
        return ' '.join(str(t) for t in tokens)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if self.model is not None and HF_AVAILABLE and torch is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # return only the newly generated suffix when possible
            if text.startswith(prompt):
                return text[len(prompt):].strip()
            return text

        # Fallback heuristic for non-model environments
        if 'VerificationResult' in prompt or 'is_sufficient' in prompt:
            obj = {
                "is_sufficient": random.choice([True, False]),
                "sufficiency_score": round(random.random(), 3),
                "missing_evidence_type": random.choice(['Style', 'Recent_Preference', 'Hard_Constraint', 'Task_Memory', 'Preference_Conflict', 'None']),
                "predicted_utility_loss": round(random.random(), 3),
                "diagnostic_reasoning": "(synthetic) Based on compressed context and memory index analysis."
            }
            return json.dumps(obj)
        if 'Judge' in prompt or 'utility_loss' in prompt:
            score = round(random.random(), 3)
            return json.dumps({"utility_loss": score, "reason": "(synthetic) judge score"})
        tail = prompt[-min(200, len(prompt)) :]
        return "(model_output) " + tail

    def generate_json(self, prompt: str, max_tokens: int = 512) -> Any:
        raw = self.generate(prompt, max_tokens=max_tokens)
        return self._extract_json(raw)

    def _extract_json(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r'(\{.*\})', text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m = re.search(r'(\[.*\])', text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return {}


@dataclass
class MemoryIndex:
    index: List[Dict[str, Any]]
    by_id: Dict[int, Dict[str, Any]]
    by_tag: Dict[str, List[Dict[str, Any]]]


class MemoryIndexLoader:
    def __init__(self, path: str, auto_normalize: bool = True):
        self.path = path
        self.auto_normalize = auto_normalize
        self.index: List[Dict[str, Any]] = []
        self.by_id: Dict[int, Dict[str, Any]] = {}
        self.by_tag: Dict[str, List[Dict[str, Any]]] = {}
        self.by_user_key: Dict[str, MemoryIndex] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Memory index file not found: {self.path}")
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            # real lamp-summary format: {"train::id": "summary"}
            normalized = []
            for key, summary in data.items():
                if not isinstance(summary, str):
                    continue
                split, user_id = self._parse_user_key(key)
                entry = {
                    "id": int(user_id) if user_id is not None else -1,
                    "type": "Task_Memory",
                    "time": "",
                    "importance": "medium",
                    "privacy_level": "low",
                    "tag": summary,
                    "user_key": key,
                    "split": split or "unknown",
                }
                normalized.append(entry)
            # build per-user index
            for entry in normalized:
                key = entry.get("user_key", "unknown")
                self.by_user_key.setdefault(key, MemoryIndex([], {}, {}))
                self._insert_entry(self.by_user_key[key], entry)
            # also keep a global view
            self.index = normalized
            self.by_id = {int(e["id"]): e for e in normalized if isinstance(e.get("id"), int)}
            self.by_tag = {}
            for e in normalized:
                tag = e.get("tag", "ungrouped")
                self.by_tag.setdefault(tag, []).append(e)

            if self.auto_normalize:
                self._write_normalized(normalized)
            return

        if not isinstance(data, list):
            raise ValueError('summary_cache.json must contain a list of metadata dicts or a lamp-summary dict')

        for entry in data:
            try:
                eid = int(entry['id'])
            except Exception:
                continue
            self.index.append(entry)
            self.by_id[eid] = entry
            tag = entry.get('tag', 'ungrouped')
            self.by_tag.setdefault(tag, []).append(entry)

    def _parse_user_key(self, key: str) -> Tuple[Optional[str], Optional[str]]:
        if '::' in key:
            split, uid = key.split('::', 1)
            return split, uid
        return None, key

    def _insert_entry(self, idx: MemoryIndex, entry: Dict[str, Any]):
        idx.index.append(entry)
        try:
            eid = int(entry.get('id', -1))
            idx.by_id[eid] = entry
        except Exception:
            pass
        tag = entry.get('tag', 'ungrouped')
        idx.by_tag.setdefault(tag, []).append(entry)

    def _write_normalized(self, normalized: List[Dict[str, Any]]):
        base = os.path.dirname(self.path)
        out_path = os.path.join(base, 'summary_cache_vgpcc.json')
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(normalized, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_user_memory(self, user_key: str) -> MemoryIndex:
        if user_key in self.by_user_key:
            return self.by_user_key[user_key]
        # fallback to global memory if not keyed
        return MemoryIndex(self.index, self.by_id, self.by_tag)


class SummaryRefiner:
    SYSTEM_PROMPT = '''
You are a memory summarization parser. Convert a user summary into a list of memory entries.
Each entry must be JSON with keys: id (int), type (one of [Style, Recent_Preference, Hard_Constraint, Task_Memory, Preference_Conflict]),
time (string or empty), importance (low/medium/high), privacy_level (low/medium/high), tag (short text).
Return a JSON array only. Do not include other text.
'''

    def __init__(self, model: ModelWrapper):
        self.model = model

    def refine_summary(self, user_key: str, summary_text: str, user_id: int) -> List[Dict[str, Any]]:
        prompt = self.SYSTEM_PROMPT + '\n' + json.dumps({
            'user_key': user_key,
            'summary_text': summary_text,
        }, ensure_ascii=False)
        parsed = self.model.generate_json(prompt, max_tokens=512)
        if isinstance(parsed, list) and parsed:
            normalized = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                item.setdefault('id', user_id)
                item.setdefault('type', 'Task_Memory')
                item.setdefault('time', '')
                item.setdefault('importance', 'medium')
                item.setdefault('privacy_level', 'low')
                item.setdefault('tag', '')
                item['user_key'] = user_key
                normalized.append(item)
            if normalized:
                return normalized
        # fallback: single entry
        return [{
            'id': user_id,
            'type': 'Task_Memory',
            'time': '',
            'importance': 'medium',
            'privacy_level': 'low',
            'tag': summary_text,
            'user_key': user_key,
        }]

    def refine_cache(self, raw_cache_path: str, output_path: str) -> str:
        with open(raw_cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return output_path
        refined = []
        for key, summary in data.items():
            if not isinstance(summary, str):
                continue
            split, user_id = key.split('::', 1) if '::' in key else ('unknown', key)
            try:
                user_int = int(user_id)
            except Exception:
                user_int = -1
            refined.extend(self.refine_summary(key, summary, user_int))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(refined, f, ensure_ascii=False, indent=2)
        return output_path


class BaseCompressor:
    SYSTEM_PROMPT = '''
You are a compression model. Given Query, Profile, and MemoryIndex entries, produce a compressed_context within the token_budget.
Return JSON with keys: compressed_context (string), kept_ids (list of ints), dropped_high_importance_ids (list of ints), coverage_stats (dict).
Do not include any extra text.
'''

    def __init__(self, model: ModelWrapper, token_budget: int = 256, use_llm: bool = True, max_memory_entries: int = 30):
        self.model = model
        self.token_budget = token_budget
        self.use_llm = use_llm
        self.max_memory_entries = max_memory_entries

    def compress(
        self,
        profile: str,
        memory_index: MemoryIndex,
        query: Optional[str] = None,
        focus_types: Optional[List[str]] = None,
        token_budget_override: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any], int, int]:
        """Return compressed_context, trajectory_dict, tokens_before, tokens_after

        trajectory_dict must contain: kept_ids, dropped_high_importance_ids, coverage_stats, compression_ratio
        """
        tokens_before = self.model.count_tokens(profile)
        token_budget = token_budget_override or self.token_budget
        # naive heuristic: pick memory entries whose tag words appear in profile, until token budget
        kept = []
        kept_texts = []
        kept_ids = []
        dropped_high = []
        coverage_stats = {}
        current_tokens = 0

        if self.use_llm:
            mem_entries = []
            for entry in memory_index.index[: self.max_memory_entries]:
                mem_entries.append({
                    'id': entry.get('id'),
                    'type': entry.get('type'),
                    'importance': entry.get('importance'),
                    'tag': entry.get('tag'),
                })
            payload = {
                'query': query or '',
                'profile': profile,
                'token_budget': token_budget,
                'focus_types': focus_types or [],
                'memory_index': mem_entries,
            }
            prompt = self.SYSTEM_PROMPT + '\n' + json.dumps(payload, ensure_ascii=False)
            parsed = self.model.generate_json(prompt, max_tokens=768)
            if isinstance(parsed, dict) and parsed.get('compressed_context'):
                compressed_context = parsed.get('compressed_context', '')
                compressed_context = self.model.truncate_to_tokens(compressed_context, token_budget)
                kept_ids = parsed.get('kept_ids', []) or []
                dropped_high = parsed.get('dropped_high_importance_ids', []) or []
                coverage_stats = parsed.get('coverage_stats', {}) or {}
                tokens_after = self.model.count_tokens(compressed_context)
                compression_ratio = round(tokens_after / tokens_before, 4) if tokens_before > 0 else 0.0
                trajectory = {
                    'kept_ids': kept_ids,
                    'dropped_high_importance_ids': dropped_high,
                    'coverage_stats': coverage_stats,
                    'compression_ratio': compression_ratio,
                    'focus_types': focus_types or [],
                    'token_budget': token_budget,
                }
                return compressed_context, trajectory, tokens_before, tokens_after

        # fallback to heuristic if LLM not available
        sorted_index = sorted(memory_index.index, key=lambda e: (e.get('importance', ''), e.get('time', '')), reverse=True)
        if focus_types:
            sorted_index = sorted(
                sorted_index,
                key=lambda e: (0 if e.get('type') in focus_types else 1, e.get('importance', '')),
            )
        for entry in sorted_index:
            entry_text = entry.get('tag', '') + ' ' + entry.get('type', '')
            entry_id = int(entry.get('id'))
            entry_tokens = self.model.count_tokens(entry_text)
            if (entry.get('tag') and entry.get('tag') in profile) or len(kept_texts) == 0:
                if current_tokens + entry_tokens + 10 <= token_budget:
                    kept.append(entry)
                    kept_texts.append(entry_text)
                    kept_ids.append(entry_id)
                    current_tokens += entry_tokens
                    coverage_stats[entry.get('type', 'other')] = coverage_stats.get(entry.get('type', 'other'), 0) + entry_tokens
                else:
                    if entry.get('importance') == 'high':
                        dropped_high.append(entry_id)

        compressed_context = '\n'.join(kept_texts)
        tokens_after = self.model.count_tokens(compressed_context)
        compression_ratio = round(tokens_after / tokens_before, 4) if tokens_before > 0 else 0.0

        trajectory = {
            'kept_ids': kept_ids,
            'dropped_high_importance_ids': dropped_high,
            'coverage_stats': coverage_stats,
            'compression_ratio': compression_ratio,
            'focus_types': focus_types or [],
            'token_budget': token_budget,
        }
        return compressed_context, trajectory, tokens_before, tokens_after


class OnlineVerifier:
    SYSTEM_PROMPT = '''
You are the Online Verifier. You must NOT request or output the original full context text.
Given: a Query, a compressed_context (summaries), a Memory Index (list of metadata), and a trajectory_dict.
Use memory index fields (type/time/importance/privacy/tag) and trajectory stats (kept_ids, dropped_high_importance_ids, coverage_stats, compression_ratio, focus_types, token_budget)
to judge sufficiency and to diagnose missing evidence types.
Respond with a strict JSON object matching the VerificationResult schema exactly.
Do NOT include any other text outside the JSON. Keys: is_sufficient (bool), sufficiency_score (0.0-1.0), missing_evidence_type (choose from ['Style','Recent_Preference','Hard_Constraint','Task_Memory','Preference_Conflict','None']), predicted_utility_loss (0.0-1.0), diagnostic_reasoning (string).
Be concise.
'''

    def __init__(self, model: ModelWrapper):
        self.model = model

    def verify(self, query: str, compressed_context: str, memory_index: MemoryIndex, trajectory: Dict[str, Any]) -> VerificationResult:
        index_stats = self._build_index_stats(memory_index)
        missing_candidates = self._missing_candidates(memory_index, trajectory)
        prompt = self.SYSTEM_PROMPT + '\n' + json.dumps({
            'query': query,
            'compressed_context': compressed_context,
            'trajectory': trajectory,
            'memory_index_sample': [k for k in memory_index.index[:10]],
            'memory_index_stats': index_stats,
            'missing_type_candidates': missing_candidates,
        }, ensure_ascii=False)

        parsed = self.model.generate_json(prompt, max_tokens=512)
        try:
            vr = VerificationResult.model_validate(parsed)
            return vr
        except ValidationError as e:
            # fallback: craft conservative negative result
            return VerificationResult(
                is_sufficient=False,
                sufficiency_score=0.0,
                missing_evidence_type='None',
                predicted_utility_loss=1.0,
                diagnostic_reasoning='Parsing failure: ' + str(e),
            )

    def _extract_json(self, text: str) -> Any:
        return self.model._extract_json(text)

    def _build_index_stats(self, memory_index: MemoryIndex) -> Dict[str, Any]:
        stats = {
            'count': len(memory_index.index),
            'type_counts': {},
            'importance_counts': {},
            'privacy_counts': {},
        }
        for entry in memory_index.index:
            etype = entry.get('type', 'other')
            stats['type_counts'][etype] = stats['type_counts'].get(etype, 0) + 1
            importance = entry.get('importance', 'unknown')
            stats['importance_counts'][importance] = stats['importance_counts'].get(importance, 0) + 1
            privacy = entry.get('privacy_level', 'unknown')
            stats['privacy_counts'][privacy] = stats['privacy_counts'].get(privacy, 0) + 1
        return stats

    def _missing_candidates(self, memory_index: MemoryIndex, trajectory: Dict[str, Any]) -> List[str]:
        covered = set(trajectory.get('coverage_stats', {}).keys())
        all_types = {e.get('type', 'other') for e in memory_index.index}
        missing = [t for t in all_types if t not in covered]
        return missing[:5]


class DownstreamExecutorAndRepairer:
    DOWNSTREAM_SYSTEM_PROMPT = '''
You are a downstream executor that, given Query and Context, must perform the LaMP-3 task (preference judgement / generation).
Return the model's answer only.
'''

    def __init__(self, model: ModelWrapper):
        self.model = model

    def repair(self, compressed_context: str, trajectory: Dict[str, Any], missing_type: str, memory_index: MemoryIndex) -> str:
        # find entries in memory index that match missing_type and were not kept
        kept = set(trajectory.get('kept_ids', []))
        candidates = [e for e in memory_index.index if e.get('type') == missing_type and int(e.get('id')) not in kept]
        # append up to 3 candidate tags as short text
        texts = []
        for c in candidates[:3]:
            texts.append(f"[{c.get('id')}] {c.get('tag', '')} {c.get('type','')}")
        if not texts:
            # If none found, return original compressed_context
            return compressed_context
        repaired = compressed_context + '\n' + '\n'.join(texts)
        return repaired

    def run_downstream(self, query: str, final_context: str) -> Tuple[str, int]:
        prompt = self.DOWNSTREAM_SYSTEM_PROMPT + '\n' + json.dumps({'query': query, 'context': final_context}, ensure_ascii=False)
        out = self.model.generate(prompt, max_tokens=512)
        tokens = self.model.count_tokens(out)
        return out, tokens


class VerifierTrainDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        if Dataset is None:
            raise RuntimeError("torch Dataset is unavailable. Ensure torch is installed.")
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        prompt = OnlineVerifier.SYSTEM_PROMPT + "\n" + json.dumps({
            'query': s.get('query', ''),
            'compressed_context': s.get('compressed_context', ''),
            'trajectory': s.get('trajectory', {}),
            'memory_index_sample': s.get('memory_index_sample', []),
        }, ensure_ascii=False)
        target = {
            'is_sufficient': s.get('teacher', {}).get('is_sufficient', False),
            'sufficiency_score': s.get('teacher', {}).get('sufficiency_score', 0.0),
            'missing_evidence_type': s.get('adversarial', {}).get('ground_truth', {}).get('missing_evidence_type', 'None'),
            'predicted_utility_loss': s.get('utility_loss', 0.0),
            'diagnostic_reasoning': 'teacher_supervision',
        }
        text = prompt + "\n" + json.dumps(target, ensure_ascii=False)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = enc['input_ids'][0]
        attention_mask = enc['attention_mask'][0]
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class VGPCCPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config.get('models', {})
        pipeline_cfg = self.config.get('pipeline', {})
        training_cfg = self.config.get('training', {})
        summary_cfg = self.config.get('summaries', {})
        model_name = model_cfg.get('qwen3_8b') or model_cfg.get('name', 'local-qwen3-8b')
        tokenizer_name = model_cfg.get('tokenizer')
        self.model = ModelWrapper(model_name, tokenizer_name, load_model=pipeline_cfg.get('load_model', True))
        verifier_base = model_cfg.get('verifier_base') or model_cfg.get('verifier_model') or model_name
        verifier_finetuned = model_cfg.get('verifier_finetuned')
        verifier_path = verifier_finetuned if verifier_finetuned and os.path.exists(verifier_finetuned) else verifier_base
        self.verifier_model = ModelWrapper(verifier_path, tokenizer_name, load_model=pipeline_cfg.get('load_model', True))

        dataset_cfg = self.config.get('datasets', {})
        self.dataset_dir = dataset_cfg.get('lamp3') or self.config.get('dataset_dir')

        mem_path = summary_cfg.get('lamp3_summary_cache') or self.config.get('summary_cache_path', 'summary_cache.json')
        refined_path = summary_cfg.get('refined_cache_path') or os.path.join(os.path.dirname(mem_path), 'summary_cache_vgpcc.json')
        use_summary_refine = summary_cfg.get('refine_with_llm', True)
        force_refine = summary_cfg.get('force_refine', False)
        if use_summary_refine and os.path.exists(mem_path) and (force_refine or not os.path.exists(refined_path)):
            refiner = SummaryRefiner(self.model)
            mem_path = refiner.refine_cache(mem_path, refined_path)
        elif os.path.exists(refined_path):
            mem_path = refined_path
        self.memory = MemoryIndexLoader(mem_path, auto_normalize=True)

        self.compressor = BaseCompressor(
            self.model,
            token_budget=self.config.get('token_budget', 256),
            use_llm=pipeline_cfg.get('use_llm_compressor', True),
            max_memory_entries=pipeline_cfg.get('max_memory_entries', 30),
        )
        self.verifier = OnlineVerifier(self.verifier_model)
        self.executor = DownstreamExecutorAndRepairer(self.model)

        self.token_budget = self.config.get('token_budget', 256)
        self.feedback_enabled = pipeline_cfg.get('feedback_enabled', True)
        self.feedback_increase_factor = pipeline_cfg.get('feedback_increase_factor', 1.5)
        self.feedback_max_budget = pipeline_cfg.get('feedback_max_budget', int(self.token_budget * 2))
        self.feedback_min_sufficiency_score = pipeline_cfg.get('feedback_min_sufficiency_score', 0.6)
        self.feedback_min_predicted_utility_loss = pipeline_cfg.get('feedback_min_predicted_utility_loss', 0.2)

        self.min_sufficiency_score = training_cfg.get('min_sufficiency_score', 0.55)
        self.min_utility_loss = training_cfg.get('min_utility_loss', 0.15)
        self.max_utility_loss_for_sufficient = training_cfg.get('max_utility_loss_for_sufficient', 0.4)
        self.min_utility_loss_for_insufficient = training_cfg.get('min_utility_loss_for_insufficient', 0.2)

    def eval(self, dataset_path_or_split: str, max_examples: Optional[int] = None, log_path: Optional[str] = None) -> Dict[str, Any]:
        stats = {'samples': 0, 'total_tokens': 0, 'downstream_tokens': 0, 'accurate': 0}
        results = []
        log_file = open(log_path, 'w', encoding='utf-8') if log_path else None
        for sample in self._iter_dataset(dataset_path_or_split, max_examples=max_examples):
            query = sample.get('input') or sample.get('query')
            profile = self._profile_to_text(sample.get('profile') or sample.get('full_context', ''))
            gold = sample.get('gold')
            user_key = sample.get('user_key')
            user_memory = self.memory.get_user_memory(user_key)

            # Step 2: Compression + verifier feedback
            compressed_context, trajectory, tbefore, tafter, vr, feedback_applied = self._compress_with_feedback(
                profile, user_memory, query
            )

            # Step 4: Targeted Repair
            final_context = compressed_context
            if not vr.is_sufficient and vr.missing_evidence_type != 'None':
                final_context = self.executor.repair(compressed_context, trajectory, vr.missing_evidence_type, user_memory)

            # Step 5: Downstream Execution
            output, d_tokens = self.executor.run_downstream(query, final_context)

            total_tokens = tbefore + tafter + d_tokens
            stats['samples'] += 1
            stats['total_tokens'] += total_tokens
            stats['downstream_tokens'] += d_tokens

            acc = None
            if gold is not None:
                # naive accuracy: exact match
                acc = 1 if output.strip() == gold.strip() else 0
                stats['accurate'] += acc

            results.append({
                'query': query,
                'is_sufficient': vr.is_sufficient,
                'sufficiency_score': vr.sufficiency_score,
                'missing_evidence_type': vr.missing_evidence_type,
                'predicted_utility_loss': vr.predicted_utility_loss,
                'feedback_applied': feedback_applied,
                'output': output,
                'tokens_before': tbefore,
                'tokens_after': tafter,
                'downstream_tokens': d_tokens,
                'total_tokens': total_tokens,
                'accuracy': acc,
            })

            if log_file:
                log_file.write(json.dumps(results[-1], ensure_ascii=False) + '\n')

        if log_file:
            log_file.close()
            summary_path = log_path + '.summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        return {'summary': stats, 'results': results}

    # Training data collection
    def collect_train_data(self, dataset_path_or_split: str, out_path: str, max_examples: Optional[int] = None, log_path: Optional[str] = None):
        saved = 0
        filtered = {'low_sufficiency_score': 0, 'utility_loss_low': 0, 'utility_loss_high': 0, 'utility_loss_inconsistent': 0}
        log_file = open(log_path, 'w', encoding='utf-8') if log_path else None
        with open(out_path, 'w', encoding='utf-8') as f_out:
            for sample in self._iter_dataset(dataset_path_or_split, max_examples=max_examples):
                query = sample.get('input') or sample.get('query')
                profile = self._profile_to_text(sample.get('profile') or sample.get('full_context', ''))
                user_key = sample.get('user_key')
                user_memory = self.memory.get_user_memory(user_key)

                # compress + verifier feedback
                compressed_context, trajectory, tbefore, tafter, vr, feedback_applied = self._compress_with_feedback(
                    profile, user_memory, query
                )

                # Teacher verifier: allowed to see full context
                teacher_prompt = '''
Teacher Verifier: You may use both full_context and compressed_context to decide sufficiency.
Return JSON: {"is_sufficient": bool, "sufficiency_score": float}
'''
                teacher_in = teacher_prompt + json.dumps({'query': query, 'full_context': profile, 'compressed_context': compressed_context}, ensure_ascii=False)
                t_parsed = self.model.generate_json(teacher_in, max_tokens=256)
                is_suff = bool(t_parsed.get('is_sufficient', False))
                score = float(t_parsed.get('sufficiency_score', 0.0)) if t_parsed.get('sufficiency_score') is not None else 0.0

                # Utility loss: use model as judge comparing full vs compressed downstream outputs
                full_out, _ = self.executor.run_downstream(query, profile)
                comp_out, _ = self.executor.run_downstream(query, compressed_context)
                judge_prompt = 'Judge the difference between FULL and COMPRESSED outputs. Return JSON {"utility_loss": float, "reason": str}\n' + json.dumps({'full': full_out, 'compressed': comp_out}, ensure_ascii=False)
                j = self.model.generate_json(judge_prompt, max_tokens=128)
                utility_loss = float(j.get('utility_loss', 0.0)) if j.get('utility_loss') is not None else 0.0

                # quality filtering
                if score < self.min_sufficiency_score:
                    filtered['low_sufficiency_score'] += 1
                    continue
                if is_suff and utility_loss > self.max_utility_loss_for_sufficient:
                    filtered['utility_loss_high'] += 1
                    continue
                if (not is_suff) and utility_loss < self.min_utility_loss_for_insufficient:
                    filtered['utility_loss_inconsistent'] += 1
                    continue
                if utility_loss < self.min_utility_loss:
                    filtered['utility_loss_low'] += 1
                    continue

                # Adversarial attack: remove one tag type (if any present in trajectory coverage_stats)
                adversarial = None
                if trajectory.get('coverage_stats'):
                    attack_type = next(iter(trajectory['coverage_stats'].keys()))
                    removed_context = self._adversarial_remove(compressed_context, attack_type)
                    adversarial = {
                        'attack_type': attack_type,
                        'adversarial_context': removed_context,
                        'ground_truth': {'is_sufficient': False, 'missing_evidence_type': attack_type}
                    }

                record = {
                    'query': query,
                    'compressed_context': compressed_context,
                    'memory_index_sample': [m for m in user_memory.index[:20]],
                    'trajectory': trajectory,
                    'verifier_online': vr.model_dump(),
                    'feedback_applied': feedback_applied,
                    'teacher': {'is_sufficient': is_suff, 'sufficiency_score': score},
                    'utility_loss': utility_loss,
                    'adversarial': adversarial,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                saved += 1
                if log_file:
                    log_file.write(json.dumps({'saved': saved, 'id': sample.get('id')}, ensure_ascii=False) + '\n')
                if max_examples and saved >= max_examples:
                    break
        if log_file:
            log_file.close()
            summary_path = log_path + '.summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({'saved': saved, 'filtered': filtered}, f, ensure_ascii=False, indent=2)
        return {'saved': saved, 'out_path': out_path, 'filtered': filtered}

    def train_verifier(self, train_jsonl: str, output_dir: str, max_length: int = 512, epochs: int = 1, batch_size: int = 1, lr: float = 5e-5):
        if not HF_AVAILABLE or torch is None:
            raise RuntimeError("transformers/torch not available; cannot train verifier.")
        if not hasattr(self.verifier_model, 'tokenizer') or self.verifier_model.tokenizer is None:
            raise RuntimeError("Tokenizer not available for verifier model.")

        from transformers import Trainer, TrainingArguments

        os.makedirs(output_dir, exist_ok=True)
        dataset = VerifierTrainDataset(train_jsonl, self.verifier_model.tokenizer, max_length=max_length)
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to=[],
        )
        trainer = Trainer(
            model=self.verifier_model.model,
            args=args,
            train_dataset=dataset,
        )
        trainer.train()
        trainer.save_model(output_dir)
        # also save tokenizer for reload
        try:
            self.verifier_model.tokenizer.save_pretrained(output_dir)
        except Exception:
            pass
        return output_dir

    def _iter_dataset(self, dataset_path_or_split: str, max_examples: Optional[int] = None):
        # If a file path exists, fall back to jsonl
        if os.path.exists(dataset_path_or_split):
            with open(dataset_path_or_split, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    if 'user_key' not in sample:
                        sample['user_key'] = f"train::{sample.get('id', 'unknown')}"
                    yield sample
                    if max_examples:
                        max_examples -= 1
                        if max_examples <= 0:
                            break
            return

        split = dataset_path_or_split
        if not self.dataset_dir:
            raise ValueError('dataset_dir is not set in config')

        questions_path = os.path.join(self.dataset_dir, f'{split}_questions.json')
        outputs_path = os.path.join(self.dataset_dir, f'{split}_outputs.json')
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.loads(f.read())

        with open(outputs_path, 'r', encoding='utf-8') as f:
            outputs = json.loads(f.read())
        golds = {g['id']: g['output'] for g in outputs.get('golds', [])}

        count = 0
        for q in questions:
            sid = q.get('id')
            sample = {
                'id': sid,
                'input': q.get('input'),
                'profile': q.get('profile'),
                'gold': golds.get(sid),
                'user_key': f"{split}::{sid}",
            }
            yield sample
            count += 1
            if max_examples and count >= max_examples:
                break

    def _profile_to_text(self, profile: Any) -> str:
        if isinstance(profile, str):
            return profile
        if isinstance(profile, list):
            chunks = []
            for item in profile:
                if isinstance(item, dict):
                    chunks.append(f"[{item.get('id')}] score={item.get('score')} {item.get('text')}")
                else:
                    chunks.append(str(item))
            return '\n'.join(chunks)
        return str(profile)

    def _adversarial_remove(self, compressed_context: str, attack_type: str) -> str:
        # primitive: remove any line containing attack_type or its tokens
        lines = compressed_context.splitlines()
        remain = [l for l in lines if attack_type not in l]
        return '\n'.join(remain)

    def _compress_with_feedback(
        self,
        profile: str,
        user_memory: MemoryIndex,
        query: str,
    ) -> Tuple[str, Dict[str, Any], int, int, VerificationResult, bool]:
        compressed_context, trajectory, tbefore, tafter = self.compressor.compress(
            profile,
            user_memory,
            query=query,
        )
        vr = self.verifier.verify(query, compressed_context, user_memory, trajectory)
        feedback_applied = False

        if not self.feedback_enabled:
            return compressed_context, trajectory, tbefore, tafter, vr, feedback_applied

        need_feedback = (
            (not vr.is_sufficient)
            or (vr.sufficiency_score < self.feedback_min_sufficiency_score)
            or (vr.predicted_utility_loss >= self.feedback_min_predicted_utility_loss)
        )

        if not need_feedback:
            return compressed_context, trajectory, tbefore, tafter, vr, feedback_applied

        focus_types = []
        if vr.missing_evidence_type and vr.missing_evidence_type != 'None':
            focus_types = [vr.missing_evidence_type]

        new_budget = min(self.feedback_max_budget, int(self.token_budget * self.feedback_increase_factor))
        if new_budget <= self.token_budget and not focus_types:
            return compressed_context, trajectory, tbefore, tafter, vr, feedback_applied

        compressed_context2, trajectory2, tbefore2, tafter2 = self.compressor.compress(
            profile,
            user_memory,
            query=query,
            focus_types=focus_types,
            token_budget_override=new_budget,
        )
        vr2 = self.verifier.verify(query, compressed_context2, user_memory, trajectory2)
        feedback_applied = True
        return compressed_context2, trajectory2, tbefore2, tafter2, vr2, feedback_applied


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='VGPCC Pipeline Runner')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--mode', type=str, choices=['eval', 'train', 'train_verifier'], default='eval')
    parser.add_argument('--split', type=str, default='dev', help='Dataset split (train/dev)')
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--train-out', type=str, default='vgpcc_train.jsonl')
    parser.add_argument('--log-path', type=str, default='vgpcc_run.log.jsonl')
    parser.add_argument('--verifier-train-jsonl', type=str, default='vgpcc_train.jsonl')
    parser.add_argument('--verifier-out', type=str, default='verifier_ft')
    parser.add_argument('--verifier-epochs', type=int, default=1)
    parser.add_argument('--verifier-batch-size', type=int, default=1)
    parser.add_argument('--verifier-lr', type=float, default=5e-5)

    args = parser.parse_args()

    cfg = args.config or os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    cfg = cfg if os.path.exists(cfg) else os.path.join(os.path.dirname(__file__), 'config.yaml')
    if not os.path.exists(cfg):
        print('config.yaml not found in parent or local folder. Create a config with model and summary_cache_path.')
    else:
        # ensure output directories exist
        if args.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(args.log_path)), exist_ok=True)
        if args.train_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.train_out)), exist_ok=True)
        p = VGPCCPipeline(cfg)
        if args.mode == 'eval':
            out = p.eval(args.split, max_examples=args.max_examples, log_path=args.log_path)
            print(json.dumps(out['summary'], ensure_ascii=False, indent=2))
        elif args.mode == 'train':
            out = p.collect_train_data(args.split, args.train_out, max_examples=args.max_examples, log_path=args.log_path)
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            out_dir = p.train_verifier(
                train_jsonl=args.verifier_train_jsonl,
                output_dir=args.verifier_out,
                epochs=args.verifier_epochs,
                batch_size=args.verifier_batch_size,
                lr=args.verifier_lr,
            )
            print(json.dumps({'verifier_out': out_dir}, ensure_ascii=False, indent=2))
