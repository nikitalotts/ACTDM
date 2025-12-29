import sys
import logging
import traceback
from pathlib import Path
import torch
import importlib
import platform
import subprocess
import json
import time

# ----- CONFIG -----
ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / 'checkpoints' / 'tencdm-bert-base-cased-384-0.0002-rocstories-cfg=0.0' / 'last.pth'
DEFAULT_DATASET = ROOT / 'datasets' / 'rocstories' / 'decoder-bert-base-cased-80-transformer'
LOGFILE = ROOT / 'check_models.log'

# ----- LOGGING SETUP -----
logger = logging.getLogger('check_models')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

fh = logging.FileHandler(LOGFILE, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


def extra_info():
    info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'torch_version': getattr(torch, '__version__', None),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    try:
        import transformers
        info['transformers_version'] = getattr(transformers, '__version__', None)
    except Exception:
        info['transformers_version'] = None
    return info


def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        return out
    except Exception as e:
        return f'ERROR running {cmd}: {e}'


def safe_torch_load(p):
    try:
        logger.debug(f'Attempting torch.load for {p} (cpu)')
        ck = torch.load(p, map_location='cpu')
        logger.info(f'torch.load successful: {p} (type={type(ck)})')
        return ck
    except Exception as e:
        logger.exception(f'torch.load failed for {p}: {e}')
        return None


def extract_state_dict(ck):
    if ck is None:
        return None
    # Common keys
    if isinstance(ck, dict):
        logger.debug('Top-level checkpoint is dict. Enumerating keys...')
        keys = list(ck.keys())
        logger.debug('Top-level keys: %s', keys[:50])
        for k in ('state_dict', 'model_state_dict', 'model', 'net', 'state_dict_ema'):
            if k in ck and isinstance(ck[k], dict):
                logger.info('Found state dict under key: %s (len=%d)', k, len(ck[k]))
                return ck[k]
        # If dict of tensors (likely raw state_dict)
        try:
            all_values_are_tensors = all(hasattr(v, 'size') for v in ck.values())
        except Exception:
            all_values_are_tensors = False
        if all_values_are_tensors:
            logger.info('Top-level dict appears to be state_dict (all values look like tensors)')
            return ck
        # try to find nested dict that looks like state_dict
        for kk, v in ck.items():
            if isinstance(v, dict):
                try:
                    if len(v) > 0 and all(hasattr(x, 'size') for x in v.values()):
                        logger.info('Found nested state-dict-like object under key: %s (len=%d)', kk, len(v))
                        return v
                except Exception:
                    continue
    logger.warning('Could not extract a state_dict from checkpoint')
    return None


def summarize_tensor(t):
    try:
        s = {
            'shape': tuple(t.size()),
            'dtype': str(t.dtype),
            'numel': t.numel(),
            'mean': float(t.mean().item()) if t.numel() > 0 else None,
            'std': float(t.std().item()) if t.numel() > 0 else None,
            'min': float(t.min().item()) if t.numel() > 0 else None,
            'max': float(t.max().item()) if t.numel() > 0 else None,
        }
        return s
    except Exception as e:
        return {'error': str(e), 'type': str(type(t))}


def inspect_state_dict(sd, limit=50):
    logger.info('State-dict size: %d keys', len(sd))
    items = list(sd.items())[:limit]
    for k, v in items:
        try:
            summary = summarize_tensor(v)
            logger.debug('  %s -> %s', k, json.dumps(summary, ensure_ascii=False))
        except Exception:
            logger.debug('  %s -> %s', k, str(type(v)))


def strip_prefix(sd, prefix):
    new = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            new[k[len(prefix):]] = v
        else:
            new[k] = v
    return new


def compare_keys(model_keys, ck_keys, n_show=20):
    missing = [k for k in model_keys if k not in ck_keys]
    unexpected = [k for k in ck_keys if k not in model_keys]
    logger.info('Missing keys count: %d', len(missing))
    logger.info('Unexpected keys count: %d', len(unexpected))
    if len(missing) > 0:
        logger.debug('Example missing keys: %s', missing[:n_show])
    if len(unexpected) > 0:
        logger.debug('Example unexpected keys: %s', unexpected[:n_show])
    return missing, unexpected


def try_load_transformers_bert(checkpoint_path, dataset_path=None):
    try:
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        logger.exception('Could not import transformers: %s', e)
        return False

    if not checkpoint_path.exists():
        logger.error('Checkpoint path does not exist: %s', checkpoint_path)
        return False

    ck = safe_torch_load(checkpoint_path)
    if ck is None:
        return False

    sd = extract_state_dict(ck)
    if sd is None:
        logger.error('No state_dict extracted from checkpoint. Top-level type: %s', type(ck))
        return False

    logger.info('Preparing state_dict for loading (removing common prefixes)')
    # normalize keys
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    # try load base bert
    try:
        logger.info('Instantiating bert-base-cased from HuggingFace')
        model = AutoModel.from_pretrained('bert-base-cased')
    except Exception as e:
        logger.exception('AutoModel.from_pretrained failed: %s', e)
        try:
            from transformers import BertModel
            model = BertModel.from_pretrained('bert-base-cased')
        except Exception as e2:
            logger.exception('Fallback BertModel failed: %s', e2)
            return False

    model_sd = model.state_dict()
    logger.info('Model state_dict keys: %d', len(model_sd))
    missing, unexpected = compare_keys(model_sd.keys(), sd.keys())

    # Try loading non-strict first
    try:
        logger.info('Attempting model.load_state_dict(..., strict=False)')
        res = model.load_state_dict(sd, strict=False)
        logger.info('load_state_dict returned: %s', str(res))
    except Exception as e:
        logger.exception('load_state_dict strict=False failed: %s', e)
        # try prefix adjustments
        logger.info('Trying to strip common prefixes like "bert." and "encoder."')
        sd2 = strip_prefix(sd, 'bert.')
        sd2 = {k.replace('encoder.', ''): v for k, v in sd2.items()}
        try:
            res = model.load_state_dict(sd2, strict=False)
            logger.info('Loaded after prefix adjust. Result: %s', str(res))
            sd = sd2
        except Exception as e2:
            logger.exception('Still failed after prefix adjust: %s', e2)
            return False

    # Inspect some tensors in checkpoint
    try:
        logger.info('Sample checkpoint tensors:')
        inspect_state_dict(sd, limit=30)
    except Exception:
        logger.exception('Failed while inspecting state dict')

    # Try tokenizer
    try:
        if dataset_path and Path(dataset_path).exists():
            logger.info('Attempting to load tokenizer from dataset path: %s', dataset_path)
            tokenizer = AutoTokenizer.from_pretrained(str(dataset_path))
        else:
            logger.info('Loading tokenizer bert-base-cased')
            tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        logger.info('Tokenizer loaded. Special tokens: %s', tokenizer.special_tokens_map)
    except Exception as e:
        logger.exception('Tokenizer load failed: %s', e)
        tokenizer = None

    # Quick forward pass
    try:
        text = 'This is a quick forward pass test.'
        if tokenizer is None:
            logger.error('Tokenizer unavailable, cannot run forward pass')
            return True
        toks = tokenizer(text, return_tensors='pt')
        logger.debug('Tokenized input keys: %s', list(toks.keys()))
        with torch.no_grad():
            out = model(**toks)
        if hasattr(out, 'last_hidden_state'):
            logger.info('Forward pass ok, last_hidden_state shape: %s', tuple(out.last_hidden_state.size()))
        else:
            logger.info('Forward pass ok, output type: %s', type(out))
    except Exception as e:
        logger.exception('Forward pass failed: %s', e)
        return False

    return True


def scan_checkpoints_dir(p):
    p = Path(p)
    if not p.exists():
        logger.warning('Checkpoints dir does not exist: %s', p)
        return
    files = list(p.rglob('*.pth')) + list(p.rglob('*.pt'))
    logger.info('Found %d .pth/.pt files under %s', len(files), p)
    for f in files:
        logger.info('--- Inspecting %s', f)
        try:
            ck = safe_torch_load(f)
            if ck is None:
                continue
            if isinstance(ck, dict):
                top_keys = list(ck.keys())
                logger.debug('Top-level keys: %s', top_keys[:50])
                sd = extract_state_dict(ck)
                if sd:
                    logger.info('Extracted state_dict with %d keys', len(sd))
                else:
                    logger.info('No state_dict-like object found inside %s', f)
            else:
                logger.info('Checkpoint top-level type: %s', type(ck))
        except Exception as e:
            logger.exception('Failed to inspect %s: %s', f, e)


def list_files_tree(root, depth=2, exts=None, limit=200):
    root = Path(root)
    if not root.exists():
        logger.warning('Path does not exist: %s', root)
        return []
    out = []
    for p in root.rglob('*'):
        if p.is_file():
            if exts and p.suffix.lower() not in exts:
                continue
            out.append(p)
            if len(out) >= limit:
                break
    logger.info('Listing %d files under %s', len(out), root)
    for f in out[:50]:
        try:
            logger.debug('  %s (size=%d)', f, f.stat().st_size)
        except Exception:
            logger.debug('  %s', f)
    return out


if __name__ == '__main__':
    start = time.time()
    logger.info('=== check_models.py start ===')
    logger.info('ROOT: %s', ROOT)

    # env info
    env_info = extra_info()
    logger.info('ENV INFO: %s', json.dumps(env_info, ensure_ascii=False))

    # pip freeze (may be long)
    try:
        freeze = run_cmd('pip freeze')
        logger.debug('pip freeze:
%s', freeze[:5000])
    except Exception:
        logger.exception('pip freeze failed')

    # list dataset path
    list_files_tree(DEFAULT_DATASET, depth=2, exts=None, limit=500)

    # run the main checks
    ok = try_load_transformers_bert(DEFAULT_CHECKPOINT, dataset_path=DEFAULT_DATASET)

    # scan checkpoints dir
    scan_checkpoints_dir(ROOT / 'checkpoints')

    elapsed = time.time() - start
    logger.info('=== finished in %.2f sec. BERT load OK? %s ===', elapsed, ok)

    # exit code
    if not ok:
        logger.error('One or more checks failed. See %s for details', LOGFILE)
        sys.exit(2)
    else:
        logger.info('All required checks passed (or loaded enough to proceed).')
        sys.exit(0)
