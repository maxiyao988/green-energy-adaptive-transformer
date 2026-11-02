"""Isolated formal trainer prefixes. This module is executed only by the user-run queue."""
import os
import sys
sys.dont_write_bytecode = True
for _key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.pop(_key, None)
os.environ.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE')
from io22b_v3 import *
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'revision_work/outputs/_deps_pyarrow15'))
import torch
torch.set_num_threads(12)
torch.set_num_interop_threads(12)
import numpy as np
import pandas as pd
import ast
import copy
import subprocess
import tempfile
import time
import traceback
from scripts.python import panel_common as common
from scripts.python import panel_pathformer_main_v2 as formal
from scripts.python import panel_pathformer_experiment3 as exp3
from scripts.python import panel_pathformer_experiment4_multiseed as exp4
from scripts.python import panel_baseline_lstm as lstm
from revision_work.scripts.causal import core

STAGE = 'training'
TELE = None
LAST = 0.
TENSOR_DIR = core.OUT / 'tensors'

def profile():
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 12
    assert os.environ.get('OMP_NUM_THREADS') is None
    assert os.environ.get('OMP_DYNAMIC') == os.environ.get('MKL_DYNAMIC') == 'FALSE'
    return dict(device='cpu', torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__,
                intra=12, inter=12, environment={k: os.environ.get(k) for k in ENV_KEYS})

def telemetry(event, force=False):
    global LAST
    now = time.monotonic()
    if not force and now - LAST < 300:
        return
    LAST = now
    row = dict(utc=utc(), event=event, pid=os.getpid(), stage=STAGE,
               profile=profile(), load_average=list(os.getloadavg()))
    try:
        result = subprocess.run(['ps', '-Ao', 'pid,pcpu,comm'], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        row['top_cpu_processes'] = sorted(lines[1:], key=lambda s: float(s.split(None, 2)[1]), reverse=True)[:12]
    except Exception as error:
        row['load_snapshot_error'] = repr(error)
    if TELE:
        TELE.write(json.dumps(row) + '\n')
        TELE.flush()
    return row

def install_guard(evaluation=False):
    global STAGE
    STAGE = 'evaluation' if evaluation else 'training'
    tmp = target(OUT / 'tmp' / str(os.getpid()))
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    test_sources = {str((ROOT / r[k]).resolve()) for r in source_rows() for k in ['source', 'canonical']}
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute().resolve()
        mode = args[1]
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and not path.is_relative_to(OUT):
            raise RuntimeError('READ-ONLY source protection: ' + str(path))
        if not evaluation and not writing:
            name = path.name.lower()
            if str(path) in test_sources or name in ['test.parquet', 'metrics.csv', 'b_forecast_skill.csv'] or path.is_relative_to(ROOT / 'revision_work/outputs/step21'):
                raise RuntimeError('STOP: TEST locked during training: ' + str(path))
    sys.addaudithook(audit)

def dataset(include_test=False):
    """Read train/val rows only before the barrier; preserve core.load_causal file binding."""
    if include_test:
        barrier()
        ds = core.load_causal()
        assert {sp: int((ds.split == sp).sum()) for sp in COUNTS} == COUNTS
        return ds
    meta = pd.read_csv(TENSOR_DIR / 'sample_index.csv')
    keep = meta.split.isin(['train', 'val']).to_numpy()
    def selected(name):
        mapped = np.load(TENSOR_DIR / name, mmap_mode='r', allow_pickle=False)
        return np.asarray(mapped[keep]).copy()
    keys = meta.loc[keep].reset_index(drop=True)
    ds = common.PanelDataset(ticker=keys.ticker.to_numpy(), anchor_date=pd.to_datetime(keys.anchor_date).to_numpy(),
         split=keys.split.to_numpy(), X_daily=selected('X_daily_causal_raw.npy'), X_weekly=selected('X_weekly_causal_raw.npy'),
         y={h: selected(f'y_h{h}_frozen.npy') for h in HS})
    assert not np.any(ds.split == 'test')
    assert {sp: int((ds.split == sp).sum()) for sp in ['train', 'val']} == {sp: COUNTS[sp] for sp in ['train', 'val']}
    normal = common.apply_normalizer(ds, common.fit_train_only_normalizer(ds), 'daily_weekly')
    for branch, arr in zip(['daily', 'weekly'], normal):
        assert np.array_equal(arr, selected(f'X_{branch}_causal_standardized.npy')), branch
    return ds

def prefix(owner, name):
    """Only a suffix is removed. Every statement through restored model.eval() is original."""
    source = Path(owner.__file__)
    tree = ast.parse(source.read_text())
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    if owner is lstm:
        cut = next(i + 1 for i, n in enumerate(node.body)
                   if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'eval'
                   and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'model')
    else:
        marker = 'test_mask' if owner is formal else 'val_mask'
        cut = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == marker for t in n.targets))
    original = node.body[:cut]
    # The exception wrapper records failures; it does not alter the loop, optimizer or restoration.
    handler = ast.ExceptHandler(type=ast.Name(id='Exception', ctx=ast.Load()), name='fit_error',
              body=ast.parse("return dict(ok=False, error=repr(fit_error), captured=locals())").body)
    node.body = [ast.Try(body=original + ast.parse('return dict(ok=True, captured=locals())').body,
                        handlers=[handler], orelse=[], finalbody=[])]
    node.name = 'registered_training_prefix'
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec')
    return code, dict(source=rel(source), function=name, prefix_line_start=original[0].lineno, prefix_line_end=original[-1].end_lineno,
                      prefix_ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=original, type_ignores=[]), include_attributes=False).encode()).hexdigest())

def model_factory(job):
    model, seed = job['model'], job['seed']
    if model == 'lstm':
        return lstm.make_model('daily_weekly', 5, 5)
    if model == 'adaptive' and seed == 42:
        return formal._make_model('daily_weekly', 'adaptive')
    if model in ['static', 'adaptive'] and seed != 42:
        return exp4._make_experiment4_model(model)
    return exp3._make_experiment3_model(model, None, None)

def inference(model, ds, job, split):
    """Use the existing Pathformer helper or the verbatim LSTM TEST inference block."""
    h = job['h']
    mask = ds.split == split
    n = int(mask.sum())
    assert n == COUNTS[split]
    model.cpu().eval()
    if job['model'] != 'lstm':
        loaders = formal._build_training_loaders('daily_weekly', h, ds, batch_size=32)
        base = loaders[['train', 'val', 'test'].index(split)]
        loader = torch.utils.data.DataLoader(base.dataset, batch_size=32, shuffle=False)
        pred, dw, ww = exp4._batched_inference(model, loader, n, torch.device('cpu'))
    else:
        norm = common.apply_normalizer(ds, common.fit_train_only_normalizer(ds), 'daily_weekly')
        tree = ast.parse(Path(lstm.__file__).read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_and_eval_frequency')
        restore = next(i for i, z in enumerate(fn.body) if isinstance(z, ast.Expr) and isinstance(z.value, ast.Call)
                       and isinstance(z.value.func, ast.Attribute) and z.value.func.attr == 'load_state_dict')
        block = next(z for z in fn.body[restore + 1:] if isinstance(z, ast.With))
        env = dict(lstm.__dict__, model=model, frequency='daily_weekly', device=torch.device('cpu'),
                   X_d_test=norm[0][mask], X_w_test=norm[1][mask], y_test=ds.y[h][mask])
        exec(compile(ast.Module(body=[block], type_ignores=[]), lstm.__file__, 'exec'), env)
        pred = env['y_pred']
        dw = ww = None
    frame = pd.DataFrame(dict(ticker=ds.ticker[mask], anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),
                              split=split, y_true=ds.y[h][mask], y_pred=np.asarray(pred).reshape(-1)))
    assert not frame.duplicated(['ticker', 'anchor_date']).any()
    if dw is not None:
        for i, name in enumerate(WEIGHTS):
            frame[name] = (dw if i < 4 else ww)[:, i % 4]
    return frame

def train(job, folder):
    global TELE
    check_protocol()
    assert not (OUT / 'ALL_TRAINING_FINISHED.json').exists(), 'No training after TEST barrier'
    install_guard()
    TELE = target(folder / 'telemetry.jsonl').open('x')
    started, clock = utc(), time.perf_counter()
    start_profile = telemetry('start', True)
    ds = dataset()
    exp4._verify_frozen_dependencies()
    exp4._verify_frozen_training_constants()
    lr = 3e-5 if job['setting'] == 'T1' else 1e-4
    delta = {5: .098083, 10: .138578, 20: .195560}[job['h']] if job['setting'] == 'T2' else 1.
    # Only process-local globals change; source files remain immutable.
    formal.LEARNING_RATE, formal.HUBER_DELTA = lr, delta
    models, hooks, val_mse = [], [], []
    acc = dict(n=0, s=0.)
    original_huber = torch.nn.HuberLoss
    def loss_factory(*a, **kw):
        kw['delta'] = delta
        loss = original_huber(*a, **kw)
        def observe(module, inputs, output):
            if models and not models[0].training:
                p, y = inputs
                diff = p.detach() - y.detach()
                acc['n'] += len(y)
                acc['s'] += float((diff * diff).sum(dtype=torch.float64))
                assert acc['n'] <= COUNTS['val']
                if acc['n'] == COUNTS['val']:
                    val_mse.append(acc['s'] / acc['n'])
                    acc.update(n=0, s=0.)
        hooks.append(loss.register_forward_hook(observe))
        return loss
    def capture(factory):
        def make(*a, **kw):
            m = factory(*a, **kw)
            models.append(m)
            hooks.append(m.register_forward_pre_hook(heartbeat))
            return m
        return make
    # Forward pre-hooks return None and never alter model inputs.
    def heartbeat(module, inputs):
        telemetry('five_minute')
        return None
    if job['model'] == 'lstm':
        owner, name = lstm, 'train_and_eval_frequency'
    elif job['model'] in ['static', 'adaptive'] and job['seed'] != 42:
        owner, name = exp4, 'train_and_validate_cell'
    else:
        owner = formal if job['model'] == 'adaptive' else exp3
        name = 'train_one_cell' if owner is formal else 'train_and_validate_cell'
    code, excerpt = prefix(owner, name)
    env = dict(owner.__dict__)
    env.update(SEED=job['seed'], LEARNING_RATE=lr, HUBER_DELTA=delta, MAX_EPOCHS=100,
               EARLY_STOPPING_PATIENCE=10, load_frozen_panel=lambda: ds, _load_dataset=lambda: ds,
               _formal_check_dataset_contract=lambda x: None, _verify_frozen_training_constants=lambda: None)
    factory_name = '_make_model' if owner is formal else '_make_experiment3_model' if owner is exp3 else '_make_experiment4_model'
    if owner is lstm:
        env['DualFrequencyLSTM'] = capture(lstm.DualFrequencyLSTM)
        # Same CPU-only override as 14c-day/B5, without changing the formal loop.
        torch.backends.mps.is_available = lambda: False
        torch.cuda.is_available = lambda: False
    else:
        env[factory_name] = capture(getattr(owner, factory_name))
    if owner is formal:
        # env.SEED initializes the run; imported loader globals retain seed42.
        env['_build_training_loaders'] = formal._build_training_loaders
    if owner is exp3:
        assert exp3.SEED == 42
    exec(code, env)
    try:
        torch.nn.HuberLoss = loss_factory
        if owner is lstm:
            result = env['registered_training_prefix'](ds, 'daily_weekly', job['h'], common.fit_train_only_normalizer(ds))
        elif owner is formal:
            result = env['registered_training_prefix']('daily_weekly', job['h'], 'adaptive')
        elif owner is exp4:
            result = env['registered_training_prefix'](job['model'], f"{job['h']}d", job['seed'])
        else:
            result = env['registered_training_prefix'](job['model'], f"{job['h']}d", None, None)
    finally:
        torch.nn.HuberLoss = original_huber
        for hook in hooks:
            hook.remove()
    local = result['captured']
    hist = local.get('history', local.get('history_rows', []))
    history = pd.DataFrame(hist)
    for key in ['epoch', 'train_loss', 'val_loss']:
        if key not in history:
            history[key] = pd.Series(dtype=float)
    history['val_mse'] = val_mse[:len(history)] if len(val_mse) >= len(history) else [np.nan] * len(history)
    history['huber_delta'], history['lr'] = delta, lr
    csv(folder / 'history.csv', history)
    best = int(local.get('best_epoch', 0))
    model = local.get('model')
    ok = bool(result['ok'] and best > 0 and np.isfinite(history[['train_loss', 'val_loss']].to_numpy()).all())
    error = result.get('error')
    paths = [folder / 'history.csv', folder / 'telemetry.jsonl']
    if model is not None and best > 0:
        if not result['ok']:
            model.load_state_dict(local['best_state'])
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        with target(folder / 'best_state.pt').open('xb') as handle:
            torch.save(state, handle)
        paths.append(folder / 'best_state.pt')
        if job['model'] == 'static':
            js(folder / 'static_logits.json', {k: v.tolist() for k, v in state.items() if 'logit' in k})
            paths.append(folder / 'static_logits.json')
        if ok:
            for split in ['train', 'val']:
                frame = inference(model, ds, job, split)
                csv(folder / (split + '.csv'), frame)
                paths.append(folder / (split + '.csv'))
    ended = utc()
    end_profile = telemetry('end', True)
    TELE.close()
    data = dict(job=job, protocol_sha256=check_protocol(), status='COMPLETE' if ok else 'FAILED', error=error,
                best_epoch=best, epochs=len(history), lr=lr, delta=delta, prefix=excerpt,
                started_utc=started, ended_utc=ended, elapsed_utc_seconds=elapsed(started, ended),
                monotonic_seconds=time.perf_counter() - clock, profile_start=start_profile, profile_end=end_profile,
                parameters=formal.count_trainable_parameters(model) if model else None, test_values_read=False,
                artifacts=[artifact(p) for p in paths])
    js(folder / 'COMPLETE.json', data)
    print(job_id(job), data['status'], data['epochs'], data['elapsed_utc_seconds'], flush=True)

def evaluate(job, folder):
    global TELE
    barrier()
    install_guard(True)
    TELE = target(folder / 'telemetry.jsonl').open('x')
    telemetry('test_evaluation_start', True)
    start, clock = utc(), time.perf_counter()
    ds = dataset(True)
    setting = job['setting']
    if setting == 'T0':
        r = next(r for r in source_rows() if (r['model'], r['h'], r['seed']) == (job['model'], job['h'], job['seed']))
        assert sha(ROOT / r['source']) == r['sha256']
        frame = pd.read_parquet(ROOT / r['source'])
        source = r['source']
        status = 'COMPLETE'
    else:
        saved_path, saved = completion(job)
        assert saved is not None
        source, status = rel(saved_path), saved['status']
        if status == 'COMPLETE':
            model = model_factory(job)
            model.load_state_dict(torch.load(saved_path.parent / 'best_state.pt', map_location='cpu', weights_only=True), strict=True)
            frame = inference(model, ds, job, 'test')
        else:
            mask = ds.split == 'test'
            frame = pd.DataFrame(dict(ticker=ds.ticker[mask], anchor_date=ds.anchor_date[mask], split='test',
                                      y_true=ds.y[job['h']][mask], y_pred=np.nan))
            if job['model'] != 'lstm':
                for name in WEIGHTS:
                    frame[name] = np.nan
    frame.anchor_date = pd.to_datetime(frame.anchor_date).dt.strftime('%Y-%m-%d')
    frame = frame.sort_values(['anchor_date', 'ticker']).reset_index(drop=True)
    mask = ds.split == 'test'
    expected = pd.DataFrame(dict(ticker=ds.ticker[mask], anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),
                                 y_true=ds.y[job['h']][mask])).sort_values(['anchor_date', 'ticker']).reset_index(drop=True)
    assert len(frame) == COUNTS['test'] and frame[['ticker', 'anchor_date']].equals(expected[['ticker', 'anchor_date']])
    assert np.array_equal(frame.y_true.to_numpy(np.float32), expected.y_true.to_numpy(np.float32))
    y, p = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
    if status == 'COMPLETE' and not np.isfinite(p).all():
        status = 'DEGENERATE'
    delta = {5: .098083, 10: .138578, 20: .195560}[job['h']] if setting == 'T2' else 1.
    frame['squared_loss'] = (p - y) ** 2
    frame['huber_loss'] = torch.nn.functional.huber_loss(torch.from_numpy(p), torch.from_numpy(y), delta=delta, reduction='none').numpy()
    frame['da_valid'] = y != 0
    frame['da_correct'] = np.where(frame.da_valid, (np.sign(p) == np.sign(y)).astype(float), np.nan)
    if status != 'COMPLETE':
        frame['da_correct'] = np.nan
    for key in ['setting', 'model', 'h', 'seed']:
        frame[key] = job[key]
    frame['status'] = status
    m = float(np.mean(ds.y[job['h']][ds.split == 'train'].astype(float)))
    ref_mse = float(np.mean((y - m) ** 2))
    ranks = common.rank_ic_by_date(y, p, frame.ticker.to_numpy(), frame.anchor_date.to_numpy()) if status == 'COMPLETE' else pd.DataFrame(dict(anchor_date=sorted(frame.anchor_date.unique()), rank_ic=np.nan))
    daily = frame.groupby('anchor_date', as_index=False).agg(mse=('squared_loss', 'mean'), huber_loss=('huber_loss', 'mean'),
                     n_valid=('da_valid', 'sum'), n_correct=('da_correct', lambda a: a.sum(min_count=1)))
    daily = daily.merge(ranks[['anchor_date', 'rank_ic']], on='anchor_date', validate='one_to_one')
    assert len(daily) == 391 and frame.groupby('anchor_date').size().eq(17).all()
    stocks = frame.groupby('ticker', as_index=False).agg(mse=('squared_loss', 'mean'), huber_loss=('huber_loss', 'mean'),
                    n_valid=('da_valid', 'sum'), n_correct=('da_correct', lambda a: a.sum(min_count=1)),
                    prediction_mean=('y_pred', 'mean'), realized_mean=('y_true', 'mean'))
    stocks['da'] = stocks.n_correct / stocks.n_valid
    stocks['forecast_bias'] = stocks.prediction_mean - stocks.realized_mean
    stocks['rank_ic'] = 'N/A: defined cross-sectionally per date'
    metrics = dict(**job, status=status, source=source, mse=float(frame.squared_loss.mean()) if status == 'COMPLETE' else None,
                   mean_rank_ic=float(daily.rank_ic.mean()) if daily.rank_ic.notna().any() else None,
                   da=float(daily.n_correct.sum() / daily.n_valid.sum()) if status == 'COMPLETE' else None,
                   r2os=1 - float(frame.squared_loss.mean()) / ref_mse if status == 'COMPLETE' else None,
                   global_train_mean=m, global_train_mean_test_mse=ref_mse, n_rows=len(frame),
                   n_valid_da=int((y != 0).sum()), n_zero_realized=int((y == 0).sum()))
    csv(folder / 'predictions_losses_weights.csv', frame)
    csv(folder / 'date_metrics.csv', daily)
    csv(folder / 'stock_metrics.csv', stocks)
    csv(folder / 'metrics.csv', pd.DataFrame([metrics]))
    router = []
    if job['model'] in ['static', 'adaptive']:
        for branch, names in [('daily', WEIGHTS[:4]), ('weekly', WEIGHTS[4:])]:
            weights = frame[names].to_numpy(float)
            finite = np.isfinite(weights).all()
            if finite:
                d = pd.DataFrame(weights)
                by = d.groupby(frame.anchor_date.to_numpy()).transform('mean')
                total, between = float(d.var().sum()), float(by.var().sum())
                span = float(np.ptp(weights, axis=0).max())
                constant = span < 1e-7
                mad, sd = float(np.abs(weights - .25).mean()), 0. if constant else float(weights.std(axis=0).mean())
                share = between / total if total > 0 and not constant else None
            else:
                total = between = span = mad = sd = share = None
                constant = False
            router.append(dict(branch=branch, mean_abs_deviation_from_equal=mad, mean_sample_sd=sd,
                               between_date_share=share, total_variance=total, between_date_variance=between,
                               max_scale_range=span, constant_to_1e7=constant))
    csv(folder / 'router_diagnostics.csv', pd.DataFrame(router, columns=['branch', 'mean_abs_deviation_from_equal', 'mean_sample_sd', 'between_date_share', 'total_variance', 'between_date_variance', 'max_scale_range', 'constant_to_1e7']))
    telemetry('test_evaluation_end', True)
    TELE.close()
    paths = [p for p in folder.iterdir() if p.is_file() and p.name != 'worker.log']
    js(folder / 'COMPLETE.json', dict(job=job, protocol_sha256=check_protocol(), status=status, started_utc=start,
       ended_utc=utc(), elapsed_utc_seconds=elapsed(start, utc()), monotonic_seconds=time.perf_counter()-clock,
       artifacts=[artifact(p) for p in paths]))

if __name__ == '__main__':
    action, encoded, destination = sys.argv[1:]
    job = json.loads(encoded)
    folder = target(Path(destination))
    folder.mkdir(parents=True, exist_ok=True)
    try:
        if action == 'train':
            train(job, folder)
        elif action == 'evaluate':
            evaluate(job, folder)
        else:
            raise ValueError(action)
    except Exception:
        text(folder / ('UNHANDLED_' + stamp() + '.txt'), traceback.format_exc())
        raise
