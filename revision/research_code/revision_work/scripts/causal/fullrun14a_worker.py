"""One Step14A training cell, delegating all training/model/data/metric paths."""
import os
import sys
sys.dont_write_bytecode = True
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.pop(key, None)
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['MKL_DYNAMIC'] = 'FALSE'
import torch
torch.set_num_threads(12)
torch.set_num_interop_threads(12)

from fullrun14a_io import *
import argparse
import platform
import subprocess
import time
import types
import numpy as np
import pandas as pd
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'revision_work/outputs/_deps_pyarrow15'))
from scripts.python import panel_common as common
from scripts.python import panel_pathformer_main_v2 as formal
from scripts.python import panel_pathformer_experiment3 as exp3
from scripts.python import panel_pathformer_experiment4_multiseed as exp4
import core as causal_core
from torch.utils.data import DataLoader


def power():
    return subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True, check=True).stdout.strip()


def check_profile():
    assert torch.get_num_threads() == 12 and torch.get_num_interop_threads() == 12
    assert os.environ.get('OMP_NUM_THREADS') is None
    assert os.environ['OMP_DYNAMIC'] == os.environ['MKL_DYNAMIC'] == 'FALSE'


def canonical(frame):
    frame = frame.copy()
    frame['anchor_date'] = pd.to_datetime(frame.anchor_date).dt.strftime('%Y-%m-%d')
    assert not frame.duplicated(['ticker', 'anchor_date']).any()
    return frame.sort_values(['ticker', 'anchor_date']).reset_index(drop=True)


class Telemetry:
    def __init__(self, path):
        self.handle = target(path).open('x')
        self.deadline = 0.
        self.rows = []

    def sample(self, event, force=False):
        now = time.perf_counter()
        if not force and now < self.deadline:
            return
        rows = []
        result = subprocess.run(['ps', '-axo', 'pid,ppid,pcpu,comm'], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines()[1:]:
            parts = line.strip().split(None, 3)
            if len(parts) == 4:
                rows.append(dict(pid=int(parts[0]), ppid=int(parts[1]), cpu_pct=float(parts[2]), command=parts[3]))
        row = dict(utc=utc(), event=event, pid=os.getpid(),
                   thread_counts=[torch.get_num_threads(), torch.get_num_interop_threads()],
                   environment={key: os.environ.get(key) for key in ENV_KEYS},
                   load_average_1_5_15=list(os.getloadavg()), power=power(),
                   top_cpu_processes=sorted(rows, key=lambda v: v['cpu_pct'], reverse=True)[:20])
        self.handle.write(json.dumps(row) + '\n')
        self.handle.flush()
        self.rows.append(row)
        self.deadline = now + 300.
        check_profile()
        print('TELEMETRY', event, row['utc'], row['thread_counts'], row['load_average_1_5_15'], flush=True)


def dependencies(input_set):
    paths = [Path(formal.__file__), Path(exp3.__file__), Path(exp4.__file__), Path(common.__file__),
             Path(causal_core.__file__), Path(__file__), HERE / 'fullrun14a_io.py',
             ROOT / 'dataset/audit/experiment3_pathformer_ablation/experiment3_single_selection.csv']
    paths += [ROOT / 'revision_work/outputs/causal_features/tensors' / name for name in
              ['sample_index.csv', 'X_daily_causal_raw.npy', 'X_weekly_causal_raw.npy',
               'y_h5_frozen.npy', 'y_h10_frozen.npy', 'y_h20_frozen.npy']] if input_set == 'C' else []
    return [dict(path=rel(path), sha256=sha(path)) for path in paths]


def dataset(input_set):
    # This is precisely Stage3b's binding, imported rather than reimplemented.
    return causal_core.load_causal() if input_set == 'C' else common.load_frozen_panel()


def preflight():
    check_profile()
    exp3._verify_frozen_dependencies()
    exp4._verify_frozen_dependencies()
    exp4._verify_frozen_training_constants()
    selection = pd.read_csv(ROOT / 'dataset/audit/experiment3_pathformer_ablation/experiment3_single_selection.csv')
    assert len(selection) == 3
    assert selection.selected_daily_patch.eq(5).all() and selection.selected_weekly_patch.eq(2).all()
    original = dataset('O')
    causal = dataset('C')
    for ds in [original, causal]:
        formal._formal_check_dataset_contract(ds)
        assert {split: int((ds.split == split).sum()) for split in COUNTS} == COUNTS
    for field in ['ticker', 'anchor_date', 'split']:
        assert np.array_equal(getattr(original, field), getattr(causal, field)), field
    for h in HORIZONS:
        assert np.array_equal(original.y[h], causal.y[h])
    normalized = common.apply_normalizer(causal, common.fit_train_only_normalizer(causal), 'daily_weekly')
    for resolution, actual in zip(['daily', 'weekly'], normalized):
        saved = np.load(causal_core.OUT / f'tensors/X_{resolution}_causal_standardized.npy')
        assert np.array_equal(actual, saved), resolution
    # Verify the seed adapter separately without training or consuming the global RNG.
    saved_seed = exp3.SEED
    exp3.SEED = 3407
    actual_seed = exp3.SEED
    exp3.SEED = 42
    train, val = exp3._build_train_val_loaders_only(5, original)
    exp3.SEED = actual_seed
    assert train.generator.initial_seed() == 42 and exp3.SEED == 3407
    exp3.SEED = saved_seed
    assert 'AC Power' in power(), 'Mac must be connected to AC power before queue launch'
    report = dict(completed_utc=utc(), passed=True, shape_daily=list(original.X_daily.shape),
                  shape_weekly=list(original.X_weekly.shape), row_counts=COUNTS,
                  keys_labels_equal=True, causal_standardization_bitwise_equal=True,
                  selection_daily=5, selection_weekly=2, loader_generator_seed=42,
                  torch_threads=12, torch_interop_threads=12, environment={key: os.environ.get(key) for key in ENV_KEYS},
                  power=power(), versions=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
                  dependencies=dependencies('C'), parameter_counts={})
    for mechanism in ['single', 'fixed', 'static', 'adaptive']:
        model = exp3._make_experiment3_model(mechanism, 5 if mechanism == 'single' else None,
                                            2 if mechanism == 'single' else None) if mechanism != 'adaptive' else formal._make_model('daily_weekly', 'adaptive')
        report['parameter_counts'][mechanism] = formal.count_trainable_parameters(model)
    jsonout(OUT / 'preflight.json', report)
    print('PREFLIGHT PASS', report['parameter_counts'], flush=True)


def run(spec, attempt):
    check_profile()
    start = time.perf_counter()
    started = utc()
    exp3._verify_frozen_dependencies()
    exp4._verify_frozen_dependencies()
    exp4._verify_frozen_training_constants()
    ds = dataset(spec['input_set'])
    formal._formal_check_dataset_contract(ds)
    assert {s: int((ds.split == s).sum()) for s in COUNTS} == COUNTS
    h, mechanism, seed = spec['horizon'], spec['mechanism'], spec['seed']
    use_exp4 = mechanism in ['static', 'adaptive'] and seed != 42
    owner = exp4 if use_exp4 else (formal if mechanism == 'adaptive' else exp3)
    factory_name = '_make_experiment4_model' if use_exp4 else ('_make_model' if owner is formal else '_make_experiment3_model')
    factory = getattr(owner, factory_name)
    checkpoint = attempt / 'best_state.pt'
    models, handles = [], []
    val_mse = []
    validation = dict(n=0, sse=0.)
    phase = dict(last=None, epoch=0)
    telemetry = Telemetry(attempt / 'telemetry.jsonl')
    telemetry.sample('start', True)
    epoch_stream = target(attempt / 'epoch_progress.jsonl').open('x')
    deps = dependencies(spec['input_set'])
    original_huber, original_save = torch.nn.HuberLoss, torch.save
    original_load = owner.load_frozen_panel
    original_seed = exp3.SEED
    original_loader = exp3._build_train_val_loaders_only
    original_dir = exp4.EXPERIMENT_DIR
    saves = owner is not formal
    original_path = owner._checkpoint_path if saves else None
    original_os = owner.os if saves else None

    def capture(*args, **kwargs):
        model = factory(*args, **kwargs)
        models.append(model)
        def progress(module, inputs):
            telemetry.sample('five_minute')
            status = 'TRAIN' if module.training else 'VALIDATION'
            if status != phase['last']:
                if status == 'TRAIN':
                    phase['epoch'] += 1
                phase['last'] = status
                record = dict(utc=utc(), epoch=phase['epoch'], phase=status)
                epoch_stream.write(json.dumps(record) + '\n')
                epoch_stream.flush()
                print('PHASE', spec['job_id'], record, flush=True)
        handles.append(model.register_forward_pre_hook(progress))
        return model

    def observed_huber(*args, **kwargs):
        criterion = original_huber(*args, **kwargs)
        def observe(module, inputs, output):
            if not models[0].training:
                pred, y = inputs
                difference = pred.detach() - y.detach()
                validation['sse'] += float((difference * difference).sum(dtype=torch.float64))
                validation['n'] += len(y)
                if validation['n'] == COUNTS['val']:
                    val_mse.append(validation['sse'] / validation['n'])
                    validation.update(n=0, sse=0.)
                    print('VAL_MSE', spec['job_id'], len(val_mse), val_mse[-1], flush=True)
                elif validation['n'] > COUNTS['val']:
                    raise RuntimeError('Unexpected validation count')
        handles.append(criterion.register_forward_hook(observe))
        return criterion

    def seeded_loader(*args, **kwargs):
        # exp3.SEED controls initialization AND its private loader generator.
        # Change the former only; call the original loader with generator=42.
        model_seed = exp3.SEED
        try:
            exp3.SEED = 42
            loaders = original_loader(*args, **kwargs)
        finally:
            exp3.SEED = model_seed
        assert loaders[0].generator.initial_seed() == 42
        return loaders

    def save_exclusive(obj, path, *args, **kwargs):
        with target(Path(path)).open('xb') as handle:
            original_save(obj, handle, *args, **kwargs)

    def publish_exclusive(source, destination):
        # Retain the newly created temporary artifact; never rename/delete files.
        with Path(source).open('rb') as inp, target(Path(destination)).open('xb') as out:
            for block in iter(lambda: inp.read(1024 * 1024), b''):
                out.write(block)

    setattr(owner, factory_name, capture)
    owner.load_frozen_panel = lambda: ds
    torch.nn.HuberLoss = observed_huber
    if owner is exp3:
        exp3.SEED = seed
        exp3._build_train_val_loaders_only = seeded_loader
    if saves:
        owner._checkpoint_path = lambda *args, **kwargs: checkpoint
        exp4.EXPERIMENT_DIR = attempt
        torch.save = save_exclusive
        owner.os = types.SimpleNamespace(**{key: getattr(os, key) for key in dir(os) if not key.startswith('__')})
        owner.os.replace = publish_exclusive
    print('START', started, spec, flush=True)
    try:
        if use_exp4:
            result = exp4.train_and_validate_cell(mechanism, f'{h}d', seed)
            entrypoint = 'experiment4.train_and_validate_cell'
        elif owner is formal:
            result = formal.train_one_cell('daily_weekly', h, 'adaptive')
            entrypoint = 'main_v2.train_one_cell'
        else:
            result = exp3.train_and_validate_cell(mechanism, f'{h}d', 5 if mechanism == 'single' else None, 2 if mechanism == 'single' else None)
            entrypoint = 'experiment3.train_and_validate_cell'
    finally:
        setattr(owner, factory_name, factory)
        owner.load_frozen_panel = original_load
        torch.nn.HuberLoss, torch.save = original_huber, original_save
        exp3.SEED = original_seed
        exp3._build_train_val_loaders_only = original_loader
        exp4.EXPERIMENT_DIR = original_dir
        if saves:
            owner._checkpoint_path, owner.os = original_path, original_os
        for handle in handles:
            handle.remove()
        epoch_stream.close()
    assert len(models) == 1
    model = models[0]
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if not saves:
        with target(checkpoint).open('xb') as handle:
            original_save(state, handle)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved.keys() == state.keys() and all(torch.equal(saved[key], value) for key, value in state.items())
    summary = result['summary'].iloc[0].to_dict() if isinstance(result['summary'], pd.DataFrame) else result['summary']
    history = pd.DataFrame(result['history'])
    assert len(history) == len(val_mse) == int(summary['epochs_trained'])
    assert history.epoch.tolist() == list(range(1, len(history) + 1))
    history['train_huber'], history['val_huber'] = history.train_loss, history.val_loss
    history['val_mse'], history['loss_type'] = val_mse, 'Huber delta=1'
    assert np.isfinite(history[['train_huber', 'val_huber', 'val_mse']]).all().all()
    with target(attempt / 'history.csv').open('x', newline='') as handle:
        history.to_csv(handle, index=False)
    with target(attempt / 'trainer_summary.csv').open('x', newline='') as handle:
        pd.DataFrame([summary]).to_csv(handle, index=False)
    if mechanism == 'static':
        logits = {key: value.tolist() for key, value in state.items() if 'logit' in key}
        assert len(logits) == 2
        jsonout(attempt / 'static_logits.json', logits)
    replay = exp4._make_experiment4_model(mechanism) if use_exp4 else (
        formal._make_model('daily_weekly', 'adaptive') if mechanism == 'adaptive' else
        exp3._make_experiment3_model(mechanism, 5 if mechanism == 'single' else None, 2 if mechanism == 'single' else None))
    replay.load_state_dict(saved, strict=True)
    replay.eval()
    train, val, test = formal._build_training_loaders('daily_weekly', h, ds)
    assert train.generator.initial_seed() == 42
    loaders = {'train': DataLoader(train.dataset, batch_size=formal.BATCH_SIZE, shuffle=False), 'val': val, 'test': test}
    metric_rows = []
    test_frame = None
    for split, loader in loaders.items():
        mask = ds.split == split
        if use_exp4:
            pred, daily, weekly = exp4._batched_inference(replay, loader, COUNTS[split], torch.device('cpu'))
        else:
            pred, daily, weekly = exp3._batched_inference(replay, loader, mechanism,
                5 if mechanism == 'single' else None, 2 if mechanism == 'single' else None, COUNTS[split], torch.device('cpu'))
        frame = pd.DataFrame(dict(ticker=ds.ticker[mask], anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),
                                  split=split, y_true=ds.y[h][mask], y_pred=pred))
        for index, column in enumerate(WEIGHTS):
            frame[column] = (daily if index < 4 else weekly)[:, index % 4]
        assert len(frame) == COUNTS[split] and not frame.duplicated(['ticker', 'anchor_date']).any()
        assert np.isfinite(frame[['y_true', 'y_pred'] + WEIGHTS]).all().all()
        assert np.array_equal(frame.ticker.to_numpy(), ds.ticker[mask])
        assert np.array_equal(pd.to_datetime(frame.anchor_date).to_numpy(), pd.to_datetime(ds.anchor_date[mask]).to_numpy())
        assert np.array_equal(frame.y_true.to_numpy(), ds.y[h][mask])
        assert np.allclose(daily.sum(1), 1, rtol=0, atol=1e-6) and np.allclose(weekly.sum(1), 1, rtol=0, atol=1e-6)
        if mechanism != 'adaptive':
            assert np.ptp(frame[WEIGHTS].to_numpy(), axis=0).max() < 1e-7
        path = attempt / f'{split}.parquet'
        with target(path).open('xb') as handle:
            handle.write(frame.to_parquet(index=False, engine='pyarrow'))
        pd.testing.assert_frame_equal(frame, pd.read_parquet(path))
        metric_rows.append(dict(split=split, **common.evaluate_predictions(frame.y_true.to_numpy(), pred,
                               frame.ticker.to_numpy(), frame.anchor_date.to_numpy())))
        if split == 'test':
            test_frame = frame
    with target(attempt / 'metrics.csv').open('x', newline='') as handle:
        pd.DataFrame(metric_rows).to_csv(handle, index=False)
    q0 = None
    if spec['queue'] == 'Q0':
        archive_path = ROOT / 'dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv'
        old = pd.read_csv(archive_path)
        old = canonical(old[old.mechanism.eq(mechanism) & old.horizon.eq('5d') & old.seed.eq(42)])
        new = canonical(test_frame)
        assert len(old) == COUNTS['test'] and old[['ticker', 'anchor_date']].equals(new[['ticker', 'anchor_date']])
        assert np.array_equal(old.y_true.to_numpy(np.float32), new.y_true.to_numpy(np.float32))
        delta = float(np.max(np.abs(new.y_pred.to_numpy(np.float64) - old.y_pred.to_numpy(np.float64))))
        old_mse = float(np.mean((old.y_pred.to_numpy(np.float32) - old.y_true.to_numpy(np.float32)) ** 2))
        q0 = dict(classification='EXACT' if delta < EXACT_THRESHOLD else 'DIFFERENT',
                  threshold=EXACT_THRESHOLD, max_abs_delta_pred=delta,
                  delta_mse=metric_rows[-1]['pooled_mse']-old_mse,
                  archived_predictions=rel(archive_path), archived_sha256=sha(archive_path))
        jsonout(attempt / 'q0_comparison.json', q0)
    telemetry.sample('end', True)
    telemetry.handle.close()
    assert all(row['thread_counts'] == [12, 12] for row in telemetry.rows)
    for dep in deps:
        assert sha(ROOT / dep['path']) == dep['sha256'], dep['path']
    artifacts = [dict(path=rel(path), bytes=path.stat().st_size, sha256=sha(path))
                 for path in sorted(attempt.iterdir()) if path.is_file() and path.name != 'worker.log']
    record = dict(**spec, started_utc=started, completed_utc=utc(), wall_clock_seconds=time.perf_counter()-start,
                  pid=os.getpid(), hostname=platform.node(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                  training_entrypoint=entrypoint, loader_generator_seed=42, daily_patch=5 if mechanism == 'single' else None,
                  weekly_patch=2 if mechanism == 'single' else None, best_epoch=int(summary['best_epoch']),
                  epochs=int(summary['epochs_trained']), formal_training_seconds=float(summary['runtime_seconds']),
                  row_counts=COUNTS, checkpoint=rel(checkpoint), checkpoint_sha256=sha(checkpoint),
                  thread_counts_start=telemetry.rows[0]['thread_counts'], thread_counts_end=telemetry.rows[-1]['thread_counts'],
                  thread_counts_changed=False, environment_start=telemetry.rows[0]['environment'],
                  environment_end=telemetry.rows[-1]['environment'],
                  load_average_start=telemetry.rows[0]['load_average_1_5_15'], load_average_end=telemetry.rows[-1]['load_average_1_5_15'],
                  power_start=telemetry.rows[0]['power'], power_end=telemetry.rows[-1]['power'],
                  telemetry_samples=len(telemetry.rows), metrics=metric_rows, q0=q0, dependencies=deps,
                  validated=True, artifacts=artifacts)
    jsonout(attempt / 'COMPLETE.json', record)
    verify_completion(attempt / 'COMPLETE.json', spec)
    print('COMPLETE', spec['job_id'], record['wall_clock_seconds'], 'Q0', q0, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--job-id')
    parser.add_argument('--attempt')
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return
    selected = [row for row in jobs() if row['job_id'] == args.job_id]
    assert len(selected) == 1
    attempt = Path(args.attempt).absolute()
    assert attempt.is_relative_to(OUT / 'runs' / args.job_id)
    try:
        run(selected[0], attempt)
    except BaseException as exc:
        if not (attempt / 'FAILED.json').exists():
            jsonout(attempt / 'FAILED.json', dict(utc=utc(), job=selected[0], error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
