"""Step14A descriptive summary only; no training or statistical inference."""
from fullrun14a_worker import *
from report_stage3bc_final import old_predictions


def markdown(frame):
    def value(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.12g}' if np.isfinite(v) else 'NOT FOUND'
        return str(v)
    return table(list(frame.columns), [[value(v) for v in row] for row in frame.itertuples(index=False, name=None)])


def main():
    session_path = Path(sys.argv[1])
    session = json.loads((session_path / 'session.json').read_text())
    ledger = json.loads((session_path / 'queue_result.json').read_text())
    records, missing = [], []
    for spec in jobs():
        path, record = completed(spec)
        if record is None:
            missing.append(spec)
        else:
            records.append((path, record))
    by_id = {record['job_id']: (path, record) for path, record in records}
    old = old_predictions()
    rows, provenance = [], []
    original_e3 = pd.read_csv(ROOT / 'dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv')
    frozen = common.load_frozen_panel()
    test_mask = frozen.split == 'test'
    expected = canonical(pd.DataFrame(dict(ticker=frozen.ticker[test_mask], anchor_date=frozen.anchor_date[test_mask])))

    def add(feature, mechanism, h, seed, frame, source):
        frame = canonical(frame)
        assert frame[['ticker', 'anchor_date']].equals(expected[['ticker', 'anchor_date']])
        y = frame.y_true.to_numpy(np.float32)
        expected_y = canonical(pd.DataFrame(dict(ticker=frozen.ticker[test_mask], anchor_date=frozen.anchor_date[test_mask],
                                                 y_true=frozen.y[h][test_mask]))).y_true.to_numpy(np.float32)
        assert np.array_equal(y, expected_y)
        result = common.evaluate_predictions(y, frame.y_pred.to_numpy(np.float32), frame.ticker.to_numpy(), frame.anchor_date.to_numpy())
        rows.append(dict(input_set=feature, mechanism=mechanism, horizon=h, seed=seed,
                         test_mse=result['pooled_mse'], mean_rank_ic=result['cross_sectional_rank_ic']))
        provenance.append(dict(input_set=feature, mechanism=mechanism, horizon=h, seed=seed,
                               predictions=rel(source), sha256=sha(source), n_rows=len(frame)))

    for (mechanism, h, seed), frame in old.items():
        source = ROOT / ('dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv' if seed == 42 and mechanism == 'adaptive'
                         else 'dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv' if seed == 42
                         else 'dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv')
        add('O', mechanism, h, seed, frame, source)
    for mechanism in ['single', 'fixed']:
        for h in HORIZONS:
            frame = original_e3[original_e3.mechanism.eq(mechanism) & original_e3.horizon.eq(f'{h}d') & original_e3.seed.eq(42)]
            add('O', mechanism, h, 42, frame, ROOT / 'dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv')
    for path, record in records:
        if record['queue'] == 'Q0':
            continue  # Same archived seed42 cell; Q0 is an explicit qualification.
        frame_path = path.parent / 'test.parquet'
        add(record['input_set'], record['mechanism'], record['horizon'], record['seed'], pd.read_parquet(frame_path), frame_path)
    metrics_df = pd.DataFrame(rows).sort_values(['input_set', 'mechanism', 'horizon', 'seed'])
    assert not metrics_df.duplicated(['input_set', 'mechanism', 'horizon', 'seed']).any()
    summaries = []
    for feature in ['O', 'C']:
        for mechanism in ['single', 'fixed', 'static', 'adaptive']:
            for h in sorted(HORIZONS):
                group = metrics_df[metrics_df.input_set.eq(feature) & metrics_df.mechanism.eq(mechanism) & metrics_df.horizon.eq(h)]
                complete = len(group) == 5 and set(group.seed) == set(SEEDS)
                summaries.append(dict(input_set=feature, mechanism=mechanism, horizon=h, n_seeds=len(group),
                    five_seed_complete=complete, test_mse_mean=float(group.test_mse.mean()) if complete else float('nan'),
                    test_mse_sd_ddof1=float(group.test_mse.std(ddof=1)) if complete else float('nan'),
                    mean_rank_ic_mean=float(group.mean_rank_ic.mean()) if complete else float('nan'),
                    mean_rank_ic_sd_ddof1=float(group.mean_rank_ic.std(ddof=1)) if complete else float('nan')))
    summary_df = pd.DataFrame(summaries)
    comparisons = []
    for path, record in records:
        if record['input_set'] != 'C' or record['mechanism'] not in ['static', 'adaptive']:
            continue
        mechanism, h, seed = record['mechanism'], record['horizon'], record['seed']
        prefix = '' if seed == 42 else 'stage3b/'
        prior_path = causal_core.OUT / f'{prefix}preds/{mechanism}_dw_h{h}_s{seed}_test.parquet'
        prior = canonical(pd.read_parquet(prior_path))
        current = canonical(pd.read_parquet(path.parent / 'test.parquet'))
        assert prior[['ticker', 'anchor_date']].equals(current[['ticker', 'anchor_date']])
        assert np.array_equal(prior.y_true.to_numpy(), current.y_true.to_numpy())
        old_metric = common.evaluate_predictions(prior.y_true.to_numpy(), prior.y_pred.to_numpy(), prior.ticker.to_numpy(), prior.anchor_date.to_numpy())
        new_metric = common.evaluate_predictions(current.y_true.to_numpy(), current.y_pred.to_numpy(), current.ticker.to_numpy(), current.anchor_date.to_numpy())
        comparisons.append(dict(mechanism=mechanism, horizon=h, seed=seed,
            max_abs_delta_pred=float(np.max(np.abs(current.y_pred.to_numpy(np.float64) - prior.y_pred.to_numpy(np.float64)))),
            delta_mse_12_minus_3=new_metric['pooled_mse']-old_metric['pooled_mse'],
            delta_mean_rank_ic_12_minus_3=new_metric['cross_sectional_rank_ic']-old_metric['cross_sectional_rank_ic'],
            mse_12=new_metric['pooled_mse'], mse_3=old_metric['pooled_mse'], rank_ic_12=new_metric['cross_sectional_rank_ic'],
            rank_ic_3=old_metric['cross_sectional_rank_ic'], prior_predictions=rel(prior_path), current_predictions=rel(path.parent / 'test.parquet')))
    comparison_df = pd.DataFrame(comparisons).sort_values(['seed', 'horizon', 'mechanism']) if comparisons else pd.DataFrame()
    q0 = [dict(mechanism=record['mechanism'], horizon=record['horizon'], seed=42, **record['q0'])
          for _, record in records if record['queue'] == 'Q0']
    runtimes = [dict(job_id=record['job_id'], queue=record['queue'], pid=record['pid'], started_utc=record['started_utc'],
        completed_utc=record['completed_utc'], wall_seconds=record['wall_clock_seconds'], epochs=record['epochs'],
        best_epoch=record['best_epoch'], threads_start='/'.join(map(str, record['thread_counts_start'])),
        threads_end='/'.join(map(str, record['thread_counts_end'])), load_start=record['load_average_start'],
        load_end=record['load_average_end'], checkpoint=record['checkpoint'], completion=rel(path)) for path, record in records]
    summary_directory = OUT if not (OUT / 'REPORT_STEP14A.md').exists() else session_path
    data_directory = session_path / 'summary'
    for name, frame in [('cell_metrics.csv', metrics_df), ('five_seed_summary.csv', summary_df),
                        ('causal_12_vs_3.csv', comparison_df), ('prediction_sources.csv', pd.DataFrame(provenance)),
                        ('q0_comparison.csv', pd.DataFrame(q0)), ('run_telemetry.csv', pd.DataFrame(runtimes))]:
        with target(data_directory / name).open('x', newline='') as handle:
            frame.to_csv(handle, index=False)
    blocking = []
    if len(q0) != 2 or any(row['classification'] != 'EXACT' for row in q0):
        blocking.append('Q0 qualification did not pass for both cells. Q2/Q3 remain BLOCKED pending the user reply; Q1 was allowed to finish. No blocked Single/Fixed runs were launched.')
    if missing:
        blocking.append(f'{len(missing)} requested runs lack verified completion artifacts. Five-seed summaries for incomplete groups are NOT FOUND; available cells are listed separately.')
    preservation = json.loads((session_path / 'protected_inputs_verification.json').read_text())
    if not preservation['passed']:
        blocking.append('Protected-input hashes changed during the queue; see the preservation verification. No files were restored or overwritten.')
    power_events = json.loads((session_path / 'power_events.json').read_text())
    if any('AC Power' not in row['power'] for row in power_events):
        blocking.append('At least one scheduler power sample did not show AC power; see power_events.json.')
    queue_seconds = ledger['wall_clock_seconds']
    display = []
    for row in summaries:
        display.append(dict(input_set=row['input_set'], mechanism=row['mechanism'], horizon=row['horizon'], n_seeds=row['n_seeds'],
            test_MSE=f"{row['test_mse_mean']:.12g} ± {row['test_mse_sd_ddof1']:.12g}" if row['five_seed_complete'] else 'NOT FOUND',
            mean_Rank_IC=f"{row['mean_rank_ic_mean']:.12g} ± {row['mean_rank_ic_sd_ddof1']:.12g}" if row['five_seed_complete'] else 'NOT FOUND'))
    sections = ['# BLOCKING\n', '\n'.join('- '+v for v in blocking) if blocking else 'None. Both Q0 cells are EXACT and every requested run has verified artifacts.',
        '\n# Step14A — Core four-mechanism queue\n', f'Completed {utc()} UTC. Queue wall time {queue_seconds:.3f} seconds ({queue_seconds/3600:.3f} hours). Verified completions: {len(records)}/86. Existing artifacts remain read only; new artifacts use exclusive creation under `{rel(OUT)}/`.\n',
        '## Q0 qualification\n', markdown(pd.DataFrame(q0)),
        f'EXACT means max |Δpred| < {EXACT_THRESHOLD:g}; all other finite comparisons are DIFFERENT. Q0 uses O Fixed/Single seed42 h5 from the imported Experiment3 trainer and compares against the archived formal TEST CSV, with frozen labels and identical keys. No patch search: Single Daily5 / Weekly2 everywhere.\n',
        '## Queue, profile and telemetry\n',
        'The first two launches are Q0. A Q1 cell occupies the third slot while Q0 runs, because Q1 is permitted regardless of Q0 outcome. Q2/Q3 are gated until both Q0 cells finish EXACT. Thereafter the scheduler follows seed42,0,1,21,3407, horizon20,10,5, interleaving Q1, Q2 where applicable, and Q3. It keeps three training processes running whenever three eligible jobs remain; gate/tail underfill and all launches/exits are recorded. No filler or duplicate training jobs are created.\n',
        'Each worker removes OMP_NUM_THREADS and the other Step3B-unset BLAS overrides, sets OMP_DYNAMIC=FALSE and MKL_DYNAMIC=FALSE before importing Torch, and immediately sets Torch intra-op12 / inter-op12. All models run on CPU. Telemetry logs counts, environment, load averages, AC-power state and top20 CPU processes at start, every five minutes and end. Background load is recorded for timing; it is not asserted to explain numerical differences. This queue deliberately uses three concurrent 12/12 workers as requested.\n',
        'The scheduler holds `/usr/bin/caffeinate -i -w <scheduler-pid>` for training, verification and reporting. Power snapshots are logged throughout. The Mac started on AC power.\n',
        f'[Session]({session_path / "session.json"}), [launch/exit events]({session_path / "events.jsonl"}), [status ledger]({session_path / "status_reference.json"}), [queue result]({session_path / "queue_result.json"}), [power samples]({session_path / "power_events.json"}).\n',
        'Completed artifacts are hash-verified on restart and skipped. Partial attempts are retained; retries use a new attempt directory. Status is flushed/fsynced after each run to a newly created append-only journal, with an exclusive full status snapshot after every result; existing journals are never reopened for writing.\n',
        '## Imported paths and input identity\n',
        'O uses the original frozen V2 tensors. C imports `core.load_causal()`, exactly the dataset binding used by `stage3bc_worker.py`; normalization, labels, anchors, chronological splits, purge, model factories, training loops and metrics are imported unchanged. Preflight verifies the C normalized tensors bitwise against the preserved Stage3a tensors, and O/C keys, splits and labels exactly. Static seed42 uses Experiment3; Adaptive seed42 MAIN V2; the other Static/Adaptive seeds Experiment4. All Single/Fixed seeds use Experiment3. For non-42 Single/Fixed, process-local Experiment3.SEED changes model initialization/dropout; an adapter calls the original Experiment3 loader temporarily with SEED42 so its independent DataLoader generator remains42. The formal module seed and its source file remain unchanged.\n',
        'All runs retain Adam1e-4, batch32, Huberδ1, clip1, max100, patience10 and validation-Huber best-state selection. Selected best-state checkpoints are reloaded strictly into imported factories for imported batched inference on TRAIN/VAL/TEST. Each cell verifies 30,634 / 6,290 / 6,647 rows, original keys and labels, finite predictions and Parquet round trips. Adaptive router weights are stored as eight columns in every split Parquet; Static logits are saved separately. History contains train/validation Huber and validation MSE.\n',
        f'[Preflight]({OUT / "preflight.json"}), [protected-input verification]({session_path / "protected_inputs_verification.json"}), [cell artifact list]({data_directory / "run_telemetry.csv"}).\n',
        '## Five-seed TEST summaries\n',
        'Mean ± sample SD (ddof1), across seeds42,0,1,21,3407. Mean Rank IC is the formal mean of within-date cross-sectional Spearman correlations. O Static/Adaptive use archived formal predictions, including MAIN V2 Adaptive seed42. O Single/Fixed use archived formal seed42 plus the new four-seed runs; Q0 separately checks the seed42 training path. C uses only this queue’s new12/12 runs. Incomplete five-seed groups are NOT FOUND, never summarized as if five seeds existed.\n',
        markdown(pd.DataFrame(display)), f'[Numeric summary]({data_directory / "five_seed_summary.csv"}), [per-seed metrics]({data_directory / "cell_metrics.csv"}), [prediction sources/hashes]({data_directory / "prediction_sources.csv"}).\n',
        '## C Static/Adaptive —12/12 minus existing3/1, per cell\n',
        'Differences use aligned TEST predictions from the same input features/keys/labels and seed. The old3/1 seed42 cells are Stage3a; the other seeds are Stage3b. ΔMSE and ΔRank IC are new12/12 minus old3/1. These comparisons describe numerical-profile sensitivity, without attributing it to a specific thread component or scheduling component.\n',
        markdown(comparison_df[['mechanism','horizon','seed','max_abs_delta_pred','delta_mse_12_minus_3','delta_mean_rank_ic_12_minus_3']]) if len(comparison_df) else 'NOT FOUND',
        f'[Complete comparison table]({data_directory / "causal_12_vs_3.csv"}).\n',
        '## Per-run telemetry and checkpoint paths\n', markdown(pd.DataFrame(runtimes)),
        'No new bootstrap, significance test, portfolio analysis or intervention experiment was run. The frozen data remain ex-post snapshots; the causal feature construction does not authenticate point-in-time vendor data.\n']
    report_path = summary_directory / 'REPORT_STEP14A.md'
    textout(report_path, '\n'.join(sections))
    jsonout(session_path / 'report_manifest.json', dict(completed_utc=utc(), report=rel(report_path), report_sha256=sha(report_path),
        files=[dict(path=rel(p), sha256=sha(p)) for p in sorted(data_directory.iterdir())], n_verified_completions=len(records), blocking=blocking))
    print('REPORT COMPLETE', report_path, 'BLOCKING', blocking, flush=True)


if __name__ == '__main__':
    main()
