"""Post-outcome independent read-only audit of the first H1 acquisition pilot."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

BASE = Path('/workspace/scratch/7adff8a3b4a7')
RUN = BASE / 'DoVIS-H1-acquisition-smoke-01'
STAGE = BASE / 'DoVIS-resumed/experiments/h1-acquisition'
CORE = STAGE.parent / 'mutual-understanding'
OUTPUT = BASE / 'DoVIS-H1-acquisition-smoke-01-independent-audit.json'
sys.path[:0] = [str(STAGE), str(CORE)]
CONDITIONS = ('aligned', 'no_update', 'shuffled', 'reset_understanding')
LANGUAGES = ('ko', 'ja')
STOPS = (0, 8, 16, 32)
REVISION = '99abc1ca0575dc7ce75709c7e00999ba9c5cc4a7'
checks = 0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def value_sha(value):
    return hashlib.sha256(canon(value).encode()).hexdigest()


def require(condition, message):
    global checks
    checks += 1
    if not condition:
        raise ValueError(message)


def identical(a, b, message):
    require(canon(a) == canon(b), message)


def expected_counts(stream, permutation, condition, language, exchange):
    peer = next(l for l in LANGUAGES if l != language)
    texts = [stream[permutation[i] if condition == 'shuffled' else i]['utterances'][peer]['text']
             for i in range(exchange)]
    updates = 0 if condition == 'no_update' else exchange
    return {'exposures': exchange, 'updates': updates,
            'characters': sum(len(t) for t in texts),
            'bytes_received': sum(len(t.encode('utf-8')) for t in texts),
            'input_units': sum(len(t.encode('utf-8')) for t in texts),
            'training_sequence_units': sum(len(t.encode('utf-8')) + 2 for t in texts) if updates else 0,
            'native_replay_exposures': 0}


def query_identity(row):
    return {k: row[k] for k in ('kind', 'scene_key', 'text', 'template', 'answer', 'choices',
                               'learner', 'peer', 'exchange')}


def audit():
    from mutual_lab.config import ExperimentConfig
    from mutual_lab.development_world import development_splits, development_manifest, render_scene, captions, grade_generation
    from mutual_lab.world import variants
    from mutual_lab.neural import torch, digest, configure_cpu
    from h1_adapter import H1DevelopmentLearner, SCOPE

    configure_cpu()
    summary = load(RUN / 'summary.json')
    plan_path = STAGE / 'configs/development-smoke.json'
    plan = load(plan_path)
    require(summary['status'] == 'completed' and summary['error'] is None, 'Pilot is not completed successfully')
    require(summary['revision'] == REVISION, 'Preparation revision differs')
    require(summary['excluded_from_research'] is True and summary['eligible_for_fresh_experiment'] is False
            and summary['eligible_for_l2_research'] is False and summary['eligible_for_l2'] is False
            and summary['protected_images_rendered_or_scored'] is False, 'Pilot scope differs')
    require(summary['all_mechanistic_checks_passed'] is True
            and summary['model_quality_gain_required_for_smoke_pass'] is False, 'Procedural/quality distinction differs')
    require(sha(plan_path) == summary['plan_sha256'], 'Frozen plan changed')
    identical(plan['languages'], list(LANGUAGES), 'Language choice differs')
    identical(plan['conditions'], list(CONDITIONS), 'Condition choice differs')
    identical(plan['evaluate_at'], list(STOPS), 'Evaluation budget differs')
    require(plan['exposures'] == 32 and plan['optimizer']['updates_per_exposure'] == 1
            and plan['optimizer']['native_replay_per_update'] == plan['optimizer']['foreign_replay_per_update'] == 0,
            'Learning budget differs')
    core_manifest = STAGE / 'configs/frozen-core-source-sha256.json'
    require(sha(core_manifest) == summary['source_verification']['manifest_sha256'], 'Core manifest changed')
    core_files = load(core_manifest)['files']
    require(len(core_files) == 42 and set(core_files) == {p.name for p in (CORE / 'mutual_lab').glob('*.py')},
            'Frozen core file set differs')
    for name, wanted in core_files.items():
        require(sha(CORE / 'mutual_lab' / name) == wanted, 'Frozen core source changed: ' + name)
    for name, wanted in summary['stage_source_sha256'].items():
        require(sha(STAGE / name) == wanted, 'Frozen stage source changed: ' + name)
    require(sha(BASE / 'DoVIS-deliverables' / plan['source']['archive_filename']) == plan['source']['archive_sha256'],
            'Original delivered archive changed')

    config_path = RUN / 'inputs/ko/config.json'
    require(sha(config_path) == plan['source']['blueprint_sha256'], 'Blueprint changed')
    config = ExperimentConfig.load(config_path)
    splits = development_splits(config, plan['data_seed'])
    manifest = development_manifest(config, plan['data_seed'])
    semantic = load(RUN / 'semantic-plan.json')
    identical(semantic['development_manifest'], manifest, 'Development manifest differs')
    require(manifest['protected_pairs_excluded'] == 22 and manifest['pair_sha256'] == plan['data_pair_sha256'],
            'Protected exclusion or data pin differs')
    require(len(splits['train']) == 220 and len(splits['dev']) == 112
            and not ({s.group for s in splits['train']} & {s.group for s in splits['dev']}),
            'Development split population/overlap differs')
    scenes = random.Random(plan['seeds']['stream']).sample(list(splits['train']), 32)
    identical(semantic['shared_scene_keys'], [s.key for s in scenes], 'Exposure stream differs')
    require(len(set(semantic['shared_scene_keys'])) == 32, 'Exposure scenes duplicated')
    permutation_rng = random.Random(plan['seeds']['shuffle'])
    for _ in range(10000):
        permutation = list(range(32)); permutation_rng.shuffle(permutation)
        if all(i != j for i, j in enumerate(permutation)):
            break
    else:
        raise ValueError('No independently recreated derangement')
    identical(semantic['shuffled_text_indices'], permutation, 'Shuffle permutation differs')
    require(sorted(permutation) == list(range(32)), 'Shuffled text marginals changed')
    evaluation_rng = random.Random(plan['seeds']['evaluation'])
    evaluation = []
    for scene in splits['dev']:
        choices = list(variants(scene.group)); evaluation_rng.shuffle(choices)
        evaluation.append((scene, choices, choices.index(scene)))
    identical(semantic['evaluation'], [{'scene_key': s.key, 'choices': [c.key for c in cs], 'answer': a}
                                     for s, cs, a in evaluation], 'Evaluation choice/answer stream differs')
    pngs = {s.key: render_scene(s) for split in splits.values() for s in split}
    require(len(pngs) == 332 and semantic['protected_images_rendered_or_scored'] is False, 'Wrong image pool')
    identical(semantic['png_sha256'], {k: hashlib.sha256(v).hexdigest() for k, v in pngs.items()}, 'PNG bytes differ')
    require(value_sha(semantic) == summary['semantic_plan_sha256'], 'Semantic plan hash differs')

    stream = load(RUN / 'frozen-peer-stream.json')
    frozen_dev = load(RUN / 'frozen-peer-evaluation.json')
    require(len(stream) == 32 and value_sha(stream) == summary['stream_sha256'], 'Peer stream differs')
    require(value_sha(frozen_dev) == summary['frozen_peer_eval_sha256'], 'Frozen peer evaluation changed')
    for language in LANGUAGES:
        require(set(frozen_dev[language]) == {s.key for s in splits['dev']}, 'Frozen speaker eval scene set differs')
    for i, entry in enumerate(stream):
        require(entry['scene_key'] == scenes[i].key and set(entry['utterances']) == set(LANGUAGES), 'Peer exposure identity differs')
        for language in LANGUAGES:
            identical(entry['researcher_only_semantic_grades'][language],
                      grade_generation(entry['utterances'][language], scenes[i], language), 'Peer semantic annotation differs')
    collisions = load(RUN / 'shuffled-association-collisions.json')
    for language in LANGUAGES:
        expected = {'items': 32,
                    'same_surface_as_aligned': sum(stream[permutation[i]]['utterances'][language]['text']
                                                   == stream[i]['utterances'][language]['text'] for i in range(32)),
                    'recognized_meaning_matches_shared_scene': sum(
                        stream[permutation[i]]['researcher_only_semantic_grades'][language]['recognized_meaning'] == scenes[i].to_dict()
                        for i in range(32)), 'rows_removed_or_repaired': 0}
        identical(collisions[language], expected, 'Shuffled collision annotation differs')
    generation = load(RUN / 'native-generation-baseline.json')
    for language in LANGUAGES:
        rows = [grade_generation(frozen_dev[language][s.key], s, language) for s in splits['dev']]
        identical(generation[language]['rows'], rows, 'Baseline generation rows differ')
        hits = sum(r['exact_match'] and r['terminated'] and not r['invalid_encoding']
                   and not r['special_tokens_in_body'] for r in rows)
        require(generation[language]['hits'] == hits and generation[language]['items'] == 112
                and generation[language]['accuracy'] == hits / 112, 'Generation summary differs')

    expected_rows = []
    for exchange in STOPS:
        for language in LANGUAGES:
            peer = next(l for l in LANGUAGES if l != language)
            for scene, choices, answer in evaluation:
                for kind, texts in (('native_authored', captions(scene, language)),
                                    ('foreign_authored', captions(scene, peer)),
                                    ('foreign_peer_generated', [frozen_dev[peer][scene.key]['text']])):
                    for template, text in enumerate(texts):
                        expected_rows.append({'kind': kind, 'scene_key': scene.key, 'text': text, 'template': template,
                                              'answer': answer, 'choices': [s.key for s in choices],
                                              'learner': language, 'peer': peer, 'exchange': exchange})
    require(len(expected_rows) == 8064, 'Independent expected row count differs')
    expected_identity = Counter(canon(r) for r in expected_rows)
    condition_records, raw_by_condition, all_points = {}, {}, []
    for condition in CONDITIONS:
        condition_summary = load(RUN / f'{condition}-summary.json')
        identical(condition_summary, summary['conditions'][condition], 'Embedded condition summary differs')
        require(condition_summary['status'] == 'completed' and condition_summary['development_only'] is True,
                'Condition is not complete development-only')
        rows = load(RUN / f'{condition}-predictions.json')
        require(len(rows) == 8064 and Counter(canon(query_identity(r)) for r in rows) == expected_identity,
                'Missing/duplicate/changed prediction queries in ' + condition)
        index = {}
        for row in rows:
            require(type(row['prediction']) is int and 0 <= row['prediction'] < 4
                    and type(row['correct']) is bool and type(row['answer']) is int
                    and type(row['template']) is int and type(row['exchange']) is int,
                    'Malformed prediction identity')
            require(row['correct'] is (row['prediction'] == row['answer']), 'Wrong raw correctness flag')
            index[canon(query_identity(row))] = row
        points = []
        for exchange in STOPS:
            for language in LANGUAGES:
                peer = next(l for l in LANGUAGES if l != language)
                for kind in ('native_authored', 'foreign_authored', 'foreign_peer_generated'):
                    selected = [r for r in rows if r['exchange'] == exchange and r['learner'] == language and r['kind'] == kind]
                    hits = sum(r['prediction'] == r['answer'] for r in selected)
                    accuracy = hits / len(selected)
                    points.append({'learner': language, 'target_language': language if kind == 'native_authored' else peer,
                                   'exchange': exchange, 'metric': kind, 'hits': hits, 'items': len(selected),
                                   'raw_accuracy': accuracy, 'chance_corrected_accuracy': (accuracy - .25) / .75,
                                   'choice_count': 4, 'baseline_forced_to_zero': False})
        identical(condition_summary['points'], points, 'Wrong aggregate counts/chance correction: ' + condition)
        all_points.extend({**p, 'condition': condition} for p in points)
        events = load(RUN / f'{condition}-events.json')
        require(len(events) == 32, 'Missing/extra training exposure events')
        for i, event in enumerate(events):
            require(type(event['exchange']) is int and event['exchange'] == i + 1
                    and event['shared_scene_key'] == scenes[i].key
                    and event['utterance_source_index'] == (permutation[i] if condition == 'shuffled' else i),
                    'Condition exposure stream/association differs')
            require(set(event['updates']) == set(LANGUAGES), 'Missing peer direction')
            for language in LANGUAGES:
                update = event['updates'][language]
                require(all(update.get(k) == v and type(update.get(k)) is type(v) for k, v in SCOPE.items()),
                        'Update history scope differs')
                identical(update['counters'], expected_counts(stream, permutation, condition, language, i + 1),
                          'Event exposure counters differ')
                require(len(update['losses']) == (0 if condition == 'no_update' else 1)
                        and all(type(v) is float and math.isfinite(v) for v in update['losses']), 'Wrong optimizer update event')
        for language in LANGUAGES:
            identical(condition_summary['counters'][language], expected_counts(stream, permutation, condition, language, 32),
                      'Final summary counters differ')
        require(condition_summary['all_evaluations_read_only'] is True
                and all(condition_summary['frozen_producer_outputs_equal'].values()), 'Reported invariants not passed')
        condition_records[condition] = {'rows_regraded': len(rows), 'points_regraded': len(points),
                                        'exposure_events_verified': len(events), 'directions': {}}
        for language in LANGUAGES:
            condition_records[condition]['directions'][language] = {
                kind: [p['hits'] for p in points if p['learner'] == language and p['metric'] == kind]
                for kind in ('native_authored', 'foreign_authored', 'foreign_peer_generated')}
        raw_by_condition[condition] = index
    identical(load(RUN / 'learning-curves.json'), all_points, 'Combined learning curve differs')
    require(len(all_points) == 96, 'Wrong aggregate point count')
    for condition in ('no_update', 'shuffled'):
        for q in expected_rows:
            if q['exchange'] == 0:
                identical(raw_by_condition[condition][canon(q)], raw_by_condition['aligned'][canon(q)], 'Matched baseline differs')
    for q in expected_rows:
        first = {**q, 'exchange': 0}
        require(raw_by_condition['no_update'][canon(q)]['prediction'] == raw_by_condition['no_update'][canon(first)]['prediction'],
                'No-update predictions changed')

    checkpoint_records = []
    restored_predictions = 0
    frozen_outputs_checked = 0
    cached_reference = {}
    normal_initial_baseline_checked = False
    for condition in CONDITIONS:
        mode = 'reset' if condition == 'reset_understanding' else 'normal'
        if mode not in cached_reference:
            learners = {language: H1DevelopmentLearner.from_checkpoint(
                RUN / f'inputs/{language}/step-009600.pt', RUN / 'inputs/prior/prior.pt',
                allow_development_training=True, language=language,
                learning_rate=plan['optimizer']['learning_rate'], seed=plan['seeds']['learner'],
                reset_understanding=condition == 'reset_understanding') for language in LANGUAGES}
            for learner in learners.values():
                learner.warm(list(pngs.values()))
            initial_branches = {l: a.branch_digests() for l, a in learners.items()}
            canonical_cache = {l: a.state_dict()['feature_caches'] for l, a in learners.items()}
            initial_weights = {l: a.state_dict()['weights'] for l, a in learners.items()}
            for language, learner in learners.items():
                baseline = [q for q in expected_rows if q['exchange'] == 0 and q['learner'] == language]
                before, cache_before = learner.state_digest(), learner.cache_digest()
                for off in range(0, len(baseline), 32):
                    batch = baseline[off:off + 32]
                    predictions = learner.predict_many([q['text'] for q in batch],
                        [[pngs[k] for k in q['choices']] for q in batch])
                    for q, prediction in zip(batch, predictions):
                        require(prediction == raw_by_condition[condition][canon(q)]['prediction'], 'Fresh baseline inference differs')
                    restored_predictions += len(batch)
                require(learner.state_digest() == before and learner.cache_digest() == cache_before, 'Baseline inference mutated state')
            cached_reference[mode] = (learners, initial_branches, canonical_cache, initial_weights)
        learners, initial_branches, canonical_cache, initial_weights = cached_reference[mode]
        identical(summary['conditions'][condition]['initial_branch_sha256'], initial_branches, 'Initial branch identity differs')
        for exchange in (16, 32):
            path = RUN / f'checkpoints/{condition}-e{exchange:06d}.pt'
            sidecar = load(path.with_suffix('.json'))
            require(sidecar['format'] == 'h1-development-pair-acquisition-v1'
                    and sidecar['kind'] == 'development_acquisition_pair' and sidecar['excluded_from_research'] is True
                    and sidecar['eligible_for_fresh_experiment'] is False and sidecar['eligible_for_l2_research'] is False
                    and sidecar['eligible_for_l2'] is False and sidecar['sha256'] == sha(path), 'Invalid actual pair checkpoint')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            expected_metadata = {'condition': condition, 'exchange': exchange, 'plan_sha256': summary['plan_sha256'],
                                 'stream_sha256': summary['stream_sha256'], 'semantic_plan_sha256': summary['semantic_plan_sha256'],
                                 'frozen_peer_eval_sha256': summary['frozen_peer_eval_sha256'], 'revision': REVISION}
            identical(payload['metadata'], expected_metadata, 'Actual pair cursor/condition/source differs')
            identical(sidecar['metadata'], expected_metadata, 'Pair sidecar metadata differs')
            require(payload['format'] == sidecar['format'] and payload['kind'] == sidecar['kind']
                    and payload['excluded_from_research'] is True and payload['eligible_for_fresh_experiment'] is False
                    and payload['eligible_for_l2_research'] is False and payload['eligible_for_l2'] is False
                    and digest(payload) == sidecar['payload_sha256'] and set(payload['learners']) == set(LANGUAGES),
                    'Actual pair payload integrity/scope differs')
            pair_record = {'condition': condition, 'exchange': exchange, 'sha256': sha(path), 'learners': {}}
            for language, learner in learners.items():
                state = payload['learners'][language]
                require(state['format'] == 'h1-development-acquisition-v1'
                        and all(state.get(k) == v and type(state.get(k)) is type(v) for k, v in SCOPE.items()), 'Actual learner scope differs')
                identical(state['counters'], expected_counts(stream, permutation, condition, language, exchange), 'Snapshot counters differ')
                require(digest(state) == sidecar['learner_snapshot_sha256'][language], 'Full learner snapshot hash differs')
                logical = {k: v for k, v in state.items() if k != 'feature_caches'}
                require(digest(logical) == sidecar['learner_state_sha256'][language], 'Logical learner hash differs')
                expected_rng = random.Random(plan['seeds']['learner'])
                for _ in range(0 if condition == 'no_update' else exchange):
                    positions = list(range(4)); expected_rng.shuffle(positions)
                identical(state['rng'], expected_rng.getstate(), 'Actual private permutation RNG differs')
                generator = torch.Generator(); generator.manual_seed(plan['seeds']['learner'])
                require(torch.equal(state['torch_rng'], generator.get_state()), 'Actual deterministic private Torch RNG differs')
                weights = state['weights']
                reference_weights = initial_weights[language]
                require(set(weights) == set(reference_weights), 'Actual state tensor names differ')
                for name, value in weights.items():
                    require(isinstance(value, torch.Tensor) and value.shape == reference_weights[name].shape
                            and value.dtype == reference_weights[name].dtype and bool(torch.isfinite(value).all()), 'Malformed learner tensor: ' + name)
                    if not name.startswith('understanding.') or condition == 'no_update':
                        require(torch.equal(value, reference_weights[name]), 'Frozen or no-update tensor changed: ' + name)
                identical(state['optimizer']['param_groups'], learner._optimizer_groups, 'Actual L2 Adam groups differ')
                parameters = list(learner.model.understanding.named_parameters())
                ids = [i for group in learner._optimizer_groups for i in group['params']]
                states = state['optimizer']['state']
                require(all(type(i) is int for i in states) and set(states) == (set(ids) if condition != 'no_update' else set()),
                        'Actual L2 Adam state coverage differs')
                for index, (name, parameter) in zip(ids, parameters):
                    if index not in states:
                        continue
                    entry = states[index]
                    require(set(entry) == {'step', 'exp_avg', 'exp_avg_sq'}
                            and isinstance(entry['step'], torch.Tensor) and entry['step'].ndim == 0
                            and float(entry['step']) == exchange, 'Actual L2 Adam step differs')
                    for field in ('exp_avg', 'exp_avg_sq'):
                        v = entry[field]
                        require(isinstance(v, torch.Tensor) and v.is_floating_point() and v.shape == parameter.shape
                                and v.dtype == parameter.dtype and bool(torch.isfinite(v).all()), 'Malformed L2 Adam moment')
                require(set(state['feature_caches']) == set(canonical_cache[language]), 'Actual cache names differ')
                for name, rows in state['feature_caches'].items():
                    require(set(rows) == set(canonical_cache[language][name]), 'Actual canonical cache image keys differ')
                    for key, value in rows.items():
                        require(torch.equal(value, canonical_cache[language][name][key]), 'Actual frozen cache values changed')
                learner.load_state_dict(state)
                require(learner.state_digest() == sidecar['learner_state_sha256'][language], 'Canonical restored learner differs')
                if exchange == 32:
                    require(learner.state_digest() == summary['conditions'][condition]['final_state_sha256'][language], 'Actual final state hash differs')
                    identical(learner.branch_digests(), summary['conditions'][condition]['final_branch_sha256'][language], 'Actual final branch hashes differ')
                    final_queries = [q for q in expected_rows if q['exchange'] == 32 and q['learner'] == language]
                    before, cache_before = learner.state_digest(), learner.cache_digest()
                    for off in range(0, len(final_queries), 32):
                        batch = final_queries[off:off + 32]
                        guesses = learner.predict_many([q['text'] for q in batch], [[pngs[k] for k in q['choices']] for q in batch])
                        for q, guess in zip(batch, guesses):
                            require(guess == raw_by_condition[condition][canon(q)]['prediction'], 'Actual final restored inference differs')
                        restored_predictions += len(batch)
                    for scene, _, _ in evaluation[:3]:
                        identical(learner.emit(pngs[scene.key]), frozen_dev[language][scene.key], 'Actual frozen producer output differs')
                        frozen_outputs_checked += 1
                    require(learner.state_digest() == before and learner.cache_digest() == cache_before, 'Final inference mutated restored state')
                pair_record['learners'][language] = {'updates': state['counters']['updates'],
                    'optimizer_states': len(states), 'moment_tensors_verified': 2 * len(states),
                    'canonical_cache_entries': sum(len(v) for v in state['feature_caches'].values()),
                    'logical_state_sha256': learner.state_digest()}
            checkpoint_records.append(pair_record)
            del payload
            gc.collect()
    require(len(checkpoint_records) == 8, 'Wrong saved pair checkpoint count')
    return {'status': 'pass', 'audit_complete': True, 'preparation_revision': REVISION,
            'scope': 'Post-outcome independent development-only audit; not a preregistered performance gate or research qualification.',
            'excluded_from_research': True, 'eligible_for_l2_research': False,
            'NN_training_or_optimizer_updates_performed_by_auditor': False,
            'protected_images_rendered_or_scored': False,
            'all42_core_source_files_unchanged': True,
            'plan_sha256': summary['plan_sha256'], 'stream_sha256': summary['stream_sha256'],
            'source_sha256': summary['stage_source_sha256'],
            'prediction_rows_regraded': 32256, 'aggregate_points_regraded': 96,
            'actual_pair_checkpoints_verified': 8, 'actual_learner_snapshots_verified': 16,
            'baseline_and_final_queries_restored': restored_predictions,
            'frozen_native_generations_restored': frozen_outputs_checked,
            'condition_results': condition_records, 'checkpoint_results': checkpoint_records,
            'errors': [], 'limitations': summary['limitations']}


if __name__ == '__main__':
    started = time.monotonic()
    initial_paths = [RUN / 'summary.json', RUN / 'semantic-plan.json', RUN / 'frozen-peer-stream.json',
                     RUN / 'frozen-peer-evaluation.json', RUN / 'native-generation-baseline.json',
                     RUN / 'shuffled-association-collisions.json', RUN / 'learning-curves.json']
    initial_paths += [RUN / f'{c}-{suffix}.json' for c in CONDITIONS for suffix in ('summary', 'predictions', 'events')]
    initial_paths += [RUN / f'checkpoints/{c}-e{e:06d}.{ext}' for c in CONDITIONS for e in (16, 32) for ext in ('pt', 'json')]
    initial_paths += [p for p in (RUN / 'inputs').rglob('*') if p.is_file()]
    before = {str(p.relative_to(RUN)): sha(p) for p in initial_paths}
    result = None
    try:
        result = audit()
    except Exception as exc:
        result = {'status': 'fail', 'audit_complete': False, 'errors': [f'{type(exc).__name__}: {exc}']}
        raise
    finally:
        after = {str(p.relative_to(RUN)): sha(p) for p in initial_paths}
        if before != after:
            result = {'status': 'fail', 'audit_complete': False, 'errors': ['Pilot input files changed during audit']}
        result['pilot_files_sha256_before'] = before
        result['pilot_files_sha256_after'] = after
        result['pilot_files_unchanged'] = before == after
        result['checks_passed'] = checks
        result['audit_script_sha256'] = sha(__file__)
        result['executed_at_utc'] = datetime.now(timezone.utc).isoformat()
        result['elapsed_seconds'] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('status', 'audit_complete', 'errors', 'checks_passed', 'elapsed_seconds')}, indent=2))
        print('report_sha256', sha(OUTPUT))
