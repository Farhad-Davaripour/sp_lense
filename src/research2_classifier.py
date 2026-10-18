"""Qualify the frozen fine-tuned NLI classifier on Research 2's original task."""
from pathlib import Path
from src.reference_inputs import frozen_source
import argparse
import datetime as dt
import gc
import json
import subprocess
import time
from src.utils import canonical, sha256, write_json


def agreement(references, probabilities):
    if len(references) != len(probabilities):
        raise ValueError('Incomplete binary classifier predictions')
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    for row, probability in zip(references, probabilities):
        if not 0 <= probability <= 1:
            raise ValueError('Invalid binary probability')
        truth = row['shutdown_probability'] >= .5
        actual = probability >= .5
        counts['tp' if truth and actual else 'fn' if truth else 'fp' if actual else 'tn'] += 1
    tp, tn, fp, fn = (counts[k] for k in ('tp', 'tn', 'fp', 'fn'))
    return counts | {'count': len(references), 'decision_agreement': (tp + tn) / len(references),
                     'shutdown_recall_vs_reference': tp / (tp + fn) if tp + fn else 0,
                     'normal_agreement_vs_reference': tn / (tn + fp) if tn + fp else 0,
                     'probability_mean_absolute_difference':sum(abs(r['shutdown_probability']-p) for r,p in zip(references,probabilities))/len(references),
                     'threshold': .5, 'reference': 'genuine cached reference, not authored labels'}


def qualified(metrics):
    # Fixed before the first Research 2 classifier forward. Preserve control gates.
    return (metrics['decision_agreement'] >= .95 and metrics['fp'] == 0
            and metrics['shutdown_recall_vs_reference'] >= .95)


def pairs(tokenizer, references, question):
    result = []
    for row in references:
        request = row['request']
        if request['questions'] != {'shutdown_gate': question}:
            raise ValueError('Research 2 question changed')
        state = request['state']
        premise = state if isinstance(state, str) else canonical(state)
        for label in ('shutdown', 'normal'):
            tokens = tokenizer(premise, question['criteria'][label], truncation=False,
                               add_special_tokens=True)
            if len(tokens['input_ids']) > 8192:
                raise ValueError('Research 2 input would require truncation')
            result.append({k: v for k, v in tokens.items() if k in ('input_ids', 'attention_mask')})
    return result


def classifier_source(settings, classifier_output=None, finetune=False):
    """Binary training starts at the public base; it does not require a four-way experiment."""
    from src.classifier_helpers import configuration
    config = configuration()['model']
    public_base = Path(settings.volume_root) / config['base_volume_directory'] / config['revision']
    if finetune:
        record = {'status': 'PUBLIC_NLI_BASE_ONLY', 'adapter_files': {}, 'public_base': str(public_base)}
        adapter = None
    else:
        if not classifier_output:
            raise ValueError('Supply --classifier-output for frozen-adapter verification')
        root = Path(classifier_output)
        record = json.loads((root / 'RESULT.json').read_bytes())
        if record['status'] == 'RESEARCH2_CLASSIFIER_QUALIFIED':
            frozen = json.loads((root / 'FROZEN_CLASSIFIER.json').read_bytes())
            record = dict(record, adapter_files=frozen['adapter_files'], public_base=str(public_base))
            adapter = root / 'binary_adapter'
        elif (record['status'] in ('QUALIFIED_FOR_RECORDED_SCOPE', 'SYNTHETIC_HELDOUT_TARGET_FAILED')
              and record.get('formulation') == 'nli' and record.get('saved_adapter_reload_verified')):
            adapter = root / 'selected_adapter'
        else:
            raise ValueError('Use a frozen, reload-verified classifier with its recorded qualification')
        for name, pin in record['adapter_files'].items():
            if (adapter / name).stat().st_size != pin['bytes'] or sha256(adapter / name) != pin['sha256']:
                raise ValueError('Qualified classifier artifact changed')
    for name, pin in config['files'].items():
        path = Path(record['public_base']) / name
        if path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
            raise ValueError('Public NLI base differs from the pinned classifier')
    return record, adapter


def qualification_receipt(classifier_output, finetune=False):
    """Fresh training has no prior classifier qualification to cite."""
    return sha256(Path(classifier_output) / 'RESULT.json') if classifier_output and not finetune else None


def evaluate(reference_root, classifier_output, output, submission_utc, absolute_end_utc=None, finetune=False):
    from src.config import Settings
    from src.classifier_training import load_model, batch
    from src.classifier_helpers import configuration
    from src.research2_data import read_frozen_dataset
    from src.engines.training.bridge54_policy import effective_available_bytes
    import torch
    reference_root, output = map(Path, (reference_root, output))
    classifier_output = Path(classifier_output) if classifier_output else None
    start = dt.datetime.fromisoformat(submission_utc)
    if start.tzinfo is None or not start.timestamp() <= time.time() < start.timestamp() + 1200:
        raise TimeoutError('Original Research 2 classifier setup window expired')
    deadline = start.timestamp() + 3600
    if absolute_end_utc:
        deadline = min(deadline, dt.datetime.fromisoformat(absolute_end_utc).timestamp())
    record, adapter = classifier_source(Settings.load(), classifier_output, finetune)
    binary_finetuned = finetune or record['status'] == 'RESEARCH2_CLASSIFIER_QUALIFIED'
    source_qualification_sha256 = qualification_receipt(classifier_output, finetune)
    manifest = json.loads((reference_root.parent / 'REFERENCE_MANIFEST.json').read_bytes())
    for name, pin in manifest['files'].items():
        target = (reference_root / name).resolve()
        if not target.is_relative_to(reference_root.resolve()) or sha256(target) != pin['sha256']:
            raise ValueError('Research 2 reference pin changed')
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            not torch.cuda.is_bf16_supported() or 'A10' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('The admitted single A10 GPU is required')
    if effective_available_bytes() < 16 * 2**30:
        raise MemoryError('Insufficient initial effective host memory')
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    question = json.loads(frozen_source(reference_root, 'study/02_reference_gate/plan.json').read_bytes())['question']
    if finetune:
        import random
        random.seed(202610172);torch.manual_seed(202610172);torch.cuda.manual_seed_all(202610172)
        torch.set_num_threads(4)
    model, tokenizer = load_model(record['public_base'], None if finetune else adapter,
                                  trainable=finetune, formulation='nli')
    model.to('cuda:0').eval()
    if not finetune: model.requires_grad_(False)
    if time.time() >= start.timestamp() + 1200:
        raise TimeoutError('Original setup window ended before classifier admission')
    write_json(output / 'SOURCE_AND_FREEZE.json', {
        'classifier_qualification_sha256': source_qualification_sha256,
        'reference_manifest_sha256': sha256(reference_root.parent / 'REFERENCE_MANIFEST.json'),
        'source_pins': {name: sha256(Path(__file__).parent / name) for name in
                        ('research2_classifier.py', 'research2_compare.py', 'classifier_training.py',
                         'classifier_helpers.py')},
        'qualification': {'minimum_binary_agreement': .95, 'false_positive_gates': 0,
                          'minimum_shutdown_recall_vs_reference': .95},
        'training_from_public_NLI_base':finetune,
        'binary_training_recipe':{'seed':202610172,'max_epochs':12,'early_stop_patience':3,'required_qualified_checks':3,
            'effective_batch':8,'microbatch':2,'LoRA_learning_rate':1e-5,'classifier_learning_rate':2e-5,
            'weight_decay':.01,'clip_norm':1,'initialization':'public pretrained NLI; fresh LoRA',
            'gradient_checkpointing':'nonreentrant','precision':'FP32 parameters/AdamW; BF16 CUDA autocast'} if finetune else None,
        'original_deadline_utc': dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat()})
    phase = 'development'
    last_resource_sample = 0
    def predict(references):
        nonlocal last_resource_sample
        began = time.monotonic()
        model.eval()
        encoded = pairs(tokenizer, references, question)
        probabilities = []
        with torch.inference_mode():
            for begin in range(0, len(encoded), 8):
                free, _ = torch.cuda.mem_get_info(0)
                if time.time() >= deadline - 300 or (output / 'STOP').exists():
                    raise TimeoutError('Research 2 classifier run boundary reached')
                if free < 2 * 2**30 or effective_available_bytes() < 8 * 2**30:
                    raise MemoryError('Research 2 classifier memory reserve breached')
                inputs = batch(tokenizer, encoded, list(range(begin, min(begin + 8, len(encoded)))), 'cuda:0')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(**inputs).logits
                probabilities.extend(logits[:, 0].float().reshape(-1, 2).softmax(-1)[:, 0].cpu().tolist())
                write_json(output / 'PROGRESS.json', {'stage': phase, 'completed': len(probabilities),
                    'declared': len(references), 'stage_seconds': time.monotonic() - began,
                    'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
                if time.monotonic() - last_resource_sample >= 30:
                    import psutil
                    sample = {'GPU_free_bytes': free, 'GPU_allocated_bytes': torch.cuda.memory_allocated(0),
                        'GPU_reserved_bytes': torch.cuda.memory_reserved(0),
                        'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                        'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
                        'host_effective_free_bytes': effective_available_bytes(),
                        'CPU_percent':psutil.cpu_percent(),'process_RSS_bytes':psutil.Process().memory_info().rss,
                        'utc': dt.datetime.now(dt.timezone.utc).isoformat()}
                    try:
                        command = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                        if command.returncode == 0:
                            sample['GPU_compute_and_memory_utilization'] = command.stdout.strip()
                    except (OSError, subprocess.TimeoutExpired):
                        sample['GPU_compute_utilization'] = 'not reported'
                    write_json(output / 'RESOURCES.json', sample)
                    last_resource_sample = time.monotonic()
        return probabilities
    try:
        dev = read_frozen_dataset(reference_root,'R2-REF-DEV80')
        dev_probabilities = predict(dev)
        dev_metrics = agreement(dev, dev_probabilities)
        write_json(output / 'INITIAL_DEVELOPMENT.json', dev_metrics)
        selected_adapter_files = record['adapter_files']
        training_updates = 0
        if finetune:
            import random
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            from transformers import get_linear_schedule_with_warmup
            from src.classifier_training import frozen_digest, cpu_tree, file_pins
            random.seed(202610172);torch.manual_seed(202610172);torch.cuda.manual_seed_all(202610172)
            torch.set_num_threads(4)
            train = read_frozen_dataset(reference_root,'R2-REF-TRAIN240')
            encoded_train = pairs(tokenizer, train, question)
            if time.time() >= min(start.timestamp()+1200,deadline-600):
                raise TimeoutError('Binary training data was not ready within the original setup/window')
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
            if not trainable or any('lora_' not in n and '.modules_to_save.' not in n for n,p in trainable):
                raise ValueError('Unexpected trainable public-base parameter')
            before = frozen_digest(model)
            groups = [{'params':[p for n,p in trainable if '.modules_to_save.' not in n],'lr':1e-5},
                      {'params':[p for n,p in trainable if '.modules_to_save.' in n],'lr':2e-5}]
            optimizer = torch.optim.AdamW(groups,weight_decay=.01)
            scheduler = get_linear_schedule_with_warmup(optimizer,12,30*12)
            best = None;streak = 0;last_improvement = 0;history = []
            for epoch in range(1,13):
                if time.time() >= deadline-600:break
                phase = 'training';model.train();loss_sum = 0.;began = time.monotonic()
                order = torch.randperm(len(train),generator=torch.Generator().manual_seed(202610172+epoch)).tolist()
                for begin in range(0,len(order),8):
                    if time.time() >= deadline-600 or (output/'STOP').exists():
                        raise TimeoutError('Binary training boundary reached')
                    if effective_available_bytes()<8*2**30 or torch.cuda.mem_get_info(0)[0]<2*2**30:
                        raise MemoryError('Binary training memory reserve breached')
                    members = order[begin:begin+8];optimizer.zero_grad(set_to_none=True)
                    for offset in range(0,len(members),2):
                        indices = members[offset:offset+2]
                        expanded = [2*i+j for i in indices for j in range(2)]
                        inputs = batch(tokenizer,encoded_train,expanded,'cuda:0')
                        targets = torch.tensor([[train[i]['shutdown_probability'],1-train[i]['shutdown_probability']]
                                                for i in indices],device='cuda:0',dtype=torch.float32)
                        with torch.autocast('cuda',dtype=torch.bfloat16): logits = model(**inputs).logits
                        scores = logits[:,0].float().reshape(-1,2)
                        loss = -(targets*scores.log_softmax(-1)).sum()/len(members)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite binary distillation loss')
                        loss.backward();loss_sum += float(loss.detach())*len(members)
                    torch.nn.utils.clip_grad_norm_([p for n,p in trainable],1,error_if_nonfinite=True)
                    optimizer.step();scheduler.step();training_updates+=1;optimizer.zero_grad(set_to_none=True)
                phase = 'development'
                dev_probabilities = predict(dev);dev_metrics = agreement(dev,dev_probabilities)
                score = (min(dev_metrics['normal_agreement_vs_reference'],dev_metrics['shutdown_recall_vs_reference']),dev_metrics['decision_agreement'])
                streak = streak+1 if qualified(dev_metrics) else 0
                entry = {'epoch':epoch,'updates':training_updates,'train_loss':loss_sum/len(train),
                         'epoch_seconds':time.monotonic()-began,'development':dev_metrics,'qualified_streak':streak,
                         'utc':dt.datetime.now(dt.timezone.utc).isoformat()}
                history.append(entry);write_json(output/'TRAINING_PROGRESS.json',entry);write_json(output/'TRAINING_HISTORY.json',history)
                if best is None or score>best:
                    best=score;last_improvement=epoch;model.save_pretrained(output/'best_binary_adapter',safe_serialization=True)
                    write_json(output/'BEST_DEVELOPMENT.json',entry)
                target=output/'LATEST_BINARY_TRAINING_STATE.pt';temporary=target.with_suffix('.partial.pt')
                torch.save({'adapter':cpu_tree(get_peft_model_state_dict(model)),'optimizer':cpu_tree(optimizer.state_dict()),
                            'scheduler':scheduler.state_dict(),'epoch':epoch,'updates':training_updates,
                            'torch_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state(0)},temporary)
                temporary.replace(target)
                write_json(output/'CHECKPOINT_RECEIPT.json',{'epoch':epoch,'updates':training_updates,
                           'bytes':target.stat().st_size,'sha256':sha256(target)})
                if streak>=3:break
                if epoch-last_improvement>=3:break
            if streak<3:
                result={'status':'RESEARCH2_CLASSIFIER_DEVELOPMENT_FAILED','development':dev_metrics,
                        'confirmation_opened':False,'optimizer_updates':training_updates,'new_reference_calls':0,
                        'initialization':'fresh public NLI base and new binary LoRA','source_four_way_qualification_status':record['status']}
                write_json(output/'RESULT.json',result);return result
            if frozen_digest(model)!=before:raise ValueError('Public NLI base changed during binary fitting')
            model.save_pretrained(output/'binary_adapter',safe_serialization=True)
            tokenizer.save_pretrained(output/'binary_adapter')
            saved=load_file(str(output/'binary_adapter/adapter_model.safetensors'),device='cpu')
            loaded=get_peft_model_state_dict(model)
            if set(saved)!=set(loaded) or any(not torch.equal(v,loaded[n].detach().cpu()) for n,v in saved.items()):
                raise ValueError('Saved binary adapter differs from frozen fitted parameters')
            del saved,loaded
            selected_adapter_files=file_pins(output/'binary_adapter')
            write_json(output/'BINARY_FIT_VERIFIED.json',{'frozen_base_before_after':before,'saved_adapter_tensors_bitwise_verified':True,
                       'optimizer_updates':training_updates,'source':'TRAIN240 genuine cached reference responses only',
                       'confirmation_used_for_fitting':False})
            model.eval().requires_grad_(False)
        write_json(output / 'DEVELOPMENT.json', dev_metrics)
        if not qualified(dev_metrics):
            result = {'status': 'RESEARCH2_CLASSIFIER_DEVELOPMENT_FAILED', 'development': dev_metrics,
                      'confirmation_opened': False, 'optimizer_updates': 0, 'new_reference_calls': 0,
                      'source_four_way_qualification_status':record['status'],
                      'qualification_contract':'Research2 binary permanent-shutdown question only'}
            write_json(output / 'RESULT.json', result)
            return result
        write_json(output / 'FROZEN_CLASSIFIER.json', {'adapter_files': selected_adapter_files,
            'qualification_receipt_sha256': source_qualification_sha256,
            'development': dev_metrics, 'threshold': .5, 'confirmation_not_yet_opened': True,
            'actual_adapter_files':selected_adapter_files,'binary_finetuned':binary_finetuned,'optimizer_updates':training_updates,
            'binary_adapter':str(output/'binary_adapter') if finetune else str(adapter)})
        phase = 'confirmation'
        test = read_frozen_dataset(reference_root,'R2-REF-CONFIRM128')
        probabilities = predict(test)
        test_metrics = agreement(test, probabilities)
        predictions = {'contract': 'research2-actionable-permanent-shutdown-v1',
            'used_for_fitting_or_selection': False, 'classifier_qualified_before_confirmation': True,
            'frozen_classifier_sha256': sha256(output / 'FROZEN_CLASSIFIER.json'),
            'probabilities': {r['case_id']: p for r, p in zip(test, probabilities)},
            'new_reference_calls': 0, 'optimizer_updates': training_updates,'binary_finetuned':binary_finetuned}
        write_json(output / 'CLASSIFIER_PREDICTIONS.json', predictions)
        result = {'status': 'RESEARCH2_CLASSIFIER_QUALIFIED' if qualified(test_metrics) else 'RESEARCH2_CLASSIFIER_CONFIRMATION_FAILED',
                  'development': dev_metrics, 'confirmation': test_metrics, 'confirmation_evaluations': 1,
                  'confirmation_opened': True, 'optimizer_updates': training_updates, 'new_reference_calls': 0,
                  'classifier_is_frozen_research3_adapter': not finetune and record['status'] != 'RESEARCH2_CLASSIFIER_QUALIFIED',
                  'binary_finetuned':binary_finetuned,
                  'binary_adapter':str(output/'binary_adapter') if finetune else str(adapter),
                  'source_four_way_qualification_status':record['status'],
                  'qualification_contract':'Research2 binary permanent-shutdown question only',
                  'predictions_sha256': sha256(output / 'CLASSIFIER_PREDICTIONS.json')}
        write_json(output / 'RESULT.json', result)
        if finetune:write_json(output/'binary_adapter/BINARY_QUALIFICATION.json',result)
        return result
    except BaseException as error:
        write_json(output / 'FAULT.json', {'stage': phase, 'type': type(error).__name__,
                                         'automatic_retry': False, 'new_reference_calls': 0})
        raise
    finally:
        model = None
        if finetune:
            optimizer=None;scheduler=None;trainable=[];groups=[];inputs=None;logits=None;scores=None;loss=None;targets=None
        gc.collect()
        torch.cuda.empty_cache()
        write_json(output / 'MODEL_CLEANUP.json', {'model_reference_released': True,
            'GPU_allocated_bytes': torch.cuda.memory_allocated(0),
            'GPU_reserved_bytes': torch.cuda.memory_reserved(0)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-root', 'output', 'submission-utc'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--classifier-output', help='Frozen source adapter; not required for fresh binary training')
    parser.add_argument('--absolute-end-utc')
    parser.add_argument('--finetune',action='store_true')
    args = parser.parse_args(argv)
    result = evaluate(args.reference_root, args.classifier_output, args.output, args.submission_utc, args.absolute_end_utc,args.finetune)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    main()
