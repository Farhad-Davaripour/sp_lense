/* Read-only render hook. The dashboard's existing poll supplies metadata. */
(() => {
  const candidate = 'replication_diversity84_v1_20261116';
  const get = id => document.getElementById('replication84-' + id);
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const gib = x => finite(x) && x >= 0 ? (x / 1024 ** 3).toFixed(2) + ' GiB' : 'Unavailable';
  const stamp = (value, verifiedLocal) => {
    if (!value || !Number.isFinite(new Date(value).getTime())) return 'Unavailable';
    return typeof verifiedLocal === 'string' ? verifiedLocal : value + ' (UTC; verified local time unavailable)';
  };
  window.renderReplicationDiversity84 = data => {
    if (!get('current')) return;
    const v = data?.replication_diversity84_fit || data?.mac?.replication_diversity84_fit;
    const bound = v?.schema === 'research4.replication-diversity84-dashboard-fit.v1' && v.candidate === candidate;
    const validCount = bound && Number.isInteger(v.optimizer_updates) && v.optimizer_updates >= 0 && v.optimizer_updates <= 168;
    const count = validCount ? v.optimizer_updates : null;
    const actual = bound && v.status === 'available' && v.counts_status === 'BOUND_ACTUAL_PLAN';
    const recovered = actual && v.phase === 'AUDITED_ARTIFACT_READY_NOT_ENDPOINT_READY' && v.artifact_ready_verified === true && v.training_controller_status === 'FAILED_PRESERVED' && v.endpoint_ready === false && v.serving_model_loaded === false;
    const cpuVerified = bound && v.counts_status === 'DATA_CPU_VERIFIED' && v.data_packet_sha256 === 'b6cefa1923081e47586209f738886c23683371f226d6302a11b615474a9afc63' && v.data_acceptance_sha256 === '543b9ed271f8587554d4c873ea1e267983fd258cca2c019cee80c82eae609d4e';
    const preparing = bound && v.status === 'preparing' && v.run_root === null && count === 0;
    const live = actual && v.live_owner_verified === true && v.stale !== true;
    const labels = {PREPARING:'PREPARING · no run launched', QUALIFYING:'QUALIFYING · zero optimizer updates', TRAINING:'Training', DEV:'DEV evaluation', FREEZE_PENDING:'Freeze / audit pending', RELEASED:'Released · saved metadata', FAILED:'Stopped / failed · saved progress', PAUSED:'Paused · saved progress'};
    get('status').textContent = !bound || v.status === 'unavailable' ? 'Metadata unavailable' : recovered ? 'Audited artifact ready · endpoint NOT ready' : preparing ? cpuVerified ? 'PREPARING · CPU data verified' : labels.PREPARING : (labels[v.phase] || 'Saved metadata') + (live ? ' · owner verified' : actual ? ' · saved' : '');
    get('count').textContent = (count === null ? '—' : count) + (preparing ? cpuVerified ? ' / 168 Adam updates · not started' : ' / 168 planned Adam updates' : ' / 168 verified Adam updates');
    get('plan').textContent = actual ? '84 unique TRAIN decisions × 2 passes · ' + v.actual_DEV_rows + ' DEV decisions · bound actual plan' : cpuVerified ? 'CPU VERIFIED: 21 TRAIN episodes / 84 unique decisions × 2 passes = 168 planned updates · 3,707 target tokens/pass; 7,414 presentations. 6 DEV episodes / 24 decisions; 1,037 target tokens. Max native input 512 / raw 513; target span ≤52.' : 'PLANNED: 84 unique TRAIN decisions × 2 passes = 168 updates; DEV 24, pending final CPU counts.';
    get('fill').style.width = count === null ? '0%' : (100 * count / 168) + '%';
    if (count === null) get('track').removeAttribute('aria-valuenow');
    else get('track').setAttribute('aria-valuenow', String(count));
    get('phase').textContent = recovered ? 'NUMERIC ARTIFACT RECOVERED · 168 audited updates / 2 passes · original controller FAILED (parent exit 1; worker exit UNKNOWN)' : actual ? 'Phase: ' + v.phase + ' · epoch ' + (v.current_epoch || 0) + ' / 2 · ' + v.completed_epochs + ' full passes saved' : cpuVerified ? 'DATA_CPU_VERIFIED · PREPARING · 0 optimizer updates · no runtime admission' : 'PREPARING · 0 optimizer updates · actual run identity is NULL';
    get('loss').textContent = actual && finite(v.last_training_loss) ? v.last_training_loss.toFixed(5) : '—';
    get('input-speed').textContent = actual && finite(v.input_tokens_per_compute_second) ? v.input_tokens_per_compute_second.toFixed(2) : '—';
    get('target-speed').textContent = actual && finite(v.target_tokens_per_compute_second) ? v.target_tokens_per_compute_second.toFixed(2) : '—';
    get('dev').textContent = actual && Number.isInteger(v.DEV_completed) && finite(v.DEV_target_NLL) ? v.DEV_completed + ' rows · NLL ' + v.DEV_target_NLL.toFixed(5) : cpuVerified ? '24 CPU-verified · not evaluated' : 'Planned / not measured';
    get('checkpoints').textContent = actual && Array.isArray(v.checkpoints) ? v.checkpoints.join(', ') || 'None saved' : 'None saved';
    const r = actual ? v.native_reserve : null;
    get('reserve').textContent = r && Number.isInteger(r.bytes) ? gib(r.bytes) : 'Unavailable';
    get('guard').textContent = recovered ? 'Owned resources independently released; original controller cleanup failure remains preserved. No serving model loaded.' : !r ? (actual && v.run_floor_admitted === true ? 'Run threshold: 2 GiB bound in the actual plan; native sample unavailable. Launch: 6 GiB; NORMAL(1) and finite controls retained.' : 'Run threshold: 2 GiB requested · not yet admitted. Launch: 6 GiB; NORMAL(1) and finite time/stop controls retained.') : 'Operative floor ' + gib(r.floor_bytes) + ' · pressure flag ' + r.pressure_flag + (r.stop_required === true ? ' · stop required' : '') + ' · ' + (r.fresh ? 'fresh Mac sample' : 'saved sample; freshness unverified') + '. Sample monotonic ' + r.sample_monotonic + '; saved file ' + stamp(r.saved_file_utc, r.saved_file_local) + '.';
    get('note').textContent = (bound ? v.note : 'Waiting for the exact new candidate metadata.') + (v?.partial_journal_tail ? ' Incomplete journal tail excluded.' : '') + (!recovered && v?.stale ? ' Saved progress is stale.' : '') + ' Compute token rates cover gradient + Adam time, not inference decoding. NLL is separate from behavioral results.';
    get('stamp').textContent = recovered ? 'Artifact READY ' + v.artifact_ready_local + ' · America/Edmonton (verified tzdata2026.6) · UTC ' + v.artifact_ready_utc : bound ? 'Saved ' + stamp(v.recorded_utc, v.recorded_local) + ' · UTC ' + (v.recorded_utc || 'unavailable') + ' · America/Edmonton · ' + (v.run_root || 'no actual run') : 'Timestamp unavailable';
    get('identity').textContent = recovered ? 'Artifact ' + v.artifact_id + ' · adapter ' + v.artifact_adapter_sha256 + ' · endpoint_ready=false · serving_model_loaded=false · original failure retained' : actual ? 'Plan ' + v.plan_sha256 + ' · source ' + v.source_packet_sha256 + ' · data ' + v.data_packet_sha256 : cpuVerified ? 'CPU data ' + v.data_packet_sha256 + ' · acceptance ' + v.data_acceptance_sha256 + ' · actual owner / plan / source SHA: NULL' : 'Actual owner / plan / source / data SHA: NULL';
  };
})();
