/* Exact metadata only; no requests, timers, model work or control actions. */
(() => {
  const plan = '1d32fa6924303355eebc2acfeedd81d5e8e79ebe10e73d61777fe27bef3f0669';
  const get = id => document.getElementById('continuity-transfer-' + id);
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const gib = x => finite(x) && x >= 0 ? (x / 1024 ** 3).toFixed(1) + ' GiB' : '—';
  window.renderContinuityTransferFit = data => {
    if (!get('current')) return;
    const v = data?.mac?.continuity_transfer_fit;
    const bound = v?.schema === 'research4.continuity-transfer-dashboard-fit.v1' && v.plan_sha256 === plan;
    const available = bound && v.status === 'available' && Number.isInteger(v.optimizer_updates) && v.optimizer_updates >= 0 && v.optimizer_updates <= 264;
    const count = available ? v.optimizer_updates : null;
    const complete = available && v.training_status === 'complete' && count === 264 && v.dev_completed === 36 && v.completion_saved_metadata_verified === true;
    const live = available && v.live_owner_verified === true && !v.stale && data?.mac?.online !== false;
    get('status').textContent = !available ? 'PREPARING · awaiting verified run identity' : complete ? 'Training and NLL complete' : v.training_status === 'stopped' ? 'Training stopped' : v.training_status === 'paused' ? 'Training paused' : v.training_status === 'terminal_cleanup_unverified' ? 'Release unverified' : v.stale ? 'Saved progress · stale' : live ? count ? 'Training · verified updates' : 'Loading · no updates yet' : 'Saved progress · owner unconfirmed';
    get('count').textContent = (count === null ? '—' : count.toLocaleString()) + ' / 264 verified updates';
    get('state').textContent = count === null ? 'Preparation is separate from optimizer progress.' : 'Contiguous finite saved Adam updates for this exact run.';
    get('fill').style.width = count === null ? '0%' : (100 * count / 264) + '%';
    if (count === null) get('track').removeAttribute('aria-valuenow');
    else get('track').setAttribute('aria-valuenow', String(count));
    get('loss').textContent = available && finite(v.last_training_loss) ? v.last_training_loss.toFixed(4) : '—';
    get('speed').textContent = available && finite(v.last_training_target_tokens_per_compute_second) ? v.last_training_target_tokens_per_compute_second.toFixed(2) : '—';
    get('peak').textContent = available ? gib(v.recorded_mlx_peak_bytes) : '—';
    get('nll').textContent = complete && finite(v.dev_pooled_target_NLL) ? v.dev_pooled_target_NLL.toFixed(5) : 'Pending';
    get('outcomes').textContent = 'Pending · 12 matched episodes per route (24 BASE/FT statuses)';
    get('preparation').textContent = bound && v.preparation_verified === true ? 'Exact corpus, six-pass plan and five source hashes verified.' : 'Awaiting the exact plan/source metadata; launch and progress are not inferred.';
    get('stamp').textContent = available ? 'Saved: ' + (v.recorded_utc || 'unavailable') + ' · ' + v.run_root : 'Run descriptor is unbound. No progress is borrowed from the previous 500-update fit.';
  };
})();
