/* Saved audited612 artifact display only; no requests, timers or model calls. */
(() => {
  const get = id => document.getElementById('latest612-' + id);
  const recordElement = () => document.getElementById('replica612-audited-record');
  window.renderLatestReplicaIdentity612 = data => {
    if (!get('current')) return;
    let r;
    try { r = JSON.parse(recordElement().textContent); } catch (_) { return; }
    const verified = r?.schema === 'research4.replica-identity612-audited-ui-record.v1' &&
      r.artifact_id === 'sp-20261116-19-38064e07' && r.updates === 612 && r.DEV_rows === 42 &&
      r.ready_sha256 === 'e3c65ecc69b0b19d2c09b81ee4dc9b7efd7afcb51169190f3324265edb4dc9d5' &&
      r.audit_sha256 === '9556c60f6664da8ce31be945c0c625e2ac8b81d12d0533f6b00f6bb7859e213a';
    if (!verified) { get('status').textContent = 'Saved version metadata unavailable'; return; }
    get('name').textContent = r.artifact_id;
    get('status').textContent = '612-update training independently audited';
    get('count').textContent = '612 / 612 verified optimizer updates';
    get('data').textContent = '306 unique TRAIN decisions · 2 complete passes · Last8 layers 56–63 · rank 16';
    get('dev').textContent = '42 / 42 DEV decisions';
    get('nll').textContent = Number(r.DEV_target_NLL).toFixed(5);
    get('started').textContent = r.fit_started_local;
    get('finished').textContent = r.fit_finished_local;
    get('ready').textContent = r.ready_local;
    get('utc').textContent = 'Artifact READY UTC: ' + r.ready_utc + ' · adapter ' + r.adapter_sha256.slice(0, 8);
    get('note').textContent = 'Saved checkpoint and no-update DEV token loss are independently audited. DEV NLL is not behavioral/Game performance. Artifact READY does not establish an active or qualified serving endpoint; live inference status is shown separately.';
    get('catalog-label').textContent = r.artifact_id + ' · latest audited version';
    get('catalog-start').textContent = r.fit_started_local;
    get('catalog-finish').textContent = r.fit_finished_local;
    get('catalog-ready').textContent = r.ready_local + ' · artifact READY (serving separate)';
  };
})();
