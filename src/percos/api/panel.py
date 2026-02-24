"""Memory Control Panel – single-page UI served by FastAPI (§12, §14.2).

Provides a user-facing interface for:
- Viewing current beliefs
- Editing / deleting beliefs
- Inspecting provenance ("Why do you think this?")
- Viewing belief history
- Approving / rejecting pending candidates
- Viewing the audit log
- Exporting memory
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

panel_router = APIRouter()

PANEL_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PCOS – Memory Control Panel</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.5; }
  .container { max-width: 1200px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 1.6rem; margin-bottom: 4px; }
  .subtitle { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 16px; }

  /* Tabs */
  .tabs { display: flex; gap: 2px; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
  .tab { padding: 8px 16px; cursor: pointer; color: var(--text-muted); border: none;
         background: none; font-size: 0.9rem; border-bottom: 2px solid transparent; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* Cards / Tables */
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }
  th { color: var(--text-muted); font-weight: 600; background: var(--surface); position: sticky; top: 0; }
  tr:hover { background: var(--surface); }

  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
  .badge-high { background: rgba(63,185,80,.15); color: var(--green); }
  .badge-medium { background: rgba(210,153,34,.15); color: var(--yellow); }
  .badge-low { background: rgba(248,81,73,.15); color: var(--red); }
  .badge-active { background: rgba(88,166,255,.15); color: var(--accent); }

  /* Buttons */
  button, .btn { padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
                 background: var(--surface); color: var(--text); cursor: pointer; font-size: 0.8rem; }
  button:hover { border-color: var(--accent); }
  .btn-danger { border-color: var(--red); color: var(--red); }
  .btn-danger:hover { background: rgba(248,81,73,.15); }
  .btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  .btn-primary:hover { opacity: 0.9; }

  /* Modal */
  .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.6);
                   z-index: 100; align-items: center; justify-content: center; }
  .modal-overlay.open { display: flex; }
  .modal { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
           padding: 20px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; }
  .modal h3 { margin-bottom: 12px; }
  .modal pre { background: var(--bg); padding: 12px; border-radius: 4px; overflow-x: auto;
               font-size: 0.8rem; margin: 8px 0; }
  .modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 12px; }

  /* Form fields */
  input, select, textarea { background: var(--bg); border: 1px solid var(--border); color: var(--text);
                            padding: 6px 10px; border-radius: 4px; font-size: 0.85rem; }
  input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); }
  textarea { width: 100%; min-height: 80px; resize: vertical; }
  label { display: block; font-size: 0.8rem; color: var(--text-muted); margin: 8px 0 4px; }

  .toolbar { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }
  .toolbar input { flex: 1; }
  .status-bar { font-size: 0.8rem; color: var(--text-muted); margin-top: 8px; }
  .empty { text-align: center; padding: 40px; color: var(--text-muted); }
  .scroll-table { max-height: 65vh; overflow-y: auto; }
</style>
</head>
<body>
<div class="container">
  <h1>&#x1f9e0; Memory Control Panel</h1>
  <p class="subtitle">Ontology-Governed Knowledge Base — Inspect, edit, and control your knowledge</p>

  <div class="tabs">
    <button class="tab active" data-tab="beliefs">Beliefs</button>
    <button class="tab" data-tab="entities">Entities</button>
    <button class="tab" data-tab="pending">Pending Review</button>
    <button class="tab" data-tab="quarantine">Quarantine</button>
    <button class="tab" data-tab="skills">Skills</button>
    <button class="tab" data-tab="policies">Policies</button>
    <button class="tab" data-tab="chat">Chat</button>
    <button class="tab" data-tab="audit">Audit Log</button>
    <button class="tab" data-tab="export">Export / Reset</button>
  </div>

  <!-- ── Entities Tab (Domain-Agnostic) ── -->
  <div id="tab-entities" class="tab-content">
    <div class="toolbar">
      <select id="entity-type-select" onchange="loadEntities()">
        <option value="">Select entity type…</option>
      </select>
      <button onclick="loadEntities()">Refresh</button>
      <button onclick="showCreateEntityForm()">+ New Entity</button>
    </div>
    <div id="create-entity-form" style="display:none; margin-bottom: 12px;">
      <div id="entity-fields-container"></div>
      <button onclick="createEntity()">Create</button>
      <button onclick="document.getElementById('create-entity-form').style.display='none'">Cancel</button>
    </div>
    <div class="scroll-table">
      <table>
        <thead><tr>
          <th>ID</th><th>Type</th><th>Data</th><th>Confidence</th><th>Scope</th><th>Created</th><th>Actions</th>
        </tr></thead>
        <tbody id="entities-body"></tbody>
      </table>
    </div>
    <div class="status-bar" id="entities-status"></div>
  </div>

  <!-- ── Beliefs Tab ── -->
  <div id="tab-beliefs" class="tab-content active">
    <div class="toolbar">
      <input id="belief-filter" type="text" placeholder="Filter by entity type…">
      <button onclick="loadBeliefs()">Refresh</button>
    </div>
    <div class="scroll-table">
      <table>
        <thead><tr>
          <th>Entity Type</th><th>Data</th><th>Confidence</th><th>Scope</th><th>Sensitivity</th><th>Source</th><th>Created</th><th>Actions</th>
        </tr></thead>
        <tbody id="beliefs-body"></tbody>
      </table>
    </div>
    <div class="status-bar" id="beliefs-status"></div>
  </div>

  <!-- ── Pending Tab ── -->
  <div id="tab-pending" class="tab-content">
    <div class="toolbar"><button onclick="loadPending()">Refresh</button></div>
    <div class="scroll-table">
      <table>
        <thead><tr>
          <th>Entity Type</th><th>Data</th><th>Confidence</th><th>Routing</th><th>Created</th><th>Actions</th>
        </tr></thead>
        <tbody id="pending-body"></tbody>
      </table>
    </div>
    <div class="status-bar" id="pending-status"></div>
  </div>

  <!-- ── Audit Tab ── -->
  <div id="tab-audit" class="tab-content">
    <div class="toolbar">
      <select id="audit-action-filter">
        <option value="">All actions</option>
        <option value="event_ingested">event_ingested</option>
        <option value="fact_committed">fact_committed</option>
        <option value="candidate_validated">candidate_validated</option>
        <option value="belief_updated">belief_updated</option>
        <option value="belief_deleted">belief_deleted</option>
        <option value="maintenance_run">maintenance_run</option>
        <option value="memory_reset">memory_reset</option>
      </select>
      <button onclick="loadAudit()">Refresh</button>
    </div>
    <div class="scroll-table">
      <table>
        <thead><tr>
          <th>Time</th><th>Action</th><th>Component</th><th>Actor</th><th>Resource</th><th>Outcome</th><th>Details</th>
        </tr></thead>
        <tbody id="audit-body"></tbody>
      </table>
    </div>
    <div class="status-bar" id="audit-status"></div>
  </div>

  <!-- ── Quarantine Tab (GAP-H2) ── -->
  <div id="tab-quarantine" class="tab-content">
    <div class="toolbar"><button onclick="loadQuarantine()">Refresh</button></div>
    <div class="scroll-table">
      <table>
        <thead><tr>
          <th>Entity Type</th><th>Data</th><th>Confidence</th><th>Routing</th><th>Created</th><th>Actions</th>
        </tr></thead>
        <tbody id="quarantine-body"></tbody>
      </table>
    </div>
    <div class="status-bar" id="quarantine-status"></div>
  </div>

  <!-- ── Skills Tab (GAP-H4) ── -->
  <div id="tab-skills" class="tab-content">
    <div class="toolbar"><button onclick="loadSkills()">Refresh</button></div>
    <div class="scroll-table">
      <table>
        <thead><tr>
          <th>Name</th><th>Description</th><th>Trigger</th><th>Version</th><th>Success Rate</th><th>Created</th><th>Actions</th>
        </tr></thead>
        <tbody id="skills-body"></tbody>
      </table>
    </div>
    <div class="status-bar" id="skills-status"></div>
  </div>

  <!-- ── Policies Tab (GAP-H4) ── -->
  <div id="tab-policies" class="tab-content">
    <div class="toolbar">
      <button onclick="loadPolicies()">Refresh</button>
      <button class="btn-primary" onclick="showCreatePolicy()">+ New Policy</button>
    </div>
    <div class="scroll-table">
      <table>
        <thead><tr>
          <th>Name</th><th>Rule</th><th>Effect</th><th>Priority</th><th>Scope</th><th>Active</th><th>Actions</th>
        </tr></thead>
        <tbody id="policies-body"></tbody>
      </table>
    </div>
    <div class="status-bar" id="policies-status"></div>
  </div>

  <!-- ── Chat Tab (GAP-M7) ── -->
  <div id="tab-chat" class="tab-content">
    <div id="chat-messages" style="max-height:55vh;overflow-y:auto;margin-bottom:12px;padding:8px;background:var(--surface);border-radius:6px;min-height:200px;"></div>
    <div style="display:flex;gap:8px;">
      <input id="chat-input" type="text" style="flex:1;" placeholder="Type a message…" onkeypress="if(event.key==='Enter')sendChat()">
      <button class="btn-primary" onclick="sendChat()">Send</button>
    </div>
  </div>

  <!-- ── Export Tab ── -->
  <div id="tab-export" class="tab-content">
    <div style="display:flex;gap:12px;margin-bottom:16px;">
      <button class="btn-primary" onclick="exportMemory()">Export All Memory</button>
      <button class="btn-danger" onclick="confirmReset()">Reset All Memory</button>
    </div>
    <pre id="export-output" style="background:var(--bg);padding:12px;border-radius:4px;max-height:60vh;overflow:auto;font-size:0.8rem;display:none;"></pre>
  </div>
</div>

<!-- ── Modal ── -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <h3 id="modal-title">Details</h3>
    <div id="modal-body"></div>
    <div class="modal-actions">
      <button onclick="closeModal()">Close</button>
      <button class="btn-primary" id="modal-save" style="display:none;" onclick="saveModal()">Save</button>
    </div>
  </div>
</div>

<script>
const API = '/api/v1';

// ── Tab switching ──
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-' + t.dataset.tab).classList.add('active');
    // Auto-load data
    const loaders = { beliefs: loadBeliefs, pending: loadPending, quarantine: loadQuarantine, skills: loadSkills, policies: loadPolicies, audit: loadAudit, entities: loadEntityTypes };
    if (loaders[t.dataset.tab]) loaders[t.dataset.tab]();
  });
});

// ── API helper ──
async function api(path, opts = {}) {
  const resp = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  return resp.json();
}

// ── Beliefs ──
async function loadBeliefs() {
  const filter = document.getElementById('belief-filter').value;
  const url = filter ? `/beliefs?entity_type=${encodeURIComponent(filter)}` : '/beliefs';
  const data = await api(url);
  const tbody = document.getElementById('beliefs-body');
  tbody.innerHTML = '';
  (data.beliefs || []).forEach(b => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${esc(b.entity_type)}</td>
      <td><code>${esc(JSON.stringify(b.entity_data || {}).slice(0,80))}</code></td>
      <td><span class="badge badge-${b.confidence}">${b.confidence}</span></td>
      <td>${esc(b.scope || '')}</td>
      <td>${esc(b.sensitivity || 'internal')}</td>
      <td>${esc(b.source || '')}</td>
      <td>${fmtDate(b.created_at)}</td>
      <td>
        <button onclick="explainBelief('${b.fact_id}')">Why?</button>
        <button onclick="historyBelief('${esc(b.entity_data && b.entity_data.name ? b.entity_data.name : b.entity_type)}')">History</button>
        <button onclick="editBelief('${b.fact_id}', ${esc(JSON.stringify(b))})">Edit</button>
        <button class="btn-danger" onclick="deleteBelief('${b.fact_id}')">Del</button>
      </td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('beliefs-status').textContent = `${(data.beliefs||[]).length} belief(s) loaded`;
}

async function explainBelief(fid) {
  const data = await api(`/beliefs/${fid}/explain`);
  openModal('Provenance – ' + fid.slice(0,8),
    `<pre>${esc(JSON.stringify(data.provenance || data, null, 2))}</pre>`);
}

let editingFact = null;
function editBelief(fid, belief) {
  editingFact = fid;
  const b = typeof belief === 'string' ? JSON.parse(belief) : belief;
  openModal('Edit Belief', `
    <label>Confidence</label>
    <select id="edit-conf"><option ${b.confidence==='high'?'selected':''}>high</option>
      <option ${b.confidence==='medium'?'selected':''}>medium</option>
      <option ${b.confidence==='low'?'selected':''}>low</option></select>
    <label>Scope</label>
    <select id="edit-scope">${_buildScopeOptions(b.scope||'global')}</select>
    <label>Sensitivity</label>
    <select id="edit-sensitivity"><option ${(b.sensitivity||'internal')==='public'?'selected':''}>public</option>
      <option ${(b.sensitivity||'internal')==='internal'?'selected':''}>internal</option>
      <option ${(b.sensitivity||'internal')==='private'?'selected':''}>private</option>
      <option ${(b.sensitivity||'internal')==='secret'?'selected':''}>secret</option></select>
    <label>Entity Data (JSON)</label>
    <textarea id="edit-data">${esc(JSON.stringify(b.entity_data||{},null,2))}</textarea>
  `, true);
}

async function saveModal() {
  if (!editingFact) return;
  let entityData;
  try { entityData = JSON.parse(document.getElementById('edit-data').value); }
  catch { alert('Invalid JSON in entity data'); return; }
  await api(`/beliefs/${editingFact}`, {
    method: 'PUT',
    body: JSON.stringify({
      confidence: document.getElementById('edit-conf').value,
      scope: document.getElementById('edit-scope').value,
      sensitivity: document.getElementById('edit-sensitivity').value,
      entity_data: entityData,
    }),
  });
  closeModal();
  editingFact = null;
  loadBeliefs();
}

async function deleteBelief(fid) {
  if (!confirm('Delete this belief?')) return;
  await api(`/beliefs/${fid}`, { method: 'DELETE' });
  loadBeliefs();
}

// ── Pending Candidates ──
async function loadPending() {
  const data = await api('/candidates/pending');
  const tbody = document.getElementById('pending-body');
  tbody.innerHTML = '';
  (data.candidates || []).forEach(c => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${esc(c.entity_type)}</td>
      <td><code>${esc(JSON.stringify(c.entity_data||{}).slice(0,80))}</code></td>
      <td><span class="badge badge-${c.confidence}">${c.confidence}</span></td>
      <td>${esc(c.routing)}</td>
      <td>${fmtDate(c.created_at)}</td>
      <td>
        <button class="btn-primary" onclick="validateCandidate('${c.candidate_id}',true)">Accept</button>
        <button class="btn-danger" onclick="validateCandidate('${c.candidate_id}',false)">Reject</button>
      </td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('pending-status').textContent = `${(data.candidates||[]).length} pending candidate(s)`;
}

async function validateCandidate(cid, accept) {
  await api('/memory/validate', {
    method: 'POST',
    body: JSON.stringify({ candidate_id: cid, accept }),
  });
  loadPending();
}

// ── Audit Log ──
async function loadAudit() {
  const action = document.getElementById('audit-action-filter').value;
  const url = action ? `/audit?action=${action}` : '/audit';
  const data = await api(url);
  const tbody = document.getElementById('audit-body');
  tbody.innerHTML = '';
  (data.entries || []).forEach(e => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${fmtDate(e.timestamp)}</td>
      <td>${esc(e.action)}</td>
      <td>${esc(e.component)}</td>
      <td>${esc(e.actor)}</td>
      <td>${esc(e.resource_type)}${e.resource_id?' #'+e.resource_id.slice(0,8):''}</td>
      <td><span class="badge badge-${e.outcome==='success'?'high':e.outcome==='blocked'?'low':'medium'}">${e.outcome}</span></td>
      <td><button onclick="viewAuditDetail(${JSON.stringify(e.details||{})})">View</button></td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('audit-status').textContent = `${(data.entries||[]).length} entries`;
}
function viewAuditDetail(details) {
  openModal('Audit Details', '<pre>' + esc(JSON.stringify(details, null, 2)) + '</pre>');
}

// ── Export / Reset ──
async function exportMemory() {
  const data = await api('/memory/export');
  const el = document.getElementById('export-output');
  el.style.display = 'block';
  el.textContent = JSON.stringify(data, null, 2);
}

async function confirmReset() {
  if (!confirm('This will DELETE ALL memory. Are you sure?')) return;
  if (!confirm('Really? This cannot be undone.')) return;
  await api('/memory/reset', { method: 'POST', body: JSON.stringify({ confirm: true }) });
  alert('Memory has been reset.');
  loadBeliefs();
}

// ── Belief History (GAP-M3) ──
async function historyBelief(entityName) {
  const data = await api(`/beliefs/history/${encodeURIComponent(entityName)}`);
  const rows = (data.history || []).map(h =>
    `<tr><td>${esc(h.entity_type)}</td><td><code>${esc(JSON.stringify(h.entity_data||{}).slice(0,60))}</code></td>` +
    `<td>${esc(h.belief_status)}</td><td>${esc(h.confidence)}</td>` +
    `<td>${h.valid_from||'—'}</td><td>${h.valid_to||'—'}</td></tr>`
  ).join('');
  openModal('Belief History – ' + entityName,
    `<div class="scroll-table"><table><thead><tr><th>Type</th><th>Data</th><th>Status</th><th>Confidence</th><th>From</th><th>To</th></tr></thead><tbody>${rows||'<tr><td colspan=6 class="empty">No history</td></tr>'}</tbody></table></div>`);
}

// ── Quarantine (GAP-H2) ──
async function loadQuarantine() {
  const data = await api('/candidates/quarantined');
  const tbody = document.getElementById('quarantine-body');
  tbody.innerHTML = '';
  (data.candidates || []).forEach(c => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${esc(c.entity_type)}</td>
      <td><code>${esc(JSON.stringify(c.entity_data||{}).slice(0,80))}</code></td>
      <td><span class="badge badge-${c.confidence}">${c.confidence}</span></td>
      <td>${esc(c.routing)}</td>
      <td>${fmtDate(c.created_at)}</td>
      <td>
        <button class="btn-primary" onclick="validateCandidate('${c.candidate_id}',true)">Accept</button>
        <button class="btn-danger" onclick="validateCandidate('${c.candidate_id}',false)">Reject</button>
      </td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('quarantine-status').textContent = `${(data.candidates||[]).length} quarantined item(s)`;
}

// ── Skills (GAP-H4) ──
async function loadSkills() {
  const data = await api('/procedures');
  const tbody = document.getElementById('skills-body');
  tbody.innerHTML = '';
  (data.procedures || []).forEach(p => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${esc(p.name)}</td>
      <td>${esc((p.description||'').slice(0,60))}</td>
      <td>${esc(p.trigger||'')}</td>
      <td>${p.version||1}</td>
      <td>${(p.success_rate||0).toFixed(2)}</td>
      <td>${fmtDate(p.created_at)}</td>
      <td><button onclick="viewSkillHistory('${p.id}')">History</button></td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('skills-status').textContent = `${(data.procedures||[]).length} skill(s)`;
}
async function viewSkillHistory(pid) {
  const data = await api(`/procedures/${pid}/history`);
  const rows = (data.history || []).map(h =>
    `<tr><td>v${h.version}</td><td>${esc(h.name)}</td><td>${(h.success_rate||0).toFixed(2)}</td><td>${fmtDate(h.created_at)}</td></tr>`
  ).join('');
  openModal('Version History', `<table><thead><tr><th>Version</th><th>Name</th><th>Success Rate</th><th>Created</th></tr></thead><tbody>${rows||'<tr><td colspan=4>No history</td></tr>'}</tbody></table>`);
}

// ── Policies (GAP-H4) ──
async function loadPolicies() {
  const data = await api('/policies');
  const tbody = document.getElementById('policies-body');
  tbody.innerHTML = '';
  (data.policies || []).forEach(p => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${esc(p.name)}</td>
      <td>${esc((p.rule||'').slice(0,60))}</td>
      <td>${esc(p.effect)}</td>
      <td>${p.priority||0}</td>
      <td>${esc(p.scope||'global')}</td>
      <td><span class="badge ${p.active?'badge-high':'badge-low'}">${p.active?'Yes':'No'}</span></td>
      <td>
        <button onclick="togglePolicy('${p.id}', ${!p.active})">${p.active?'Disable':'Enable'}</button>
        <button class="btn-danger" onclick="deletePolicy('${p.id}')">Del</button>
      </td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('policies-status').textContent = `${(data.policies||[]).length} policy/policies`;
}
function showCreatePolicy() {
  openModal('Create Policy', `
    <label>Name</label><input id="pol-name" style="width:100%">
    <label>Rule</label><textarea id="pol-rule"></textarea>
    <label>Effect</label>
    <select id="pol-effect"><option>deny</option><option>allow</option><option>require_approval</option></select>
    <label>Priority</label><input id="pol-priority" type="number" value="0" style="width:100%">
    <label>Scope</label>
    <select id="pol-scope">${_buildScopeOptions('global')}</select>
  `, true);
  document.getElementById('modal-save').onclick = async () => {
    await api('/policies', { method:'POST', body: JSON.stringify({
      name: document.getElementById('pol-name').value,
      rule: document.getElementById('pol-rule').value,
      effect: document.getElementById('pol-effect').value,
      priority: parseInt(document.getElementById('pol-priority').value)||0,
      scope: document.getElementById('pol-scope').value,
    })});
    closeModal(); loadPolicies();
  };
}
async function togglePolicy(pid, active) {
  await api(`/policies/${pid}`, { method:'PUT', body: JSON.stringify({active}) });
  loadPolicies();
}
async function deletePolicy(pid) {
  if (!confirm('Delete this policy?')) return;
  await api(`/policies/${pid}`, { method:'DELETE' });
  loadPolicies();
}

// ── Chat (GAP-M7) ──
async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  const messagesDiv = document.getElementById('chat-messages');
  messagesDiv.innerHTML += `<div style="margin:6px 0;"><strong>You:</strong> ${esc(msg)}</div>`;
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  try {
    const data = await api('/chat', { method:'POST', body: JSON.stringify({message: msg}) });
    messagesDiv.innerHTML += `<div style="margin:6px 0;color:var(--accent);"><strong>PCOS:</strong> ${esc(data.response||'(no response)')}</div>`;
  } catch(e) {
    messagesDiv.innerHTML += `<div style="margin:6px 0;color:var(--red);">Error: ${esc(e.message)}</div>`;
  }
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// ── Modal ──
function openModal(title, html, showSave = false) {
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-body').innerHTML = html;
  document.getElementById('modal-save').style.display = showSave ? '' : 'none';
  document.getElementById('modal').classList.add('open');
}
function closeModal() { document.getElementById('modal').classList.remove('open'); }
document.getElementById('modal').addEventListener('click', e => {
  if (e.target === e.currentTarget) closeModal();
});

// ── Entities (Domain-Agnostic) ──
let _schemaData = null;
async function loadEntityTypes() {
  const data = await api('/schema');
  _schemaData = data;
  const sel = document.getElementById('entity-type-select');
  sel.innerHTML = '<option value="">Select entity type…</option>';
  const types = data.entity_types || {};
  Object.keys(types).forEach(t => {
    const opt = document.createElement('option');
    opt.value = t;
    opt.textContent = t;
    sel.appendChild(opt);
  });
}

function _getSchemaScopes() {
  if (_schemaData && _schemaData.scopes && _schemaData.scopes.length) return _schemaData.scopes;
  return ['global'];
}

function _buildScopeOptions(selected) {
  return _getSchemaScopes().map(s =>
    `<option ${s===selected?'selected':''}>${esc(s)}</option>`
  ).join('');
}

async function loadEntities() {
  const etype = document.getElementById('entity-type-select').value;
  if (!etype) return;
  const data = await api(`/entities/${encodeURIComponent(etype)}`);
  const tbody = document.getElementById('entities-body');
  tbody.innerHTML = '';
  (data.entities || []).forEach(e => {
    const tr = document.createElement('tr');
    const id = e.entity_id || '';
    const displayData = {...e};
    delete displayData.entity_id;
    delete displayData.entity_type;
    delete displayData.created_at;
    delete displayData.confidence;
    delete displayData.scope;
    tr.innerHTML = `
      <td><code>${esc(id.slice(0,8))}</code></td>
      <td>${esc(e.entity_type || etype)}</td>
      <td><code>${esc(JSON.stringify(displayData).slice(0,120))}</code></td>
      <td><span class="badge badge-${e.confidence || 'medium'}">${esc(e.confidence || '')}</span></td>
      <td>${esc(e.scope || '')}</td>
      <td>${fmtDate(e.created_at)}</td>
      <td>
        <button class="btn-danger" onclick="deleteEntity('${etype}', '${id}')">Del</button>
      </td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('entities-status').textContent = `${(data.entities || []).length} entities`;
}

function showCreateEntityForm() {
  const etype = document.getElementById('entity-type-select').value;
  if (!etype) { alert('Select an entity type first'); return; }
  const container = document.getElementById('entity-fields-container');
  container.innerHTML = '';
  const types = (_schemaData && _schemaData.entity_types) || {};
  const fields = types[etype] || types[etype.charAt(0).toUpperCase() + etype.slice(1)] || [];
  if (Array.isArray(fields) && fields.length > 0) {
    fields.forEach(f => {
      const fName = f.name || f;
      const fType = (f.type || 'str').toLowerCase();
      const fReq = f.required ? ' *' : '';
      let input;
      if (fType === 'bool' || fType === 'boolean') {
        input = `<select id="ef-${fName}"><option value="true">Yes</option><option value="false" selected>No</option></select>`;
      } else if (fType === 'int' || fType === 'integer' || fType === 'float') {
        input = `<input id="ef-${fName}" type="number" style="width:100%">`;
      } else if (fType === 'datetime') {
        input = `<input id="ef-${fName}" type="datetime-local" style="width:100%">`;
      } else if (fType === 'list[str]' || fType === 'list' || fType.startsWith('list')) {
        input = `<input id="ef-${fName}" type="text" style="width:100%" placeholder="comma-separated values">`;
      } else {
        input = `<input id="ef-${fName}" type="text" style="width:100%">`;
      }
      container.innerHTML += `<label>${esc(fName)}${fReq} <small style="color:var(--text-muted)">(${esc(fType)})</small></label>${input}`;
    });
  } else {
    container.innerHTML = '<label>Entity Data (JSON):</label><textarea id="ef-raw-json" placeholder=\'{"name": "Example"}\'></textarea>';
  }
  document.getElementById('create-entity-form').style.display = 'block';
}

async function createEntity() {
  const etype = document.getElementById('entity-type-select').value;
  if (!etype) { alert('Select an entity type first'); return; }
  let data;
  const rawEl = document.getElementById('ef-raw-json');
  if (rawEl) {
    try { data = JSON.parse(rawEl.value); }
    catch(e) { alert('Invalid JSON'); return; }
  } else {
    data = {};
    const types = (_schemaData && _schemaData.entity_types) || {};
    const fields = types[etype] || types[etype.charAt(0).toUpperCase() + etype.slice(1)] || [];
    fields.forEach(f => {
      const fName = f.name || f;
      const fType = (f.type || 'str').toLowerCase();
      const el = document.getElementById('ef-' + fName);
      if (!el) return;
      const val = el.value.trim();
      if (!val) return;
      if (fType === 'bool' || fType === 'boolean') data[fName] = val === 'true';
      else if (fType === 'int' || fType === 'integer') data[fName] = parseInt(val) || 0;
      else if (fType === 'float') data[fName] = parseFloat(val) || 0;
      else if (fType === 'list[str]' || fType === 'list' || fType.startsWith('list'))
        data[fName] = val.split(',').map(s => s.trim()).filter(Boolean);
      else data[fName] = val;
    });
  }
  await api(`/entities/${encodeURIComponent(etype)}`, {
    method: 'POST', body: JSON.stringify({entity_type: etype, data: data})
  });
  document.getElementById('create-entity-form').style.display = 'none';
  loadEntities();
}

async function deleteEntity(etype, id) {
  if (!confirm('Delete this entity?')) return;
  await api(`/entities/${encodeURIComponent(etype)}/${id}`, { method: 'DELETE' });
  loadEntities();
}

// ── Helpers ──
function esc(s) { const d = document.createElement('div'); d.textContent = String(s || ''); return d.innerHTML; }
function fmtDate(s) { if (!s) return '—'; const d = new Date(s); return d.toLocaleString(); }

// Initial load
loadBeliefs();
</script>
</body>
</html>
"""


@panel_router.get("/panel", response_class=HTMLResponse)
async def memory_control_panel():
    """Serve the Memory Control Panel SPA."""
    return HTMLResponse(content=PANEL_HTML)
