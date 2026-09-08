const API = "http://127.0.0.1:8000";

const input = document.querySelector("#image");
const dropZone = document.querySelector("#drop-zone");
const preview = document.querySelector("#preview");
const previewWrap = document.querySelector("#preview-wrap");
const fileName = document.querySelector("#file-name");
const analyze = document.querySelector("#analyze");
const state = document.querySelector("#state");
const results = document.querySelector("#results");

let file;
const absent = "Not available";

function text(value) {
  return value === null || value === undefined || value === "" ? absent : String(value);
}

function json(value) {
  return JSON.stringify(value ?? null, null, 2);
}

function score(value) {
  return value !== null && value !== undefined && Number.isFinite(Number(value))
    ? `${(Number(value) * 100).toFixed(2)}%`
    : absent;
}

function evidence(data, type) {
  return Array.isArray(data?.evidence)
    ? data.evidence.find(item => item?.type === type)
    : null;
}

function setFile(next) {
  if (!next) return;
  file = next;
  analyze.disabled = false;
  preview.src = URL.createObjectURL(file);
  fileName.textContent = file.name;
  previewWrap.hidden = false;
  state.className = "state";
  state.textContent = `${file.name} selected. Ready for analysis.`;
}

function metrics(target, rows) {
  const element = document.querySelector(target);
  element.replaceChildren();

  rows.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "metric";

    const name = document.createElement("span");
    name.textContent = label;

    const detail = document.createElement("strong");
    detail.textContent = text(value);

    card.append(name, detail);
    element.append(card);
  });
}

function list(target, values) {
  target.replaceChildren();

  const items = Array.isArray(values)
    ? values.filter(value => value !== null && value !== undefined && value !== "")
    : [];

  if (!items.length) {
    target.textContent = absent;
    return;
  }

  const ul = document.createElement("ul");
  ul.className = "plain-list";

  items.forEach(value => {
    const li = document.createElement("li");
    li.textContent = text(value);
    ul.append(li);
  });

  target.append(ul);
}

function detail(title, value) {
  const box = document.createElement("details");

  const summary = document.createElement("summary");
  summary.textContent = title;

  const pre = document.createElement("pre");
  pre.textContent = json(value);

  box.append(summary, pre);
  return box;
}

function renderSummary(data) {
  const values = [
    ["Pipeline", data.status],
    ["AI Detection", data.ai_detection?.label || data.ai_detection?.status],
    ["Manipulation", data.manipulation?.label || data.manipulation?.status],
    ["Traceability", data.traceability?.provenance_status || data.traceability?.status]
  ];

  document.querySelector("#summary").replaceChildren(
    ...values.map(([label, value]) => {
      const card = document.createElement("div");

      const name = document.createElement("span");
      name.textContent = label;

      const status = document.createElement("strong");
      status.textContent = text(value);

      card.append(name, status);
      return card;
    })
  );
}

function renderTraceability(trace) {
  const target = document.querySelector("#traceability-findings");
  target.replaceChildren();

  if (!trace) {
    target.textContent = "Traceability unavailable.";
    return;
  }

  metrics("#traceability-findings", [
    ["Provenance", trace.provenance_status],
    ["Filename", trace.filename],
    ["File type", trace.extension],
    ["File size", trace.size_bytes ? `${(trace.size_bytes / 1024).toFixed(1)} KB` : absent],
    ["SHA-256", trace.sha256]
  ]);

  const c2pa = trace.c2pa;
  if (!c2pa) return;

  const manifest = c2pa.manifests?.[c2pa.active_manifest];
  const generator = manifest?.claim_generator_info?.[0];
  const actions =
    manifest?.assertions?.find(a => a.label === "c2pa.actions.v2")?.data?.actions || [];
  const created = actions.find(a => a.action === "c2pa.created");

  const block = document.createElement("div");
  block.className = "trace-block";

  const heading = document.createElement("h3");
  heading.textContent = "C2PA Provenance";

  const body = document.createElement("div");
  body.className = "trace-list";

  [
    ["Claim generator", generator?.name],
    [
      "Software",
      created?.softwareAgent
        ? `${created.softwareAgent.name} ${created.softwareAgent.version || ""}`.trim()
        : null
    ],
    ["Digital source", created?.digitalSourceType],
    ["Created", created?.when],
    ["Validation", c2pa.validation_state],
    [
      "Validation status",
      c2pa.validation_status?.map(x => x.explanation).join("; ")
    ]
  ].forEach(([label, value]) => {
    const row = document.createElement("p");
    row.innerHTML = `<strong>${label}:</strong> ${text(value)}`;
    body.append(row);
  });

  block.append(heading, body);
  target.append(block);
}

function render(data) {
  const ai = data.ai_detection || {};
  const manipulation = data.manipulation || {};
  const metadata = data.metadata || {};
  const fingerprints = data.fingerprints || {};
  const report = data.evidence_report || {};

  const properties = evidence(metadata, "image_properties")?.values || {};
  const metadataState = evidence(metadata, "metadata") || {};
  const regions = evidence(manipulation, "suspicious_regions");

  const signals = [
    ["Error Level Analysis", evidence(manipulation, "error_level_analysis")?.score],
    ["Noise Residual", evidence(manipulation, "noise_residual")?.score],
    ["Edge Texture", evidence(manipulation, "edge_texture")?.score],
    ["JPEG source", evidence(manipulation, "jpeg_source")?.value]
  ];

  renderSummary(data);

  metrics("#ai-findings", [
    ["AI detector score", score(ai.ai_score)],
    ["Human detector score", score(ai.human_score)],
    ["Assessment", ai.label],
    ["Confidence", ai.confidence],
    ["Model", ai.model],
    ["Device", ai.device]
  ]);

  metrics("#manipulation-findings", [
    ["Signal score", score(manipulation.score)],
    ["Label", manipulation.label],
    ["Confidence", manipulation.confidence]
  ]);

  document.querySelector("#component-signals").replaceChildren(
    ...signals.map(([label, value]) => {
      const card = document.createElement("div");
      card.className = "signal";

      const name = document.createElement("span");
      name.textContent = label;

      const detail = document.createElement("strong");
      detail.textContent = value === undefined ? absent : text(value);

      card.append(name, detail);
      return card;
    })
  );

  metrics("#region-findings", [
    ["Regions flagged", regions?.count],
    ["Review status", regions ? "Available" : absent]
  ]);

  metrics("#metadata-findings", [
    ["Format", properties.format],
    [
      "Dimensions",
      properties.width && properties.height
        ? `${properties.width} Ã— ${properties.height}`
        : absent
    ],
    ["Color mode", properties.mode],
    [
      "File size",
      properties.file_size !== null && properties.file_size !== undefined
        ? `${(Number(properties.file_size) / 1024).toFixed(1)} KB`
        : absent
    ],
    ["Metadata", metadataState.state]
  ]);

  document.querySelector("#metadata-note").textContent =
    metadataState.state === "absent"
      ? "No EXIF metadata was present in the analysis result."
      : "Metadata findings are limited to the available analysis result.";

  document.querySelector("#sha-value").textContent =
    fingerprints.sha256?.artifacts?.[0]?.value ||
    fingerprints.sha256?.value ||
    absent;

  document.querySelector("#phash-value").textContent =
    fingerprints.perceptual_hash?.artifacts?.[0]?.value ||
    fingerprints.perceptual_hash?.value ||
    absent;

  renderTraceability(data.traceability);

  document.querySelector("#overall-assessment").textContent =
    text(report.overall_assessment);

  const support = report.supporting_evidence || {};

  document.querySelector("#evidence-findings").replaceChildren(
    ...[
      ["Investigation Summary", [report.investigation_summary]],
      ["Key Findings", report.key_findings],
      ["Observed Evidence", support.observed],
      ["Interpretation", support.interpretation],
      ["Limitations", support.limitations],
      ["Warnings & Limitations", report.warnings_limitations],
      ["Pipeline Errors", report.pipeline_errors]
    ].map(([title, values]) => {
      const block = document.createElement("div");

      const heading = document.createElement("h3");
      heading.textContent = title;

      const body = document.createElement("div");
      list(body, values);

      block.append(heading, body);
      return block;
    })
  );

  document.querySelector("#details").replaceChildren(
    ...[
      ["AI detector response", ai],
      ["Manipulation detector response", manipulation],
      ["Suspicious-region coordinates", regions],
      ["Metadata response", metadata],
      ["Fingerprint response", fingerprints],
      ["Traceability response", data.traceability],
      ["Evidence report", report],
      [
        "Warnings and pipeline errors",
        {
          warnings: report.warnings_limitations,
          errors: data.errors
        }
      ]
    ].map(([title, value]) => detail(title, value))
  );

  results.hidden = false;
}

input.addEventListener("change", () => setFile(input.files[0]));

["dragenter", "dragover"].forEach(type => {
  dropZone.addEventListener(type, event => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach(type => {
  dropZone.addEventListener(type, event => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});

dropZone.addEventListener("drop", event => {
  setFile(event.dataTransfer.files[0]);
});

document.querySelectorAll("[data-copy]").forEach(button => {
  button.addEventListener("click", async () => {
    const value = document.querySelector(`#${button.dataset.copy}`).textContent;

    try {
      await navigator.clipboard.writeText(value);
      document.querySelector("#copy-state").textContent = "Copied to clipboard.";
    } catch {
      document.querySelector("#copy-state").textContent = "Copying was unavailable.";
    }
  });
});

analyze.addEventListener("click", async () => {
  if (!file) return;

  analyze.disabled = true;
  state.className = "state";
  state.textContent = "Analyzing available forensic signals...";

  try {
    const form = new FormData();
    form.append("image", file);

    const response = await fetch(`${API}/analyze/image`, {
      method: "POST",
      body: form
    });

    const contentType = response.headers.get("content-type") || "";

    if (!contentType.includes("application/json")) {
      throw new Error(
        `Backend returned ${response.status}. Check that FastAPI is running on port 8000.`
      );
    }

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Analysis request failed.");
    }

    render(data);
    state.textContent =
      "Analysis complete. Review results as investigative signals.";
  } catch (error) {
    state.className = "state error";
    state.textContent = error.message;
  } finally {
    analyze.disabled = !file;
  }
});

document.querySelector("#analyze-morph").addEventListener("click", async () => {
  const target = document.querySelector("#morph-target").files[0];
  const referenceA = document.querySelector("#morph-reference-a").files[0];
  const referenceB = document.querySelector("#morph-reference-b").files[0];
  const state = document.querySelector("#morph-state");
  const output = document.querySelector("#morph-results");
  if (!target || !referenceA) { state.className = "state error"; state.textContent = "Target and Reference A are required."; return; }
  state.className = "state"; state.textContent = "Comparing face evidence...";
  try {
    const form = new FormData(); form.append("target", target); form.append("reference_a", referenceA);
    if (referenceB) form.append("reference_b", referenceB);
    const response = await fetch(`${API}/analyze/morph`, { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Morph analysis failed.");
    metrics("#morph-results", [["Decision", data.label], ["Confidence", data.confidence], ["Target / A", score(data.reference_similarities?.target_reference_a)], ["Target / B", score(data.reference_similarities?.target_reference_b)], ["Reference A / B", score(data.reference_similarities?.reference_a_reference_b)], ["Similarity difference", score(data.similarity_difference)], ["Binary forensic signal", data.binary_forensic_signal?.label], ["Warnings", (data.evidence || [data.error]).join(" ")]]);
    output.hidden = false; state.textContent = data.status === "success" ? "Morph evidence analysis complete." : (data.error || "Morph analysis could not be completed.");
  } catch (error) { state.className = "state error"; state.textContent = error.message; }
});

/* ================= MORPH INVESTIGATION ================= */
(() => {
  const style = document.createElement("style");
  style.textContent = `
    .morph-panel{margin-top:28px;padding:22px}
    .morph-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
    .morph-upload{border:1px dashed var(--border);border-radius:8px;padding:18px;text-align:center;background:#0b151f}
    .morph-upload input{width:100%;margin-top:10px}
    .morph-name{margin:8px 0 0;color:#8dd9f5;font-size:.75rem;overflow-wrap:anywhere}
    .morph-actions{margin-top:16px;display:flex;align-items:center;gap:14px}
    .morph-results{margin-top:22px}
    .morph-result-head{display:flex;justify-content:space-between;align-items:center;padding:16px;border:1px solid var(--border);border-radius:8px;background:#0b151f}
    .morph-result-label{font-size:1.25rem;font-weight:700;text-transform:uppercase}
    .morph-result-score{font-size:1.5rem;font-weight:700}
    .morph-note{margin-top:12px;color:var(--muted);font-size:.75rem;line-height:1.5}
    @media(max-width:700px){.morph-grid{grid-template-columns:1fr}.morph-result-head{display:block}}
  `;
  document.head.append(style);

  const uploadPanel = document.querySelector(".upload-panel");
  if (!uploadPanel || document.querySelector("#morph-panel")) return;

  const panel = document.createElement("section");
  panel.id = "morph-panel";
  panel.className = "panel morph-panel";
  panel.innerHTML = `
    <div class="section-heading">
      <div>
        <p class="section-kicker">FACE MORPHING FORENSICS</p>
        <h2>Morph Investigation</h2>
      </div>
      <span class="card-tag">PRIMARY USP</span>
    </div>

    <p class="technical-intro">
      Compare a suspected face against one or two trusted reference images
      using a dedicated morph classifier and SFace differential face analysis.
    </p>

    <div class="morph-grid">
      <label class="morph-upload">
        <strong>TARGET / SUSPECTED IMAGE</strong>
        <span>Image being investigated</span>
        <input id="morph-target" type="file" accept="image/png,image/jpeg,image/webp">
        <p id="morph-target-name" class="morph-name">No file selected</p>
      </label>

      <label class="morph-upload">
        <strong>REFERENCE A</strong>
        <span>Trusted genuine image of the same person</span>
        <input id="morph-ref-a" type="file" accept="image/png,image/jpeg,image/webp">
        <p id="morph-ref-a-name" class="morph-name">No file selected</p>
      </label>

      <label class="morph-upload">
        <strong>REFERENCE B <small>(OPTIONAL)</small></strong>
        <span>Second trusted genuine image</span>
        <input id="morph-ref-b" type="file" accept="image/png,image/jpeg,image/webp">
        <p id="morph-ref-b-name" class="morph-name">No file selected</p>
      </label>
    </div>

    <div class="morph-actions">
      <button id="analyze-morph" type="button" disabled>Analyze Morph</button>
      <p id="morph-state" class="state" aria-live="polite">Select target and Reference A.</p>
    </div>

    <div id="morph-results" class="morph-results" hidden></div>
  `;

  uploadPanel.after(panel);

  const target = document.querySelector("#morph-target");
  const refA = document.querySelector("#morph-ref-a");
  const refB = document.querySelector("#morph-ref-b");
  const button = document.querySelector("#analyze-morph");
  const state = document.querySelector("#morph-state");
  const output = document.querySelector("#morph-results");

  const update = () => {
    document.querySelector("#morph-target-name").textContent = target.files[0]?.name || "No file selected";
    document.querySelector("#morph-ref-a-name").textContent = refA.files[0]?.name || "No file selected";
    document.querySelector("#morph-ref-b-name").textContent = refB.files[0]?.name || "No file selected";
    button.disabled = !(target.files[0] && refA.files[0]);
  };

  [target, refA, refB].forEach(input => input.addEventListener("change", update));

  const pct = value =>
    Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "Not available";

  const value = value =>
    value === null || value === undefined ? "Not available" : String(value);

  button.addEventListener("click", async () => {
    button.disabled = true;
    state.className = "state";
    state.textContent = "Running morph and differential face analysis...";
    output.hidden = true;

    try {
      const form = new FormData();
      form.append("target", target.files[0]);
      form.append("reference_a", refA.files[0]);
      if (refB.files[0]) form.append("reference_b", refB.files[0]);

      const response = await fetch("http://127.0.0.1:8000/analyze/morph", {
        method: "POST",
        body: form
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Morph analysis failed.");

      const model = data.morph_model || {};
      const sim = data.reference_similarities || {};
      const face = data.target_face || {};

      output.innerHTML = `
        <div class="morph-result-head">
          <div>
            <span class="assessment-label">MORPH ASSESSMENT</span>
            <div class="morph-result-label">${value(data.label)}</div>
            <div class="morph-note">Confidence: ${value(data.confidence)}</div>
          </div>
          <div>
            <span class="assessment-label">MORPH SCORE</span>
            <div class="morph-result-score">${pct(model.morph_score)}</div>
          </div>
        </div>

        <div class="metrics" style="margin-top:14px">
          <div class="metric"><span>Bona fide score</span><strong>${pct(model.bona_fide_score)}</strong></div>
          <div class="metric"><span>Target ? Reference A</span><strong>${pct(sim.target_reference_a)}</strong></div>
          <div class="metric"><span>Target ? Reference B</span><strong>${pct(sim.target_reference_b)}</strong></div>
          <div class="metric"><span>Reference A ? B</span><strong>${pct(sim.reference_a_reference_b)}</strong></div>
          <div class="metric"><span>Similarity difference</span><strong>${pct(data.similarity_difference)}</strong></div>
          <div class="metric"><span>Face confidence</span><strong>${pct(face.confidence)}</strong></div>
          <div class="metric"><span>Model</span><strong>${value(model.model)}</strong></div>
          <div class="metric"><span>Device</span><strong>${value(model.device)}</strong></div>
        </div>

        <div class="assessment">
          <span class="assessment-label">FORENSIC EVIDENCE</span>
          <p>${(data.evidence || []).map(item => `• ${value(item)}`).join("<br>") || "Not available"}</p>
        </div>

        <details>
          <summary>Raw morph detector response</summary>
          <pre>${JSON.stringify(data, null, 2)}</pre>
        </details>
      `;

      output.hidden = false;
      state.textContent = "Morph analysis complete. Review the forensic evidence.";
    } catch (error) {
      state.className = "state error";
      state.textContent = error.message;
    } finally {
      button.disabled = !(target.files[0] && refA.files[0]);
    }
  });
})();
