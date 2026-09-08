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
  return value !== null &&
    value !== undefined &&
    Number.isFinite(Number(value))
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
    ["Web Traceability", data.traceability?.status]
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

  if (!trace || trace.status === "unavailable") {
    target.textContent = "Web traceability is currently unavailable.";
    return;
  }

  metrics("#traceability-findings", [
    ["Status", trace.status],
    ["Module", trace.module]
  ]);

  const groups = [
    [
      "Web entities",
      trace.web_entities?.map(item => item.description || item.entity_id)
    ],
    [
      "Matching pages",
      trace.pages_with_matching_images?.map(item => item.title || item.url)
    ],
    [
      "Full matching images",
      trace.full_matching_image_urls
    ],
    [
      "Partial matching images",
      trace.partial_matching_image_urls
    ],
    [
      "Visually similar images",
      trace.visually_similar_image_urls
    ]
  ];

  groups.forEach(([title, values]) => {
    if (!Array.isArray(values) || !values.length) return;

    const block = document.createElement("div");
    block.className = "trace-block";

    const heading = document.createElement("h3");
    heading.textContent = title;

    const listElement = document.createElement("ul");
    listElement.className = "trace-list";

    values.forEach(item => {
      const li = document.createElement("li");

      const value = typeof item === "string"
        ? item
        : item?.url || item?.description || item?.entity_id;

      if (typeof value === "string" && /^https?:\/\//i.test(value)) {
        const link = document.createElement("a");
        link.href = value;
        link.textContent = value;
        link.target = "_blank";
        link.rel = "noreferrer";
        li.append(link);
      } else {
        li.textContent = text(value);
      }

      listElement.append(li);
    });

    block.append(heading, listElement);
    target.append(block);
  });
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
    [
      "Error Level Analysis",
      evidence(manipulation, "error_level_analysis")?.score
    ],
    [
      "Noise Residual",
      evidence(manipulation, "noise_residual")?.score
    ],
    [
      "Edge Texture",
      evidence(manipulation, "edge_texture")?.score
    ],
    [
      "JPEG source",
      evidence(manipulation, "jpeg_source")?.value
    ]
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
        ? `${properties.width} × ${properties.height}`
        : absent
    ],
    ["Color mode", properties.mode],
    [
      "File size",
      properties.file_size !== null &&
      properties.file_size !== undefined &&
      Number.isFinite(Number(properties.file_size))
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

input.addEventListener("change", () => {
  setFile(input.files[0]);
});

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

    const response = await fetch("/analyze/image", {
      method: "POST",
      body: form
    });

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