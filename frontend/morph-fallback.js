(() => {
  if (document.querySelector("#morph-fallback-panel")) return;

  const panel = document.createElement("section");
  panel.id = "morph-fallback-panel";
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
      using the dedicated morph detector and SFace differential analysis.
    </p>

    <div class="morph-grid">
      <label class="morph-upload">
        <strong>TARGET / SUSPECTED IMAGE</strong>
        <span>Image being investigated</span>
        <input id="fallback-morph-target" type="file"
          accept="image/png,image/jpeg,image/webp">
        <p id="fallback-target-name" class="morph-name">No file selected</p>
      </label>

      <label class="morph-upload">
        <strong>REFERENCE A</strong>
        <span>Trusted genuine image of the same person</span>
        <input id="fallback-morph-ref-a" type="file"
          accept="image/png,image/jpeg,image/webp">
        <p id="fallback-ref-a-name" class="morph-name">No file selected</p>
      </label>

      <label class="morph-upload">
        <strong>REFERENCE B <small>(OPTIONAL)</small></strong>
        <span>Second trusted genuine image</span>
        <input id="fallback-morph-ref-b" type="file"
          accept="image/png,image/jpeg,image/webp">
        <p id="fallback-ref-b-name" class="morph-name">No file selected</p>
      </label>
    </div>

    <div class="morph-actions">
      <button id="fallback-analyze-morph" type="button" disabled>
        Analyze Morph
      </button>
      <p id="fallback-morph-state" class="state">
        Select target and Reference A.
      </p>
    </div>

    <div id="fallback-morph-results" class="metrics" hidden></div>
  `;

  const videoPanel = document.querySelector("#video-panel");

  if (videoPanel) {
    videoPanel.after(panel);
  } else {
    document.querySelector("main")?.append(panel);
  }

  const target = document.querySelector("#fallback-morph-target");
  const refA = document.querySelector("#fallback-morph-ref-a");
  const refB = document.querySelector("#fallback-morph-ref-b");
  const button = document.querySelector("#fallback-analyze-morph");
  const state = document.querySelector("#fallback-morph-state");
  const results = document.querySelector("#fallback-morph-results");

  const update = () => {
    button.disabled = !(target.files[0] && refA.files[0]);
  };

  const bindName = (input, output) => {
    input.addEventListener("change", () => {
      output.textContent =
        input.files[0]?.name || "No file selected";
      update();
    });
  };

  bindName(
    target,
    document.querySelector("#fallback-target-name")
  );

  bindName(
    refA,
    document.querySelector("#fallback-ref-a-name")
  );

  bindName(
    refB,
    document.querySelector("#fallback-ref-b-name")
  );

  button.addEventListener("click", async () => {
    button.disabled = true;
    results.hidden = true;
    state.textContent = "Analyzing faces and morph evidence...";

    try {
      const form = new FormData();

      form.append("target", target.files[0]);
      form.append("reference_a", refA.files[0]);

      if (refB.files[0]) {
        form.append("reference_b", refB.files[0]);
      }

      const response = await fetch(
        "http://127.0.0.1:8000/analyze/morph",
        {
          method: "POST",
          body: form
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Morph analysis failed."
        );
      }

      const model = data.morph_model || {};
      const similarity = data.reference_similarities || {};
      const face = data.target_face || {};

      results.innerHTML = `
        <div class="metric">
          <span>Assessment</span>
          <strong>${data.label || "Not available"}</strong>
        </div>

        <div class="metric">
          <span>Confidence</span>
          <strong>${data.confidence || "Not available"}</strong>
        </div>

        <div class="metric">
          <span>Morph Score</span>
          <strong>
            ${Number.isFinite(Number(model.morph_score))
              ? (Number(model.morph_score) * 100).toFixed(2) + "%"
              : "Not available"}
          </strong>
        </div>

        <div class="metric">
          <span>Bona Fide Score</span>
          <strong>
            ${Number.isFinite(Number(model.bona_fide_score))
              ? (Number(model.bona_fide_score) * 100).toFixed(2) + "%"
              : "Not available"}
          </strong>
        </div>

        <div class="metric">
          <span>Target ↔ Reference A</span>
          <strong>
            ${Number.isFinite(Number(similarity.target_reference_a))
              ? Number(similarity.target_reference_a).toFixed(4)
              : "Not available"}
          </strong>
        </div>

        <div class="metric">
          <span>Target Face Confidence</span>
          <strong>
            ${Number.isFinite(Number(face.confidence))
              ? (Number(face.confidence) * 100).toFixed(2) + "%"
              : "Not available"}
          </strong>
        </div>

        <div class="metric">
          <span>Model</span>
          <strong>${model.model || "Not available"}</strong>
        </div>

        <div class="metric">
          <span>Device</span>
          <strong>${model.device || "Not available"}</strong>
        </div>

        ${
          data.similarity_difference !== null &&
          data.similarity_difference !== undefined
            ? `
              <div class="metric">
                <span>Similarity Difference</span>
                <strong>
                  ${Number(data.similarity_difference).toFixed(4)}
                </strong>
              </div>
            `
            : ""
        }

        <div class="assessment">
          <span class="assessment-label">FORENSIC EVIDENCE</span>
          <p>
            ${(data.evidence || [])
              .map(item => item)
              .join("<br>") || "No additional evidence reported."}
          </p>
        </div>
      `;

      results.hidden = false;
      state.textContent = "Morph analysis complete.";
    } catch (error) {
      state.textContent = error.message;
    } finally {
      button.disabled = !(target.files[0] && refA.files[0]);
    }
  });
})();
