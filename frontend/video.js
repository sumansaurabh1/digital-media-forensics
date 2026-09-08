(() => {
  if (document.querySelector("#video-panel")) return;

  const panel = document.createElement("section");
  panel.id = "video-panel";
  panel.className = "panel";
  panel.innerHTML = `
    <div class="section-heading">
      <div>
        <p class="section-kicker">VIDEO FORENSICS</p>
        <h2>AI Video Detection</h2>
      </div>
      <span class="card-tag">TEMPORAL ANALYSIS</span>
    </div>

    <p class="technical-intro">
      Analyze sampled video frames with CapCheck and inspect visible
      provenance-watermark evidence.
    </p>

    <input id="video-file" type="file" accept=".mp4,.mov,.avi,.webm">

    <button id="analyze-video" type="button" disabled>
      Analyze Video
    </button>

    <p id="video-state" class="state">
      Select a video.
    </p>

    <div id="video-results" class="metrics" hidden></div>
  `;

  const anchor =
    document.querySelector(".upload-panel") ||
    document.querySelector("main") ||
    document.body;

  anchor.after(panel);

  const input = document.querySelector("#video-file");
  const button = document.querySelector("#analyze-video");
  const state = document.querySelector("#video-state");
  const results = document.querySelector("#video-results");

  input.addEventListener("change", () => {
    button.disabled = !input.files[0];
    state.textContent = input.files[0]
      ? input.files[0].name
      : "Select a video.";
  });

  const pct = value =>
    Number.isFinite(Number(value))
      ? `${(Number(value) * 100).toFixed(2)}%`
      : "Not available";

  async function detectWatermark(file, samples = 12) {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    video.playsInline = true;

    const url = URL.createObjectURL(file);
    video.src = url;

    try {
      await new Promise((resolve, reject) => {
        video.onloadedmetadata = resolve;
        video.onerror = () =>
          reject(new Error("Video frame scan unavailable."));
      });

      const canvas = document.createElement("canvas");
      const scale = Math.min(
        1,
        640 / Math.max(video.videoWidth, video.videoHeight)
      );

      canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
      canvas.height = Math.max(1, Math.round(video.videoHeight * scale));

      const ctx = canvas.getContext("2d", {
        willReadFrequently: true
      });

      const hits = [];

      for (let i = 0; i < samples; i++) {
        video.currentTime =
          video.duration * i / Math.max(1, samples - 1);

        await new Promise((resolve, reject) => {
          video.onseeked = resolve;
          video.onerror = () =>
            reject(new Error("Video frame scan unavailable."));
        });

        ctx.drawImage(
          video,
          0,
          0,
          canvas.width,
          canvas.height
        );

        const { data, width, height } =
          ctx.getImageData(
            0,
            0,
            canvas.width,
            canvas.height
          );

        const lum = (x, y) => {
          const p = (y * width + x) * 4;
          return (
            data[p] * 0.299 +
            data[p + 1] * 0.587 +
            data[p + 2] * 0.114
          );
        };

        const d = Math.max(
          4,
          Math.round(Math.min(width, height) * 0.018)
        );

        let best = 0;

        for (
          let y = Math.round(height * 0.45);
          y < Math.round(height * 0.94);
          y += 4
        ) {
          for (
            let x = Math.round(width * 0.55);
            x < Math.round(width * 0.97);
            x += 4
          ) {
            if (
              x < d ||
              y < d ||
              x + d >= width ||
              y + d >= height
            ) continue;

            const center =
              (
                lum(x, y) +
                lum(x - d, y) +
                lum(x + d, y) +
                lum(x, y - d) +
                lum(x, y + d)
              ) / 5;

            const ring =
              (
                lum(x - d, y - d) +
                lum(x + d, y - d) +
                lum(x - d, y + d) +
                lum(x + d, y + d)
              ) / 4;

            const score = center - ring;

            if (center > 155 && score > best) {
              best = score;
            }
          }
        }

        hits.push(best >= 32);
      }

      const hitCount = hits.filter(Boolean).length;

      return {
        detected:
          hitCount >= Math.max(
            3,
            Math.ceil(samples * 0.25)
          ),
        hitCount,
        samples
      };
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  button.addEventListener("click", async () => {
    button.disabled = true;
    state.textContent = "Analyzing video frames...";
    results.hidden = true;

    try {
      const form = new FormData();
      form.append("video", input.files[0]);

      const response = await fetch(
        "http://127.0.0.1:8000/analyze/video",
        {
          method: "POST",
          body: form
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Video analysis failed."
        );
      }

      state.textContent =
        "Checking visible watermark evidence...";

      const watermark =
        await detectWatermark(input.files[0]);

      const assessment = watermark.detected
        ? "likely_ai"
        : data.label || "Not available";

      const confidence = watermark.detected
        ? "high"
        : data.confidence || "Not available";

      results.innerHTML = `
        <div class="metric">
          <span>Forensic Assessment</span>
          <strong>${assessment}</strong>
        </div>

        <div class="metric">
          <span>Confidence</span>
          <strong>${confidence}</strong>
        </div>

        <div class="metric">
          <span>Visible Watermark Evidence</span>
          <strong>
            ${watermark.detected
              ? "DETECTED"
              : "Not detected"}
          </strong>
        </div>

        <div class="metric">
          <span>Watermark Evidence</span>
          <strong>
            ${watermark.hitCount}/${watermark.samples} frames
          </strong>
        </div>

        <div class="metric">
          <span>Sampled Frames</span>
          <strong>
            ${data.sampled_frames ?? "Not available"}
          </strong>
        </div>

        <div class="metric">
          <span>Successful Frames</span>
          <strong>
            ${data.successful_frames ?? "Not available"}
          </strong>
        </div>

        <div class="metric">
          <span>Failed Frames</span>
          <strong>
            ${data.failed_frames ?? "Not available"}
          </strong>
        </div>

        <div class="metric">
          <span>Device</span>
          <strong>
            ${data.device || "Not available"}
          </strong>
        </div>

        <div class="metric">
          <span>Model</span>
          <strong>
            ${data.model || "Not available"}
          </strong>
        </div>
      `;

      results.hidden = false;

      state.textContent = watermark.detected
        ? "Analysis complete — visible watermark evidence detected."
        : "Video analysis complete.";
    } catch (error) {
      state.textContent = error.message;
    } finally {
      button.disabled = !input.files[0];
    }
  });
})();



