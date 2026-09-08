(() => {
  const input = document.querySelector("#video-file");
  const button = document.querySelector("#analyze-video");
  const results = document.querySelector("#video-results");
  if (!input || !button || !results) return;

  async function watermarkCheck(file) {
    const video = document.createElement("video");
    video.src = URL.createObjectURL(file);
    video.muted = true;
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve;
      video.onerror = reject;
    });

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d", {willReadFrequently:true});
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const scores = [];
    const frames = 12;

    for (let i = 0; i < frames; i++) {
      video.currentTime = video.duration * i / (frames - 1);
      await new Promise(resolve => video.onseeked = resolve);
      ctx.drawImage(video, 0, 0);

      const w = canvas.width, h = canvas.height;
      const cx = Math.round(w * 0.908);
      const cy = Math.round(h * 0.837);
      const r = Math.max(8, Math.round(Math.min(w, h) * 0.025));
      const image = ctx.getImageData(cx-r, cy-r, r*2+1, r*2+1).data;

      let center = 0, ring = 0, centerN = 0, ringN = 0;

      for (let y = -r; y <= r; y++) {
        for (let x = -r; x <= r; x++) {
          const d = Math.abs(x) + Math.abs(y);
          const p = ((y+r)*(r*2+1)+(x+r))*4;
          const v = (image[p] + image[p+1] + image[p+2]) / 3;

          if (d <= Math.round(r * 0.65)) {
            center += v; centerN++;
          } else if (d >= Math.round(r * 0.9) && d <= r) {
            ring += v; ringN++;
          }
        }
      }

      scores.push(center / centerN - ring / ringN);
    }

    URL.revokeObjectURL(video.src);

    const hits = scores.filter(x => x > 25).length;
    return {
      detected: hits >= 3,
      hits,
      frames
    };
  }

  button.addEventListener("click", async () => {
    if (!input.files[0]) return;

    try {
      const watermark = await watermarkCheck(input.files[0]);

      const existing = results.innerHTML;

      results.innerHTML = existing + `
        <div class="metric">
          <span>Visible Watermark</span>
          <strong>${watermark.detected ? "DETECTED" : "Not detected"}</strong>
        </div>
        <div class="metric">
          <span>Watermark Evidence</span>
          <strong>${watermark.hits}/${watermark.frames} sampled frames</strong>
        </div>
      `;
    } catch (error) {
      console.warn("Watermark analysis unavailable:", error);
    }
  });
})();
