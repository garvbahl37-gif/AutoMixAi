const API_BASE = "http://localhost:8000";

export const api = {
  /**
   * Upload an audio file.
   * @param {File} file
   * @returns {Promise<{file_id, filename, duration, message}>}
   */
  async uploadFile(file) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Upload failed");
    }
    return res.json();
  },

  /**
   * Analyze an uploaded file (BPM + beat detection).
   * @param {string} fileId
   * @returns {Promise<{file_id, bpm, beat_times, duration, sample_rate, message}>}
   */
  async analyzeFile(fileId) {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_id: fileId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Analysis failed");
    }
    return res.json();
  },

  /**
   * Generate a beat-synchronized mix from two tracks.
   * @param {string} fileIdA
   * @param {string} fileIdB
   * @param {number} crossfadeDuration
   * @returns {Promise<{output_file_id, duration, bpm_a, bpm_b, target_bpm, message}>}
   */
  async mixTracks(fileIdA, fileIdB, crossfadeDuration = 5.0) {
    const res = await fetch(`${API_BASE}/mix`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_id_a: fileIdA,
        file_id_b: fileIdB,
        crossfade_duration: crossfadeDuration,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Mixing failed");
    }
    return res.json();
  },

  /**
   * Get the download URL for a mixed output file.
   * @param {string} fileId
   * @returns {string}
   */
  getOutputUrl(fileId) {
    return `${API_BASE}/output/${fileId}`;
  },
};
