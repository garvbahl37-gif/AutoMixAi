/**
 * AutoMixAI API Client
 *
 * In production (Vercel), calls go to the HuggingFace Space backend.
 * In development, calls go to the local backend on port 8002.
 *
 * HF Space URL: Update HF_SPACE_URL after creating your HF Space.
 */

const HF_SPACE_URL = "https://bharatverse11-automixbackend.hf.space";
const HF_BEAT_URL = "https://bharatverse11-automixai-beat-generator.hf.space";
const LOCAL_URL = "http://localhost:8002";

// Auto-detect: use HF Space in production, local in dev
const API_BASE =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? LOCAL_URL
    : HF_SPACE_URL;

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
   * Generate a beat-synchronized DJ mix from two tracks.
   * Supports advanced DJ mixing controls.
   * @param {string} fileIdA
   * @param {string} fileIdB
   * @param {number} crossfadeDuration
   * @param {object} options - { bassBoost, brightness, vocalBoost, panA, panB, eqTransition }
   * @returns {Promise<{output_file_id, duration, bpm_a, bpm_b, target_bpm, message}>}
   */
  async mixTracks(fileIdA, fileIdB, crossfadeDuration = 8.0, options = {}) {
    const res = await fetch(`${API_BASE}/mix`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_id_a: fileIdA,
        file_id_b: fileIdB,
        crossfade_duration: crossfadeDuration,
        bass_boost: options.bassBoost || 0.0,
        brightness: options.brightness || 0.0,
        vocal_boost: options.vocalBoost || 0.0,
        pan_a: options.panA || 0.0,
        pan_b: options.panB || 0.0,
        eq_transition: options.eqTransition !== false,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Mixing failed");
    }
    return res.json();
  },

  /**
   * Generate a synthesised drum beat from a text prompt.
   * @param {string} prompt
   * @param {number} bars
   * @returns {Promise<{output_file_id, genre, bpm, bars, complexity, description, duration, pattern}>}
   */
  async generateBeat(prompt, bars = 4) {
    const res = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, bars }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Beat generation failed");
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

  /**
   * AI Beat Generation via MusicGen (separate HF Space).
   * @param {string} prompt
   * @param {number} duration - seconds (3-30)
   * @param {number} temperature - creativity (0.5-1.5)
   * @param {number} guidanceScale - prompt adherence (1.0-10.0)
   * @returns {Promise<{output_file_id, prompt, duration, model, sample_rate, message}>}
   */
  async generateBeatAI(prompt, duration = 10, temperature = 1.0, guidanceScale = 3.0) {
    const beatBase =
      window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
        ? LOCAL_URL
        : HF_BEAT_URL;

    const res = await fetch(`${beatBase}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        duration,
        temperature,
        guidance_scale: guidanceScale,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "AI beat generation failed");
    }
    return res.json();
  },

  /**
   * Get the download URL for an AI-generated beat.
   * @param {string} fileId
   * @returns {string}
   */
  getBeatOutputUrl(fileId) {
    const beatBase =
      window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
        ? LOCAL_URL
        : HF_BEAT_URL;
    return `${beatBase}/output/${fileId}`;
  },
};
