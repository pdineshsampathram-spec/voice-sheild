/**
 * AudioWorklet processor for capturing PCM audio.
 * Runs in a separate thread, avoiding main thread jank.
 * Converts float32 samples to int16 PCM and sends to main thread.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._bufferSize = 4096;
    this._buffer = new Float32Array(this._bufferSize);
    this._writeIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0]; // mono

    for (let i = 0; i < channelData.length; i++) {
      this._buffer[this._writeIndex++] = channelData[i];

      if (this._writeIndex >= this._bufferSize) {
        // Convert float32 to int16 PCM
        const pcm = new Int16Array(this._bufferSize);
        for (let j = 0; j < this._bufferSize; j++) {
          const s = Math.max(-1, Math.min(1, this._buffer[j]));
          pcm[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        this._writeIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
