/**
 * Stack-and-record multiple canvases (WebGL + 2D overlay) into a single WebM file.
 *
 * Returns a {start, stop, isRecording} controller. Layers are drawn in order on
 * an offscreen canvas, so pass `[glCanvas, overlayCanvas]` for back-to-front.
 */
export interface RecorderController {
  start(): void;
  stop(): void;
  isRecording(): boolean;
}

export interface RecorderOptions {
  /** Captured frame rate (default 30). */
  fps?: number;
  /** Video bitrate in bits/sec (default 8 Mbps). */
  videoBitsPerSecond?: number;
  /** File-name prefix for the downloaded WebM (default `anny`). */
  filenamePrefix?: string;
}

export function createCanvasRecorder(
  layers: HTMLCanvasElement[],
  opts: RecorderOptions = {},
): RecorderController {
  const fps = opts.fps ?? 30;
  const bitrate = opts.videoBitsPerSecond ?? 8_000_000;
  const prefix = opts.filenamePrefix ?? "anny";

  if (layers.length === 0) throw new Error("createCanvasRecorder needs at least one canvas");
  const W = layers[0].width;
  const H = layers[0].height;

  let recorder: MediaRecorder | null = null;
  let chunks: Blob[] = [];

  return {
    start() {
      if (recorder && recorder.state === "recording") return;
      chunks = [];

      const merge = document.createElement("canvas");
      merge.width = W; merge.height = H;
      const mctx = merge.getContext("2d")!;
      const mergeStream = merge.captureStream(fps);

      const drawMerge = () => {
        if (!recorder || recorder.state !== "recording") return;
        for (const layer of layers) mctx.drawImage(layer, 0, 0);
        requestAnimationFrame(drawMerge);
      };

      recorder = new MediaRecorder(mergeStream, {
        mimeType: "video/webm;codecs=vp9",
        videoBitsPerSecond: bitrate,
      });
      recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${prefix}-${Date.now()}.webm`;
        a.click();
        URL.revokeObjectURL(url);
      };
      recorder.start();
      requestAnimationFrame(drawMerge);
    },

    stop() {
      if (recorder && recorder.state === "recording") recorder.stop();
    },

    isRecording() {
      return recorder?.state === "recording";
    },
  };
}
