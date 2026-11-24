/*
 * gesture.c - hand gesture recognition (left / right) on Raspberry Pi
 *
 * 1. Reads raw YUV420 frames from the camera via "rpicam-vid".
 * 2. Center-crops each frame, converts to RGB and resizes to the model size.
 * 3. Runs the TensorFlow Lite model and prints the detected label.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- TensorFlow Lite C API (libtensorflowlite_c.so) ----
 * Declared here because the headers installed by the libtensorflow-lite-c
 * recipe only include files that are not installed. */
typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteTensor TfLiteTensor;
enum { kTfLiteFloat32 = 1, kTfLiteUInt8 = 3 };

TfLiteModel *TfLiteModelCreateFromFile(const char *path);
TfLiteInterpreter *TfLiteInterpreterCreate(const TfLiteModel *m, const void *opts);
int TfLiteInterpreterAllocateTensors(TfLiteInterpreter *i);
int TfLiteInterpreterInvoke(TfLiteInterpreter *i);
TfLiteTensor *TfLiteInterpreterGetInputTensor(const TfLiteInterpreter *i, int32_t idx);
const TfLiteTensor *TfLiteInterpreterGetOutputTensor(const TfLiteInterpreter *i, int32_t idx);
int TfLiteTensorType(const TfLiteTensor *t);
int32_t TfLiteTensorDim(const TfLiteTensor *t, int32_t dim);
size_t TfLiteTensorByteSize(const TfLiteTensor *t);
int TfLiteTensorCopyFromBuffer(TfLiteTensor *t, const void *data, size_t size);
int TfLiteTensorCopyToBuffer(const TfLiteTensor *t, void *data, size_t size);

/* ---- settings ---- */
#define MODEL_PATH  "/usr/share/gesture/model.tflite"
#define LABELS_PATH "/usr/share/gesture/labels.txt"
#define CAM_W 640              /* multiple of 128 -> no row padding in YUV420 */
#define CAM_H 480
#define MIN_SCORE 0.70f        /* only report when the model is this sure */
#define MAX_CLASSES 16

static char labels[MAX_CLASSES][64];

static int load_labels(void)
{
    FILE *f = fopen(LABELS_PATH, "r");
    char line[128];
    int n = 0;
    if (!f) return 0;
    while (n < MAX_CLASSES && fgets(line, sizeof line, f)) {
        char *name = strchr(line, ' ');   /* Teachable Machine: "0 left" */
        name = name ? name + 1 : line;
        name[strcspn(name, "\r\n")] = 0;
        snprintf(labels[n++], sizeof labels[0], "%s", name);
    }
    fclose(f);
    return n;
}

static uint8_t clamp(float v) { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);   /* line-buffered so journalctl shows output */

    int nlabels = load_labels();
    TfLiteModel *model = TfLiteModelCreateFromFile(MODEL_PATH);
    if (!model || nlabels == 0) {
        fprintf(stderr, "cannot load %s or %s\n", MODEL_PATH, LABELS_PATH);
        return 1;
    }
    TfLiteInterpreter *itp = TfLiteInterpreterCreate(model, NULL);
    TfLiteInterpreterAllocateTensors(itp);

    TfLiteTensor *in = TfLiteInterpreterGetInputTensor(itp, 0);
    const TfLiteTensor *out = TfLiteInterpreterGetOutputTensor(itp, 0);
    int mh = TfLiteTensorDim(in, 1), mw = TfLiteTensorDim(in, 2);   /* e.g. 224x224 */
    int in_float = TfLiteTensorType(in) == kTfLiteFloat32;
    int out_float = TfLiteTensorType(out) == kTfLiteFloat32;
    int nout = TfLiteTensorByteSize(out) / (out_float ? 4 : 1);
    if (nout > MAX_CLASSES) nout = MAX_CLASSES;

    uint8_t *frame = malloc(CAM_W * CAM_H * 3 / 2);
    uint8_t *rgb = malloc(mw * mh * 3);
    float *rgbf = malloc(mw * mh * 3 * sizeof(float));
    uint8_t out8[MAX_CLASSES];
    float score[MAX_CLASSES];

    FILE *cam = popen("rpicam-vid -t 0 -n --width 640 --height 480 --framerate 5 "
                      "--codec yuv420 -o -", "r");
    if (!cam) { perror("rpicam-vid"); return 1; }

    const uint8_t *Y = frame, *U = frame + CAM_W * CAM_H, *V = U + CAM_W * CAM_H / 4;
    int crop = CAM_H, x0 = (CAM_W - CAM_H) / 2;     /* center square 480x480 */
    int last = -1;

    printf("gesture: started, model %dx%d, %d classes\n", mw, mh, nout);

    while (fread(frame, CAM_W * CAM_H * 3 / 2, 1, cam) == 1) {
        /* crop + resize (nearest neighbour) + YUV -> RGB */
        for (int y = 0; y < mh; y++) {
            for (int x = 0; x < mw; x++) {
                int sx = x0 + x * crop / mw, sy = y * crop / mh;
                float yy = Y[sy * CAM_W + sx];
                float u = U[(sy / 2) * (CAM_W / 2) + sx / 2] - 128.0f;
                float v = V[(sy / 2) * (CAM_W / 2) + sx / 2] - 128.0f;
                uint8_t *p = &rgb[(y * mw + x) * 3];
                p[0] = clamp(yy + 1.402f * v);
                p[1] = clamp(yy - 0.344f * u - 0.714f * v);
                p[2] = clamp(yy + 1.772f * u);
            }
        }

        if (in_float) {                          /* float model: scale to [-1, 1] */
            for (int i = 0; i < mw * mh * 3; i++) rgbf[i] = rgb[i] / 127.5f - 1.0f;
            TfLiteTensorCopyFromBuffer(in, rgbf, mw * mh * 3 * sizeof(float));
        } else {                                 /* quantized model: raw pixels */
            TfLiteTensorCopyFromBuffer(in, rgb, mw * mh * 3);
        }

        TfLiteInterpreterInvoke(itp);

        if (out_float) {
            TfLiteTensorCopyToBuffer(out, score, nout * sizeof(float));
        } else {
            TfLiteTensorCopyToBuffer(out, out8, nout);
            for (int i = 0; i < nout; i++) score[i] = out8[i] / 255.0f;
        }

        int best = 0;
        for (int i = 1; i < nout; i++) if (score[i] > score[best]) best = i;

        /* print only when the detected gesture changes */
        if (score[best] >= MIN_SCORE && best != last) {
            printf("%s (%.2f)\n", best < nlabels ? labels[best] : "?", score[best]);
            last = best;
        }
    }

    fprintf(stderr, "camera stream ended\n");
    pclose(cam);
    return 1;
}
