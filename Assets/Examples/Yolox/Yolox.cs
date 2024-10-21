using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.Collections;
using Unity.Mathematics;

namespace TensorFlowLite
{
    /// <summary>
    /// Licensed under Apache-2.0.
    /// See LICENSE for full license information.
    /// https://github.com/Megvii-BaseDetection/YOLOX
    /// 
    /// Converted Onnx model from PINTO_model_zoo
    /// Licensed under MIT.
    /// https://github.com/PINTO0309/PINTO_model_zoo/tree/main/132_YOLOX
    /// </summary>
    public sealed class Yolox : BaseVisionTask
    {
        [Serializable]
        public class Options
        {
            [FilePopup("*.tflite")]
            public string modelPath = string.Empty;
            public AspectMode aspectMode = AspectMode.Fit;
            public TfLiteDelegateType delegateType = TfLiteDelegateType.GPU;

            [Header("Yolox options")]
            public TextAsset labelFile;
            [Range(1, 100)]
            public int maxDetections = 10;
            [Range(0f, 1f)]
            public float probThreshold = 0.3f;
            [Range(0f, 1f)]
            public float nmsThreshold = 0.45f;
        }

        public readonly struct Detection : IComparable<Detection>
        {
            public readonly int label;
            public readonly Rect rect;
            public readonly float probability;

            public Detection(Rect rect, int label, float probability)
            {
                this.rect = rect;
                this.label = label;
                this.probability = probability;
            }

            public int CompareTo(Detection other)
            {
                return other.probability.CompareTo(probability);
            }
        }

        private readonly struct Anchor
        {
            public readonly int grid0;
            public readonly int grid1;
            public readonly int stride;

            public Anchor(int grid0, int grid1, int stride)
            {
                this.grid0 = grid0;
                this.grid1 = grid1;
                this.stride = stride;
            }

            public static Anchor[] GenerateAnchors(int width, int height)
            {
                ReadOnlySpan<int> strides = stackalloc int[] { 8, 16, 32 };
                List<Anchor> anchors = new();

                foreach (int stride in strides)
                {
                    int numGridY = height / stride;
                    int numGridX = width / stride;
                    for (int g1 = 0; g1 < numGridY; g1++)
                    {
                        for (int g0 = 0; g0 < numGridX; g0++)
                        {
                            anchors.Add(new Anchor(g0, g1, stride));
                        }
                    }
                }
                return anchors.ToArray();
            }
        }

        public readonly ReadOnlyCollection<string> labelNames;
        private const int NUM_CLASSES = 80;
        private readonly Anchor[] anchors;
        private readonly Options options;

        private NativeArray<float> output0;
        private NativeArray<Detection> proposalsArray;
        private NativeArray<Detection> detectionsArray;
        private int detectionCount = 0;

        public ReadOnlySpan<Detection> Detections => detectionsArray.AsReadOnlySpan()[..detectionCount];

        public Yolox(Options options)
        {
            this.options = options;

            var interpreterOptions = new InterpreterOptions();
            interpreterOptions.AutoAddDelegate(options.delegateType, typeof(float));
            Load(FileUtil.LoadFile(options.modelPath), interpreterOptions);

            int maxDetections = options.maxDetections;
            proposalsArray = new NativeArray<Detection>(maxDetections, Allocator.Persistent);
            detectionsArray = new NativeArray<Detection>(maxDetections, Allocator.Persistent);

            var labels = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            labelNames = Array.AsReadOnly(labels);
            Assert.AreEqual(NUM_CLASSES, labelNames.Count);
            anchors = Anchor.GenerateAnchors(width, height);

            var outShape = interpreter.GetOutputTensorInfo(0).shape;
            int outSize = outShape.Aggregate((a, b) => a * b);
            Debug.Log($"Output shape: {string.Join(", ", outShape)}, size: {outSize}");
            output0 = new NativeArray<float>(outSize, Allocator.Persistent);
        }

        public override void Dispose()
        {
            base.Dispose();
            output0.Dispose();
        }

        protected override void PostProcess()
        {
            Span<float> output = output0.AsSpan();
            interpreter.GetOutputTensorData(0, output);
            var proposals = GenerateProposals(output, options.probThreshold);
            proposals.Sort();
            detectionCount = NMS(proposals, detectionsArray, options.nmsThreshold);
        }

        // TODO: consider using Burst
        private NativeSlice<Detection> GenerateProposals(
            in ReadOnlySpan<float> feat_blob, float prob_threshold)
        {
            int num_anchors = anchors.Length;

            float widthScale = 1f / width;
            float heightScale = 1f / height;

            int proposalsCount = 0;

            for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
            {
                var anchor = anchors[anchor_idx];
                int grid0 = anchor.grid0;
                int grid1 = anchor.grid1;
                int stride = anchor.stride;

                int basic_pos = anchor_idx * (NUM_CLASSES + 5);

                // yolox/models/yolo_head.py decode logic
                float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
                float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
                float w = math.exp(feat_blob[basic_pos + 2]) * stride;
                float h = math.exp(feat_blob[basic_pos + 3]) * stride;
                // Normalize model space to 0..1
                x_center *= widthScale;
                y_center *= heightScale;
                w *= widthScale;
                h *= heightScale;

                // Skip if out of bounds
                if (x_center < 0 || x_center > 1 || y_center < 0 || y_center > 1)
                {
                    continue;
                }

                float x0 = x_center - w * 0.5f;
                float y0 = y_center - h * 0.5f;

                float box_objectness = feat_blob[basic_pos + 4];
                for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
                {
                    float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
                    float box_prob = box_objectness * box_cls_score;
                    if (box_prob > prob_threshold)
                    {
                        // Insert with sorted descent order
                        proposalsArray[proposalsCount] = new Detection(
                            new Rect(x0, y0, w, h),
                            class_idx,
                            box_prob
                        );
                        proposalsCount++;

                        if (proposalsCount >= proposalsArray.Length)
                        {
                            break;
                        }
                    }
                }

                if (proposalsCount >= proposalsArray.Length)
                {
                    break;
                }
            }

            if (proposalsCount == 0)
            {
                return proposalsArray.Slice(0, 0);
            }

            return proposalsArray.Slice(0, Math.Min(proposalsCount, proposalsArray.Length));
        }

        private static int NMS(
            in NativeSlice<Detection> proposals,
            NativeArray<Detection> detections,
            float iou_threshold)
        {
            int detectedCount = 0;

            foreach (Detection a in proposals)
            {
                bool keep = true;
                for (int i = 0; i < detectedCount; i++)
                {
                    Detection b = detections[i];

                    // Ignore different classes
                    if (b.label != a.label)
                    {
                        continue;
                    }
                    float iou = a.rect.IntersectionOverUnion(b.rect);
                    if (iou > iou_threshold)
                    {
                        keep = false;
                    }
                }
                if (keep)
                {
                    detections[detectedCount] = a;
                    detectedCount++;
                }
            }

            return detectedCount;
        }
    }
}
