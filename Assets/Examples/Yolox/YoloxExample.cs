using System;
using UnityEngine;
using UnityEngine.UIElements;
using TensorFlowLite;
using TextureSource;

[RequireComponent(
    typeof(VirtualTextureSource),
    typeof(UIDocument))]
public sealed class YoloxExample : MonoBehaviour
{
    [SerializeField]
    Yolox.Options options;

    Yolox yolox;
    Image cameraBackground;
    Label[] boxes;

    void Start()
    {
        yolox = new Yolox(options);

        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.AddListener(OnTexture);
        }

        var root = GetComponent<UIDocument>().rootVisualElement;
        cameraBackground = root.Q<Image>("CameraBackground");

        // Create detection boxes
        boxes = new Label[options.maxDetections];
        for (int i = 0; i < options.maxDetections; i++)
        {
            var box = new Label
            {
                text = $"{i}",
                enableRichText = false,
                emojiFallbackSupport = false,
                usageHints = UsageHints.DynamicTransform,
                visible = false
            };
            box.style.position = Position.Absolute;
            box.AddToClassList("prototype-detection-box");
            boxes[i] = box;
            root.Add(box);
        }
    }

    void OnDestroy()
    {
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.RemoveListener(OnTexture);
        }

        yolox?.Dispose();
    }

    void OnTexture(Texture texture)
    {
        cameraBackground.image = texture;

        yolox.Run(texture);
        UpdateDetections(yolox.Detections);
    }

    void UpdateDetections(ReadOnlySpan<Yolox.Detection> detections)
    {
        var labelNames = yolox.labelNames;

        float width = cameraBackground.layout.width;
        float height = cameraBackground.layout.height;

        // Update detection boxes
        int i;
        for (i = 0; i < detections.Length; i++)
        {
            var detection = detections[i];
            var box = boxes[i];
            box.visible = true;
            box.text = $"{labelNames[detection.label]}: {(int)(detection.probability * 100)}%";

            Rect r = detection.rect;
            box.style.left = r.x * width;
            box.style.top = r.y * height;
            box.style.width = r.width * width;
            box.style.height = r.height * height;
        }
        for (; i < options.maxDetections; i++)
        {
            boxes[i].visible = false;
        }
    }
}
