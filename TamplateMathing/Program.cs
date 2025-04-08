using System;
using System.IO;
using System.Linq;
using OpenCvSharp;

class InteractiveTemplateMatcher
{
    private static Mat _sourceImage;
    private static Mat _displayImage;
    private static Point _roiStart;
    private static Rect _selectedROI;
    private static bool _isSelecting;
    private static Point _currentMousePos = new Point(0, 0);
    private static string _templatePath = "template.jpg";
    private static string _outputDir = "outputs";
    private static string _firstImagePath;

    static void Main(string[] args)
    {
        string sourcesDir = "sources";

        // 初始化目录
        Directory.CreateDirectory(_outputDir);
        var imageFiles = Directory.GetFiles(sourcesDir)
            .Where(f => f.EndsWith(".jpg") || f.EndsWith(".png")).ToArray();

        if (imageFiles.Length == 0)
        {
            Console.WriteLine("未找到图像文件！");
            return;
        }

        // 第一阶段：处理首张图像生成模板
        _firstImagePath = imageFiles[0];
        _sourceImage = Cv2.ImRead(_firstImagePath, ImreadModes.Color);
        if (_sourceImage.Empty())
        {
            Console.WriteLine("首张图像加载失败");
            return;
        }

        if (!GenerateTemplate()) return;

        // 第二阶段：批量处理所有图像
        using (Mat template = Cv2.ImRead(_templatePath, ImreadModes.Color))
        {
            if (template.Empty())
            {
                Console.WriteLine("模板加载失败");
                return;
            }

            foreach (var path in imageFiles)
            {
                string outputPath = Path.Combine(_outputDir, Path.GetFileName(path));
                ProcessImage(path, outputPath, template);
            }
        }

        Console.WriteLine("处理完成！结果保存在: " + Path.GetFullPath(_outputDir));
    }

    /// <summary>
    /// 交互式生成模板
    /// </summary>
    private static bool GenerateTemplate()
    {
        const string windowName = "选择模板区域 (拖拽选择，按ENTER确认)";
        Cv2.NamedWindow(windowName, WindowFlags.GuiNormal);
        Cv2.SetMouseCallback(windowName, OnMouse);
        _displayImage = _sourceImage.Clone();

        bool success = false;
        while (true)
        {
            Cv2.ImShow(windowName, _displayImage);
            int key = Cv2.WaitKey(10);

            if (key == 27) break; // ESC退出

            if (key == 13 && _selectedROI.Width > 0) // ENTER确认
            {
                using (Mat template = new Mat(_sourceImage, _selectedROI))
                {
                    Cv2.ImWrite(_templatePath, template);
                    Console.WriteLine("模板已生成: " + _templatePath);

                    // 处理并保存首张结果
                    string firstOutput = Path.Combine(_outputDir, Path.GetFileName(_firstImagePath));
                    ProcessSingleImage(_sourceImage, firstOutput, template, _selectedROI);
                    success = true;
                }
                break;
            }
        }
        Cv2.DestroyWindow(windowName);
        return success;
    }

    /// <summary>
    /// 处理单张图像
    /// </summary>
    private static void ProcessImage(string sourcePath, string outputPath, Mat template)
    {
        using (Mat src = Cv2.ImRead(sourcePath, ImreadModes.Color))
        {
            if (src.Empty())
            {
                Console.WriteLine($"加载失败: {sourcePath}");
                return;
            }

            Rect roi = ValidateROI(src, _selectedROI);
            ProcessSingleImage(src, outputPath, template, roi);
        }
    }

    /// <summary>
    /// 核心处理逻辑
    /// </summary>
    private static void ProcessSingleImage(Mat src, string outputPath, Mat template, Rect roi)
    {
        // 模板匹配
        Point matchLoc;
        double confidence;
        using (Mat roiMat = new Mat(src, roi))
        {
            if (!MatchTemplate(roiMat, template, out matchLoc, out confidence)) return;
        }

        // 计算实际位置
        Point actualLoc = new Point(roi.X + matchLoc.X, roi.Y + matchLoc.Y);

        // 绘制结果
        DrawResult(src, template, actualLoc, confidence);
        Cv2.ImWrite(outputPath, src);
        Console.WriteLine($"{Path.GetFileName(outputPath)}: 置信度 {confidence:P2}");
    }

    /// <summary>
    /// ROI有效性校验
    /// </summary>
    private static Rect ValidateROI(Mat image, Rect roi)
    {
        return new Rect(
            Clamp(roi.X, 0, image.Width - 1),
            Clamp(roi.Y, 0, image.Height - 1),
            Clamp(roi.Width, 1, image.Width - roi.X),
            Clamp(roi.Height, 1, image.Height - roi.Y)
        );
    }

    public static int Clamp(int value, int min, int max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    /// <summary>
    /// 模板匹配算法
    /// </summary>
    private static bool MatchTemplate(Mat sourceROI, Mat template, out Point matchLoc, out double confidence)
    {
        matchLoc = new Point();
        confidence = 0;

        using (Mat srcGray = new Mat())
        using (Mat tmpGray = new Mat())
        using (Mat result = new Mat())
        {
            Cv2.CvtColor(sourceROI, srcGray, ColorConversionCodes.BGR2GRAY);
            Cv2.CvtColor(template, tmpGray, ColorConversionCodes.BGR2GRAY);

            Cv2.MatchTemplate(srcGray, tmpGray, result, TemplateMatchModes.CCoeffNormed);
            result.MinMaxLoc(out _, out confidence, out _, out matchLoc);

            return confidence > 0.75;
        }
    }

    /// <summary>
    /// 绘制结果（线宽调整为1像素）
    /// </summary>
    private static void DrawResult(Mat src, Mat template, Point location, double confidence)
    {
        // 绘制匹配区域轮廓
        using (Mat matchedRegion = new Mat(src, new Rect(location, template.Size())))
        using (Mat gray = new Mat())
        {
            Cv2.CvtColor(matchedRegion, gray, ColorConversionCodes.BGR2GRAY);
            foreach (var contour in GetContours(gray))
            {
                Point[] globalContour = contour.Select(p =>
                    new Point(p.X + location.X, p.Y + location.Y)).ToArray();
                Cv2.Polylines(src, new[] { globalContour }, true, Scalar.Cyan, 1); // 线宽1
            }
        }

        // 绘制中心十字（线宽2像素）
        Point center = new Point(location.X + template.Width / 2, location.Y + template.Height / 2);
        Cv2.DrawMarker(src, center, Scalar.Yellow, MarkerTypes.Cross, 20, 2);

        // 添加文字
        Cv2.PutText(src, $"Confidence: {confidence:P2}",
            new Point(center.X - 60, center.Y - 30),
            HersheyFonts.HersheySimplex, 0.6, Scalar.Yellow, 1);
    }

    /// <summary>
    /// 获取轮廓（过滤小面积噪声）
    /// </summary>
    private static Point[][] GetContours(Mat image)
    {
        using (Mat edges = new Mat())
        {
            Cv2.Canny(image, edges, 100, 200);
            Cv2.FindContours(edges, out Point[][] contours, out _,
                RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            return contours.Where(c => Cv2.ContourArea(c) > 30).ToArray();
        }
    }

    /// <summary>
    /// 鼠标回调（线宽调整为1像素）
    /// </summary>
    private static void OnMouse(MouseEventTypes @event, int x, int y, MouseEventFlags flags, IntPtr userdata)
    {
        _currentMousePos = new Point(x, y);
        switch (@event)
        {
            case MouseEventTypes.LButtonDown:
                _isSelecting = true;
                _roiStart = new Point(x, y);
                _selectedROI = new Rect(x, y, 0, 0);
                break;

            case MouseEventTypes.LButtonUp:
                _isSelecting = false;
                Point end = new Point(x, y);
                _selectedROI = new Rect(
                    Math.Min(_roiStart.X, end.X),
                    Math.Min(_roiStart.Y, end.Y),
                    Math.Abs(end.X - _roiStart.X),
                    Math.Abs(end.Y - _roiStart.Y)
                );
                _displayImage = _sourceImage.Clone();
                Cv2.Rectangle(_displayImage, _selectedROI, Scalar.Green, 1); // 确认框线宽1
                break;
        }

        // 实时绘制选择框（线宽1像素）
        if (_isSelecting)
        {
            _displayImage = _sourceImage.Clone();
            Cv2.Rectangle(_displayImage, _roiStart, _currentMousePos, Scalar.Red, 1);
        }
    }
}