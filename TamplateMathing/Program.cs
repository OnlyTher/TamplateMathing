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
        Directory.CreateDirectory(_outputDir);

        // 获取所有图像文件
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
    /// 交互式生成模板（核心安全校验）
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

            if (key == 13 && _selectedROI.Width > 0)
            {
                // ROI安全校验
                Rect safeROI = ValidateROI(_sourceImage, _selectedROI);
                if (safeROI.Width <= 0 || safeROI.Height <= 0)
                {
                    Console.WriteLine("无效的ROI区域！");
                    break;
                }

                using (Mat template = new Mat(_sourceImage, safeROI))
                {
                    Cv2.ImWrite(_templatePath, template);
                    Console.WriteLine("模板已生成: " + _templatePath);

                    // 处理并保存首张结果
                    string firstOutput = Path.Combine(_outputDir, Path.GetFileName(_firstImagePath));
                    ProcessSingleImage(_sourceImage, firstOutput, template, safeROI);
                    success = true;
                }
                break;
            }
        }
        Cv2.DestroyWindow(windowName);
        return success;
    }

    /// <summary>
    /// 处理单张图像（带安全校验）
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

            // 使用校验后的ROI
            Rect roi = ValidateROI(src, _selectedROI);
            ProcessSingleImage(src, outputPath, template, roi);
        }
    }

    /// <summary>
    /// 核心处理逻辑（三重安全校验）
    /// </summary>
    private static void ProcessSingleImage(Mat src, string outputPath, Mat template, Rect roi)
    {
        // 1. ROI二次校验
        Rect safeROI = ValidateROI(src, roi);
        if (safeROI.Width <= 0 || safeROI.Height <= 0)
        {
            Console.WriteLine($"{Path.GetFileName(outputPath)}: 无效的ROI区域");
            return;
        }

        // 2. 安全截取ROI区域
        using (Mat roiMat = new Mat(src, safeROI))
        {
            // 3. 模板匹配
            Point matchLoc;
            double confidence;
            if (!MatchTemplate(roiMat, template, out matchLoc, out confidence)) return;

            // 计算全局坐标（带边界检查）
            Point actualLoc = new Point(
                Math.Min(safeROI.X + matchLoc.X, src.Width - template.Width),
                Math.Min(safeROI.Y + matchLoc.Y, src.Height - template.Height)
            );

            // 最终安全校验
            if (actualLoc.X < 0 || actualLoc.Y < 0)
            {
                Console.WriteLine($"{Path.GetFileName(outputPath)}: 匹配位置越界");
                return;
            }

            // 绘制结果
            DrawResult(src, template, actualLoc, confidence);
            Cv2.ImWrite(outputPath, src);
            Console.WriteLine($"{Path.GetFileName(outputPath)}: 置信度 {confidence:P2}");
        }
    }

    /// <summary>
    /// ROI边界安全校验（核心方法）
    /// </summary>
    private static Rect ValidateROI(Mat image, Rect roi)
    {
        int x = Clamp(roi.X, 0, image.Width - 1);
        int y = Clamp(roi.Y, 0, image.Height - 1);
        int width = Clamp(roi.Width, 1, image.Width - x);
        int height = Clamp(roi.Height, 1, image.Height - y);
        return new Rect(x, y, width, height);
    }

    /// <summary>
    /// 数值范围限定（兼容所有C#版本）
    /// </summary>
    private static int Clamp(int value, int min, int max) =>
        (value < min) ? min : (value > max) ? max : value;

    /// <summary>
    /// 模板匹配算法（带异常捕获）
    /// </summary>
    private static bool MatchTemplate(Mat sourceROI, Mat template, out Point matchLoc, out double confidence)
    {
        matchLoc = new Point();
        confidence = 0;

        try
        {
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
        catch (Exception ex)
        {
            Console.WriteLine($"模板匹配异常: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// 安全绘制结果（四重保护）
    /// </summary>
    private static void DrawResult(Mat src, Mat template, Point location, double confidence)
    {
        // 1. 计算安全绘制区域
        Rect drawRect = new Rect(
            location.X,
            location.Y,
            Math.Min(template.Width, src.Width - location.X),
            Math.Min(template.Height, src.Height - location.Y)
        );

        // 2. 安全校验
        if (drawRect.Width <= 0 || drawRect.Height <= 0)
        {
            Console.WriteLine("绘制区域无效");
            return;
        }

        // 3. 安全截取区域
        using (Mat safeRegion = new Mat(src, drawRect))
        using (Mat gray = new Mat())
        {
            try
            {
                Cv2.CvtColor(safeRegion, gray, ColorConversionCodes.BGR2GRAY);
                Point[][] contours = GetContours(gray);

                // 4. 安全坐标转换
                foreach (var contour in contours)
                {
                    Point[] globalContour = contour.Select(p =>
                        new Point(p.X + location.X, p.Y + location.Y)).ToArray();
                    Cv2.Polylines(src, new[] { globalContour }, true, Scalar.Red, 1);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"轮廓绘制异常: {ex.Message}");
            }
        }

        // 绘制中心十字（最终安全位置）
        Point center = new Point(
            Clamp(location.X + template.Width / 2, 0, src.Width - 1),
            Clamp(location.Y + template.Height / 2, 0, src.Height - 1)
        );
        Cv2.DrawMarker(src, center, Scalar.Yellow, MarkerTypes.Cross, 10, 1);

        // 添加文字（自动避让边界）
        int textX = Clamp(center.X - 60, 10, src.Width - 120);
        int textY = Clamp(center.Y - 30, 20, src.Height - 20);
        Cv2.PutText(src, $"Confidence: {confidence:P2}",
            new Point(textX, textY),
            HersheyFonts.HersheySimplex, 0.5, Scalar.Yellow, 1);
    }

    /// <summary>
    /// 获取安全轮廓（带异常处理）
    /// </summary>
    private static Point[][] GetContours(Mat image)
    {
        try
        {
            using (Mat edges = new Mat())
            {
                Cv2.Canny(image, edges, 100, 200);
                Cv2.FindContours(edges, out Point[][] contours, out _,
                    RetrievalModes.External, ContourApproximationModes.ApproxSimple);
                return contours.Where(c => Cv2.ContourArea(c) > 30).ToArray();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"轮廓提取异常: {ex.Message}");
            return Array.Empty<Point[]>();
        }
    }

    /// <summary>
    /// 鼠标回调（带绘制保护）
    /// </summary>
    private static void OnMouse(MouseEventTypes @event, int x, int y, MouseEventFlags flags, IntPtr userdata)
    {
        try
        {
            _currentMousePos = new Point(
                Clamp(x, 0, _sourceImage.Width - 1),
                Clamp(y, 0, _sourceImage.Height - 1)
            );

            switch (@event)
            {
                case MouseEventTypes.LButtonDown:
                    _isSelecting = true;
                    _roiStart = _currentMousePos;
                    _selectedROI = new Rect(_currentMousePos, new Size(0, 0));
                    break;

                case MouseEventTypes.LButtonUp:
                    _isSelecting = false;
                    Point end = _currentMousePos;
                    _selectedROI = new Rect(
                        Math.Min(_roiStart.X, end.X),
                        Math.Min(_roiStart.Y, end.Y),
                        Math.Abs(end.X - _roiStart.X),
                        Math.Abs(end.Y - _roiStart.Y)
                    );
                    _displayImage = _sourceImage.Clone();
                    Cv2.Rectangle(_displayImage, ValidateROI(_sourceImage, _selectedROI), Scalar.Green, 1);
                    break;
            }

            if (_isSelecting)
            {
                _displayImage = _sourceImage.Clone();
                Cv2.Rectangle(_displayImage, _roiStart, _currentMousePos, Scalar.Red, 1);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"鼠标回调异常: {ex.Message}");
        }
    }
}