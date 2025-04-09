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
    private static string _failureDir = "failures";
    private static string _firstImagePath;

    static void Main(string[] args)
    {
        string sourcesDir = "sources";
        Directory.CreateDirectory(_outputDir);
        Directory.CreateDirectory(_failureDir);

        var imageFiles = Directory.GetFiles(sourcesDir)
           .Where(f => f.EndsWith(".jpg") || f.EndsWith(".png")).ToArray();

        if (imageFiles.Length == 0)
        {
            Console.WriteLine("未找到图像文件！");
            return;
        }

        _firstImagePath = imageFiles[0];
        _sourceImage = Cv2.ImRead(_firstImagePath, ImreadModes.Color);
        if (_sourceImage.Empty())
        {
            Console.WriteLine("首张图像加载失败");
            return;
        }

        if (!GenerateTemplate()) return;

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
        Console.WriteLine("处理完成！匹配成功的图像保存在: " + Path.GetFullPath(_outputDir));
    }

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

            if (key == 27) break;

            if (key == 13 && _selectedROI.Width > 0)
            {
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

    private static void ProcessImage(string sourcePath, string outputPath, Mat template)
    {
        using (Mat src = Cv2.ImRead(sourcePath, ImreadModes.Color))
        {
            if (src.Empty())
            {
                Console.WriteLine($"加载失败: {sourcePath}");
                return;
            }

            Rect roi = new Rect(0, 0, src.Width, src.Height);
            ProcessSingleImage(src, outputPath, template, roi);
        }
    }

    private static void ProcessSingleImage(Mat src, string outputPath, Mat template, Rect roi)
    {
        Rect safeROI = ValidateROI(src, roi);
        if (safeROI.Width <= 0 || safeROI.Height <= 0)
        {
            Console.WriteLine($"{Path.GetFileName(outputPath)}: 无效的ROI区域");
            return;
        }

        using (Mat roiMat = new Mat(src, safeROI))
        {
            Point matchLoc;
            double confidence;
            if (!MatchTemplate(roiMat, template, out matchLoc, out confidence))
            {
                string failureReason = confidence == -1 ? "模板匹配异常" : $"置信度不足 ({confidence:P2})";
                HandleFailure(src, outputPath, failureReason);
                return;
            }

            Point actualLoc = new Point(safeROI.X + matchLoc.X, safeROI.Y + matchLoc.Y);

            DrawResult(src, template, actualLoc, confidence);
            Cv2.ImWrite(outputPath, src);
            Console.WriteLine($"{Path.GetFileName(outputPath)}: 置信度 {confidence:P2}");
        }
    }

    private static void HandleFailure(Mat src, string outputPath, string failureReason)
    {
        string fileName = Path.GetFileName(outputPath);
        string failurePath = Path.Combine(_failureDir, fileName);
        Cv2.ImWrite(failurePath, src);
        Console.WriteLine($"匹配失败：{fileName}，原因：{failureReason}");
    }

    private static Rect ValidateROI(Mat image, Rect roi)
    {
        int x = Clamp(roi.X, 0, image.Width - 1);
        int y = Clamp(roi.Y, 0, image.Height - 1);
        int width = Clamp(roi.Width, 1, image.Width - x);
        int height = Clamp(roi.Height, 1, image.Height - y);
        return new Rect(x, y, width, height);
    }

    private static int Clamp(int value, int min, int max) =>
        (value < min) ? min : (value > max) ? max : value;

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
            confidence = -1;
            return false;
        }
    }

    private static void DrawResult(Mat src, Mat template, Point location, double confidence)
    {
        // 仅修改DrawResult方法，其他代码保持不变
        Rect drawRect = new Rect(location.X, location.Y, template.Width, template.Height);
        Rect safeDrawRect = ValidateROI(src, drawRect);

        using (Mat safeRegion = new Mat(src, safeDrawRect))
        using (Mat gray = new Mat())
        {
            Cv2.CvtColor(safeRegion, gray, ColorConversionCodes.BGR2GRAY);
            Point[][] contours = GetContours(gray);

            // 核心修改：通过轮廓计算真实中心点
            Point center = CalculateContourCenter(contours);

            // 转换为全局坐标
            Point globalCenter = new Point(
                center.X + safeDrawRect.X,
                center.Y + safeDrawRect.Y
            );

            // 绘制轮廓
            foreach (var contour in contours)
            {
                Point[] globalContour = contour.Select(p =>
                    new Point(p.X + safeDrawRect.X, p.Y + safeDrawRect.Y)).ToArray();
                Cv2.Polylines(src, new[] { globalContour }, true, Scalar.Red, 1);
            }

            // 绘制基于轮廓的中心点
            if (globalCenter.X > 0 && globalCenter.Y > 0)
            {
                Cv2.DrawMarker(src, globalCenter, Scalar.Yellow, MarkerTypes.Cross, 10, 1);
                Cv2.PutText(src, $"Confidence: {confidence:P1}",
                    new Point(globalCenter.X - 45, globalCenter.Y - 25),
                    HersheyFonts.HersheySimplex, 0.3, Scalar.Blue, 1);
            }
        }
    }

    /// <summary>
    /// 新增方法：通过轮廓矩计算精确中心
    /// </summary>
    private static Point CalculateContourCenter(Point[][] contours)
    {
        if (contours.Length == 0) return new Point(-1, -1);

        // 找到面积最大的轮廓
        var largestContour = contours
            .OrderByDescending(c => Cv2.ContourArea(c))
            .First();

        // 计算轮廓矩
        Moments m = Cv2.Moments(largestContour);
        return new Point(
            (int)(m.M10 / m.M00),
            (int)(m.M01 / m.M00)
        );
    }

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