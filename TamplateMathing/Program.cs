using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;

public class RobustTemplateMatcher
{
    // 配置参数
    private const double DefaultConfidenceThreshold = 0.75;
    private const int MinTemplateSize = 20;
    private const int MorphologySize = 3;
    private const int MorphologyIterations = 2;

    // 运行状态
    private static Mat _sourceImage;
    private static Mat _displayImage;
    private static Point _roiStart;
    private static Rect _selectedROI;
    private static bool _isSelecting;
    private static string _templatePath = "template.jpg";
    private static string _outputDir = "outputs";
    private static string _failureDir = "failures";

    static void Main(string[] args)
    {
        Directory.CreateDirectory(_outputDir);
        Directory.CreateDirectory(_failureDir);

        var imageFiles = Directory.GetFiles("sources")
            .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                       f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f)
            .ToArray();

        if (imageFiles.Length == 0)
        {
            Console.WriteLine("错误：未找到源图像！");
            return;
        }

        // 加载首张图像
        _sourceImage = Cv2.ImRead(imageFiles[0], ImreadModes.Color);
        if (_sourceImage.Empty())
        {
            Console.WriteLine("错误：首张图像加载失败");
            return;
        }

        // 交互式模板选择
        if (!SelectTemplate())
        {
            Console.WriteLine("模板选择失败！");
            return;
        }

        // 批量处理
        using (var template = Cv2.ImRead(_templatePath, ImreadModes.Color))
        {
            if (template.Empty())
            {
                Console.WriteLine("错误：模板加载失败");
                return;
            }

            double threshold = GetConfidenceThreshold(args);
            Console.WriteLine($"当前置信度阈值：{threshold:P0}");

            foreach (var path in imageFiles)
            {
                ProcessImage(path, template, threshold);
            }
        }

        Console.WriteLine($"\n处理完成！\n成功图像：{Path.GetFullPath(_outputDir)}" +
                         $"\n失败图像：{Path.GetFullPath(_failureDir)}");
    }

    private static bool SelectTemplate()
    {
        const string windowName = "选择模板区域 (拖拽选择，ENTER确认)";
        Cv2.NamedWindow(windowName, WindowFlags.GuiNormal);
        Cv2.SetMouseCallback(windowName, OnMouse);
        _displayImage = _sourceImage.Clone();

        bool success = false;
        while (true)
        {
            Cv2.ImShow(windowName, _displayImage);
            int key = Cv2.WaitKey(10);

            if (key == 27) break; // ESC退出

            if (key == 13) // ENTER确认
            {
                var safeROI = ValidateROI(_selectedROI);
                if (safeROI.Width >= MinTemplateSize && safeROI.Height >= MinTemplateSize)
                {
                    using (var template = new Mat(_sourceImage, safeROI))
                    {
                        Cv2.ImWrite(_templatePath, template);
                        Console.WriteLine($"模板已保存：{Path.GetFullPath(_templatePath)}");
                        success = true;
                        break;
                    }
                }
                Console.WriteLine($"ROI尺寸不足：{safeROI.Size}");
            }
        }
        Cv2.DestroyAllWindows();
        return success;
    }

    private static void ProcessImage(string sourcePath, Mat template, double threshold)
    {
        string fileName = Path.GetFileName(sourcePath);
        using (var src = Cv2.ImRead(sourcePath, ImreadModes.Color))
        {
            if (src.Empty())
            {
                MoveToFailure(sourcePath, "图像加载失败");
                return;
            }

            // 执行模板匹配
            using (var result = new Mat())
            {
                Cv2.MatchTemplate(src, template, result, TemplateMatchModes.CCoeffNormed);
                result.MinMaxLoc(out _, out double confidence, out _, out Point maxLoc);

                if (confidence < threshold)
                {
                    MoveToFailure(sourcePath, $"置信度不足 ({confidence:P2})");
                    return;
                }

                // 绘制优化后的结果
                DrawOptimizedResult(src, template, maxLoc, confidence);
                Cv2.ImWrite(Path.Combine(_outputDir, fileName), src);
                Console.WriteLine($"成功：{fileName} ({confidence:P2})");
            }
        }
    }

    private static void DrawOptimizedResult(Mat src, Mat template, Point matchLoc, double confidence)
    {
        Rect matchRect = new Rect(matchLoc, template.Size());
        Rect safeRect = ValidateROI(matchRect);

        // 获取优化后的轮廓
        using (var roi = new Mat(src, safeRect))
        using (var gray = new Mat())
        {
            Cv2.CvtColor(roi, gray, ColorConversionCodes.BGR2GRAY);
            var contours = GetContours(gray);

            // 绘制平滑轮廓
            foreach (var contour in contours)
            {
                var smoothContour = SmoothContour(contour)
                    .Select(p => new Point(p.X + safeRect.X, p.Y + safeRect.Y))
                    .ToArray();
                Cv2.Polylines(src, new[] { smoothContour }, true, Scalar.Red, 2);
            }

            // 绘制中心标记
            var center = new Point(
                safeRect.X + template.Width / 2,
                safeRect.Y + template.Height / 2
            );
            Cv2.DrawMarker(src, center, Scalar.Blue, MarkerTypes.Cross, 30, 2);
            Cv2.PutText(src, $"{confidence:P1}", center + new Point(-40, -40),
                HersheyFonts.HersheySimplex, 1.2, Scalar.Green, 2);
        }
    }

    #region 核心算法模块
    private static Point[][] GetContours(Mat grayImage)
    {
        using (var processed = new Mat())
        {
            // 噪声抑制预处理
            Cv2.MedianBlur(grayImage, processed, 3);
            Cv2.GaussianBlur(processed, processed, new Size(5, 5), 1.5);
            Cv2.EqualizeHist(processed, processed);

            // 边缘检测
            Cv2.Canny(processed, processed, 80, 160, 3);

            // 形态学优化
            var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(MorphologySize, MorphologySize));
            Cv2.MorphologyEx(processed, processed, MorphTypes.Close, kernel, iterations: MorphologyIterations);

            // 轮廓提取
            Cv2.FindContours(processed, out Point[][] contours, out _,
                RetrievalModes.List, ContourApproximationModes.ApproxTC89KCOS);

            // 轮廓筛选
            return contours.Where(c =>
            {
                double area = Cv2.ContourArea(c);
                if (area < 100) return false;

                var rect = Cv2.BoundingRect(c);
                double aspect = (double)rect.Width / rect.Height;
                double solidity = area / Cv2.ContourArea(Cv2.ConvexHull(c));

                return aspect.Between(0.3, 3.0) && solidity > 0.8;
            }).ToArray();
        }
    }

    private static Point[] SmoothContour(Point[] contour)
    {
        // 多边形近似
        var approx = Cv2.ApproxPolyDP(contour, 0.01 * Cv2.ArcLength(contour, true), true);

        // 高斯平滑
        var smoothed = new List<Point>();
        for (int i = 0; i < approx.Length; i++)
        {
            int prev = (i - 1 + approx.Length) % approx.Length;
            int next = (i + 1) % approx.Length;
            smoothed.Add(new Point(
                (approx[prev].X + 2 * approx[i].X + approx[next].X) / 4,
                (approx[prev].Y + 2 * approx[i].Y + approx[next].Y) / 4
            ));
        }
        return smoothed.ToArray();
    }
    #endregion

    #region 工具方法
    private static Rect ValidateROI(Rect roi)
    {
        return new Rect(
            Clamp(roi.X, 0, _sourceImage.Width - 1),
            Clamp(roi.Y, 0, _sourceImage.Height - 1),
            Clamp(roi.Width, 1, _sourceImage.Width - roi.X),
            Clamp(roi.Height, 1, _sourceImage.Height - roi.Y)
        );
    }

    private static double GetConfidenceThreshold(string[] args)
    {
        if (args.Length > 0 && double.TryParse(args[0], out double t))
            return Clamp_double(t, 0.5, 0.95);
        return DefaultConfidenceThreshold;
    }

    private static void MoveToFailure(string sourcePath, string reason)
    {
        try
        {
            string dest = Path.Combine(_failureDir, Path.GetFileName(sourcePath));
            File.Copy(sourcePath, dest, true);
            Console.WriteLine($"失败：{Path.GetFileName(sourcePath)} - {reason}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"文件保存失败：{ex.Message}");
        }
    }

    private static void OnMouse(MouseEventTypes e, int x, int y, MouseEventFlags f, IntPtr d)
    {
        var pt = new Point(x, y).Clamp(_sourceImage.Size());

        switch (e)
        {
            case MouseEventTypes.LButtonDown:
                _isSelecting = true;
                _roiStart = pt;
                _selectedROI = new Rect(pt, new Size(0, 0));
                break;

            case MouseEventTypes.MouseMove when _isSelecting:
                _displayImage = _sourceImage.Clone();
                Cv2.Rectangle(_displayImage, _roiStart, pt, Scalar.Red, 2);
                break;

            case MouseEventTypes.LButtonUp:
                _isSelecting = false;
                _selectedROI = new Rect(
                    Math.Min(_roiStart.X, pt.X),
                    Math.Min(_roiStart.Y, pt.Y),
                    Math.Abs(pt.X - _roiStart.X),
                    Math.Abs(pt.Y - _roiStart.Y)
                );
                break;
        }

    }
    public static int Clamp(int value, int min, int max)
    {
        return Math.Max(min, Math.Min(value, max));
    }

    public static double Clamp_double(double value, double min, double max)
    {
        return Math.Max(min, Math.Min(value, max));
    }
    #endregion
}


public static class Extensions
{
    public static int Clamp(int value, int min, int max)
    {
        return Math.Max(min, Math.Min(value, max));
    }
    public static Point Clamp(this Point p, Size size) => new Point(
        Clamp(p.X, 0, size.Width - 1),
        Clamp(p.Y, 0, size.Height - 1)
    );

    public static bool Between(this double value, double min, double max) =>
        value >= min && value <= max;
}