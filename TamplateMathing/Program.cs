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
    private static string _templatePath = "template.bmp";
    private static string _outputDir = "outputs";
    private static string _failureDir = "failures";

    static void Main(string[] args)
    {
        Directory.CreateDirectory(_outputDir);
        Directory.CreateDirectory(_failureDir);

        var imageFiles = Directory.GetFiles("sources")
            .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                       f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                       f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase))
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

        using (var roi = new Mat(src, safeRect))
        using (var gray = new Mat())
        {
            Cv2.CvtColor(roi, gray, ColorConversionCodes.BGR2GRAY);
            var contours = GetContours(gray);

            if (contours.Length > 0)
            {
                var mainContour = contours.First();

                // 转换坐标系到原图
                var globalContour = mainContour
                    .Select(p => new Point(p.X + safeRect.X, p.Y + safeRect.Y))
                    .ToArray();

                // 绘制最小面积轮廓
                Cv2.DrawContours(src, new[] { globalContour }, -1, Scalar.Red, 2);


                // 中心点计算
                var localCenter = CalculateContourCenter(mainContour);
                if (localCenter.X > 0 && localCenter.Y > 0)
                {
                    var globalCenter = new Point(
                        safeRect.X + localCenter.X,
                        safeRect.Y + localCenter.Y
                    );
                    Cv2.DrawMarker(src, globalCenter, Scalar.Blue, MarkerTypes.Cross, 30, 2);
                    Cv2.PutText(src, $"{confidence:P1}", globalCenter + new Point(-40, -40),
                        HersheyFonts.HersheySimplex, 1.2, Scalar.Green, 2);
                }
            }
        }
    }


    #region 核心算法模块
    private static Point[][] GetContours(Mat grayImage)
    {
        Cv2.ImShow("1. Original Gray", grayImage);
        Cv2.WaitKey(1000);

        using (var processed = new Mat())
        {
            // 增强边缘保留预处理
            Cv2.MedianBlur(grayImage, processed, 5);
            Cv2.ImShow("2. After MedianBlur", processed);
            Cv2.WaitKey(1000);


            // 使用经典Canny边缘检测
            Cv2.Canny(processed, processed, 70, 135);
            Cv2.ImShow("3. Canny Edges", processed);
            Cv2.WaitKey(1000);

            // 增大闭合运算的kernel尺寸并增加迭代
            var closeKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(8, 8));
            Cv2.MorphologyEx(processed, processed, MorphTypes.Close, closeKernel, iterations: 5);
            Cv2.ImShow("3. Canny Edges", processed);
            Cv2.WaitKey(1000);

            //自适应阈值
            Cv2.AdaptiveThreshold(processed, processed, 255,
                AdaptiveThresholdTypes.GaussianC,
                ThresholdTypes.Binary, 21, 5);
            Cv2.ImShow("4. After AdaptiveThreshold", processed);
            Cv2.WaitKey(1000);

            // 轮廓检测
            Cv2.FindContours(processed, out Point[][] contours, out HierarchyIndex[] hierarchy,
                RetrievalModes.List,  // 改为检索所有独立轮廓
                ContourApproximationModes.ApproxTC89KCOS);

            List<Point[]> validContours = new List<Point[]>();
            Rect roiArea = new Rect(0, 0, grayImage.Width, grayImage.Height);

            foreach (var contour in contours)
            {
                var boundRect = Cv2.BoundingRect(contour);
                bool isROIBorder =
                    (boundRect.X <= 2 && boundRect.Width >= roiArea.Width - 4) ||
                    (boundRect.Y <= 2 && boundRect.Height >= roiArea.Height - 4);

                // 优化筛选条件
                double area = Cv2.ContourArea(contour);
                double aspectRatio = (double)boundRect.Width / boundRect.Height;

                // 计算轮廓的圆形度（排除线状噪声）
                double perimeter = Cv2.ArcLength(contour, true);
                double circularity = (4 * Math.PI * area) / (perimeter * perimeter);

                // 调试信息输出
                Console.WriteLine($"候选轮廓: 面积={area} 宽高比={aspectRatio:0.00} 圆形度={circularity:0.00}");

                if (!isROIBorder &&
                    area > grayImage.Width * grayImage.Height * 0.1 && // 面积至少占ROI区域的10%
                    aspectRatio.Between(0.5, 2.5) &&     // 放宽宽高比限制
                    circularity > 0.3)                  // 排除线状轮廓
                {
                    validContours.Add(contour);
                }
            }

            // 关键修改3：按面积降序排列，取前3个候选
            var sortedContours = validContours
                .OrderByDescending(c => Cv2.ContourArea(c))
                .Take(3)
                .ToArray();

            // 关键修改4：添加形状验证（选择最接近矩形的轮廓）
            var bestContour = sortedContours
                .OrderBy(c =>
                {
                    var approx = Cv2.ApproxPolyDP(c, 0.02 * Cv2.ArcLength(c, true), true);
                    return Math.Abs(4 - approx.Length); // 四边形得分为0
                })
                .FirstOrDefault();

            return bestContour != null ? new[] { bestContour } : new Point[0][];
        }
    }


    private static Point CalculateContourCenter(Point[] contour)
    {
        Moments m = Cv2.Moments(contour);
        return m.M00 > 0 ?
            new Point((int)(m.M10 / m.M00), (int)(m.M01 / m.M00)) :
            new Point(-1, -1);
    }
    #endregion




    private static Point[] SmoothContour(Point[] contour)
    {
        // 增加点密度（插值处理）
        var denseContour = new List<Point>();
        for (int i = 0; i < contour.Length; i++)
        {
            Point p1 = contour[i];
            Point p2 = contour[(i + 1) % contour.Length];

            // 每两个点之间插入3个插值点
            for (double t = 0; t <= 1; t += 0.25)
            {
                denseContour.Add(new Point(
                    (int)(p1.X * (1 - t) + p2.X * t),
                    (int)(p1.Y * (1 - t) + p2.Y * t)
                ));
            }
        }

        // 使用带角度约束的多边形近似
        double epsilon = 0.02 * Cv2.ArcLength(denseContour, true);
        var approx = Cv2.ApproxPolyDP(denseContour, epsilon, true);


        // 在调试显示
        using (var debugImg = new Mat(400, 400, MatType.CV_8UC3, Scalar.Black))
        {
            Cv2.DrawContours(debugImg, new[] { denseContour }, -1, Scalar.Gray, 1);
            Cv2.DrawContours(debugImg, new[] { approx }, -1, Scalar.Green, 2);
            Cv2.ImShow("Contour Smoothing Process", debugImg);
            Cv2.WaitKey(1000);
        }

        return approx;
    }



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

    public static double Area(this Rect rect)
    {
        return rect.Width * (double)rect.Height;
    }
}