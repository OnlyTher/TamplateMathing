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
            int key = Cv2.WaitKey(1000);

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

                // 亚像素级轮廓优化
                var preciseContour = RefineContour(mainContour, gray);

                // 转换坐标系并绘制
                var globalContour = preciseContour
                    .Select(p => new Point(p.X + safeRect.X, p.Y + safeRect.Y))
                    .ToArray();

                // 凸包处理确保闭合
                var hull = Cv2.ConvexHull(globalContour);
                Cv2.Polylines(src, new[] { hull }, true, Scalar.Red, 2);

                // 质量评估可视化
                double quality = CalculateContourQuality(hull);
                Cv2.PutText(src, $"Quality: {quality:0.00}",
                    new Point(10, 30), HersheyFonts.HersheySimplex,
                    0.8, quality > 0.7 ? Scalar.Green : Scalar.Red, 2);
            }
        }
    }

    private static double CalculateContourQuality(IEnumerable<Point> contour)
    {
        // 综合质量评估（0-1）
        var points = contour.ToArray();

        // 凸性检测
        double hullArea = Cv2.ContourArea(Cv2.ConvexHull(points));
        double contourArea = Cv2.ContourArea(points);
        double convexity = contourArea / hullArea;

        // 角度一致性
        double angleVariance = CalculateAngleVariance(points);

        // 矩形匹配度
        var rect = Cv2.BoundingRect(points);
        double rectMatch = contourArea / rect.Area();

        return (convexity * 0.6 + (1 - angleVariance) * 0.3 + rectMatch * 0.1);
    }

    private static Point[] RefineContour(Point[] contour, Mat gray)
    {
        // 亚像素边缘优化
        var points = contour.Select(p => new Point2f(p.X, p.Y)).ToArray();
        Cv2.CornerSubPix(gray, points, new Size(3, 3), new Size(-1, -1),
            new TermCriteria(
            CriteriaTypes.Eps | CriteriaTypes.MaxIter, 30, 0.01));

        // 样条插值平滑
        return SmoothContour(points.Select(p => new Point((int)p.X, (int)p.Y)).ToArray());
    }


    #region 核心算法模块
    private static Point[][] GetContours(Mat grayImage)
    {
        Cv2.ImShow("1. Original Gray", grayImage);
        Cv2.WaitKey(1000); // 快速调试模式

        using (var validatedImage = Ensure8UC1(grayImage))
        using (var processed = new Mat())
        {
            // ===== [预处理阶段] =====
            // 修改点1：优化降噪参数
            Cv2.MedianBlur(validatedImage, processed, 7);  // 孔径7平衡降噪与细节保留
            using (var bilateralTemp = new Mat())
            {
                Cv2.BilateralFilter(processed, bilateralTemp, 9, 150, 150); // 增强空间域滤波
                bilateralTemp.CopyTo(processed);
            }
            using (var clahe = Cv2.CreateCLAHE(1.2, new Size(12, 12))) // 自适应直方图均衡
            {
                clahe.Apply(processed, processed);
            }

            // ===== [边缘增强阶段] =====
            // 修改点2：改进的Canny阈值计算
            CalculateCannyThresholds(processed, out double low, out double high, 0.7);
            Cv2.Canny(processed, processed, low, high);
            Cv2.ImShow("2. Canny Edges", processed);
            Cv2.WaitKey(1000);

            // ===== [线段连接阶段] =====
            // 修改点3：十字特征定向连接
            using (var connectedEdges = new Mat())
            {
                // 水平方向连接（15像素长度阈值）
                var hKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 1));
                Cv2.MorphologyEx(processed, connectedEdges, MorphTypes.Close, hKernel, iterations: 1);

                // 垂直方向连接（15像素长度阈值）
                var vKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(1, 5));
                Cv2.MorphologyEx(connectedEdges, connectedEdges, MorphTypes.Close, vKernel, iterations: 1);

                connectedEdges.CopyTo(processed);
            }
            Cv2.ImShow("3. Connected Edges", processed);
            Cv2.WaitKey(1000);

            // ===== [轮廓提取阶段] =====
            // 修改点4：使用树形结构保留层级关系
            Cv2.FindContours(processed, out Point[][] contours, out HierarchyIndex[] hierarchy,
                RetrievalModes.Tree,  // 重要！保留轮廓层级
                ContourApproximationModes.ApproxNone); // 保留所有轮廓点

            // ===== [十字特征筛选] =====
            List<Point[]> crossCandidates = new List<Point[]>();
            foreach (var contour in contours)
            {
                // 基础过滤：最小点数限制
                if (contour.Length < 25) continue; // 修改点5：增加细节要求

                // 计算旋转矩形特征
                var rotatedRect = Cv2.MinAreaRect(contour);

                // 特征1：宽高比过滤（0.8-1.2范围）
                double aspectRatio = rotatedRect.Size.Width / rotatedRect.Size.Height;
                if (aspectRatio < 0.8 || aspectRatio > 1.2) continue;

                // 特征2：中心区域检测
                var centerROI = new Rect(
                    (int)(rotatedRect.Center.X - rotatedRect.Size.Width * 0.3),
                    (int)(rotatedRect.Center.Y - rotatedRect.Size.Height * 0.3),
                    (int)(rotatedRect.Size.Width * 0.6),
                    (int)(rotatedRect.Size.Height * 0.6));
                if (!contour.Any(p => centerROI.Contains(p))) continue;

                // 特征3：辐射状线段检测
                if (CountRadialLines(contour, rotatedRect.Center) < 3) continue; // 至少3个方向

                crossCandidates.Add(contour);
            }

            // ===== [最优轮廓选择] =====
            // 修改点6：多维度评分系统
            var bestContour = crossCandidates
                .Select(c => new {
                    Contour = c,
                    Area = Cv2.ContourArea(c),
                    LineScore = CountRadialLines(c, Cv2.MinAreaRect(c).Center),
                    FillRatio = CalculateFillRatio(c)
                })
                .OrderByDescending(x => x.LineScore * 0.5 + x.FillRatio * 0.3 + x.Area * 0.2)
                .FirstOrDefault()?.Contour;

            // ===== [轮廓后处理] =====
            // 修改点7：强制生成闭合轮廓
            if (bestContour != null)
            {
                // 多边形近似简化轮廓
                var approx = Cv2.ApproxPolyDP(bestContour,
                    0.015 * Cv2.ArcLength(bestContour, true),  // 动态epsilon
                    true);

                // 凸包处理确保闭合性
                var hull = Cv2.ConvexHull(approx);

                // 最终结果可视化
                using (var resultVis = grayImage.Clone())
                {
                    Cv2.CvtColor(resultVis, resultVis, ColorConversionCodes.GRAY2BGR);
                    Cv2.DrawContours(resultVis, new[] { hull }, -1, Scalar.Red, 2);

                    // 绘制特征点
                    foreach (var p in hull)
                    {
                        Cv2.Circle(resultVis, p, 3, Scalar.Green, -1);
                    }
                    Cv2.ImShow("Final Detection", resultVis);
                    Cv2.WaitKey(0);
                }

                return new[] { hull };
            }

            return new Point[0][];
        }
    }

    // 修改点8：辐射状线段计数
    private static int CountRadialLines(Point[] contour, Point2f center)
    {
        const double ANGLE_TOLERANCE = Math.PI / 6; // 30度容差
        var angleBins = new Dictionary<double, bool>
    {
        { 0, false },          // 右
        { Math.PI/2, false },  // 下
        { Math.PI, false },    // 左
        { 3*Math.PI/2, false } // 上
    };

        foreach (var p in contour)
        {
            double dx = p.X - center.X;
            double dy = p.Y - center.Y;
            double angle = Math.Atan2(dy, dx);
            angle = angle < 0 ? angle + 2 * Math.PI : angle; // 转换到0-2π范围

            foreach (var key in angleBins.Keys.ToList())
            {
                if (Math.Abs(angle - key) < ANGLE_TOLERANCE ||
                    Math.Abs(angle - key - 2 * Math.PI) < ANGLE_TOLERANCE)
                {
                    angleBins[key] = true;
                }
            }
        }

        return angleBins.Count(pair => pair.Value);
    }

    // 修改点9：填充率计算
    private static double CalculateFillRatio(Point[] contour)
    {
        var rect = Cv2.BoundingRect(contour);
        double contourArea = Cv2.ContourArea(contour);
        double rectArea = rect.Width * rect.Height;
        return rectArea > 0 ? contourArea / rectArea : 0;
    }

    // 修改点10：改进的Canny阈值计算
    private static void CalculateCannyThresholds(Mat gray, out double low, out double high, double percentile = 0.7)
    {
        using (var gradX = new Mat())
        using (var gradY = new Mat())
        {
            Cv2.Sobel(gray, gradX, MatType.CV_32F, 1, 0);
            Cv2.Sobel(gray, gradY, MatType.CV_32F, 0, 1);

            using (var magnitude = new Mat())
            {
                Cv2.Magnitude(gradX, gradY, magnitude);

                // 基于百分位的动态阈值
                double maxVal, medianVal;
                Cv2.MinMaxLoc(magnitude, out _, out maxVal);
                medianVal = CalculatePercentile(magnitude, percentile);

                high = Math.Min(maxVal * 0.85, medianVal * 3.2); // 提高高阈值
                low = high * 0.4; // 保持1:2.5比例
            }
        }
    }

    // 新增方法：计算轮廓中心（修改点12）
    private static Point CalculateContourCenter(Point[] contour)
    {
        Moments m = Cv2.Moments(contour);
        if (Math.Abs(m.M00) < 1e-7) // 避免除零错误
            return new Point(0, 0);
        return new Point(
            (int)(m.M10 / m.M00),
            (int)(m.M01 / m.M00)
        );
    }

    // 修改点12：添加类型安全转换方法
    private static Mat Ensure8UC1(Mat input)
    {
        if (input.Type() == MatType.CV_8UC1)
            return input.Clone();

        using (var temp = new Mat())
        {
            if (input.Channels() > 1)
            {
                Cv2.CvtColor(input, temp, ColorConversionCodes.BGR2GRAY);
            }
            else
            {
                input.ConvertTo(temp, MatType.CV_8UC1);
            }
            return temp.Clone();
        }
    }

    // 修改点13：基于梯度统计的Canny阈值计算

    // 修改点14：百分位数计算方法
    private static double CalculatePercentile(Mat mat, double percentile)
    {
        using (Mat flat = mat.Reshape(0, 1))
        {
            var sorted = new float[flat.Total()];
            System.Runtime.InteropServices.Marshal.Copy(flat.Data, sorted, 0, sorted.Length);
            Array.Sort(sorted);
            int index = (int)(sorted.Length * percentile);
            return sorted[index];
        }
    }

    //计算中值
    private static double CalculateMedian(Mat gray)
    {
        if (gray.Empty() || gray.Channels() != 1 || gray.Type() != MatType.CV_8UC1)
            return 0;

        using (var continuousMat = gray.Clone())
        using (var mat = continuousMat.Reshape(0, 1))
        {
            long total = mat.Total();
            if (total == 0) return 0;

            // 使用Marshal安全复制数据
            var sorted = new byte[total];
            System.Runtime.InteropServices.Marshal.Copy(
                source: mat.Data,
                destination: sorted,
                startIndex: 0,
                length: (int)total);

            Array.Sort(sorted);
            int mid = sorted.Length / 2;
            return (sorted.Length % 2 != 0) ?
                sorted[mid] :
                (sorted[mid - 1] + sorted[mid]) / 2.0;
        }
    }

    private static double CalculateContourScore(Point[] contour, Rect roiArea)
    {
        // 形状匹配度（0-1，越接近1越好）
        var approx = Cv2.ApproxPolyDP(contour, 0.02 * Cv2.ArcLength(contour, true), true);
        double areaRatio = Cv2.ContourArea(contour) / roiArea.Area();

        // 矩形匹配度
        var rect = Cv2.BoundingRect(contour);
        double rectMatch = Cv2.ContourArea(contour) / rect.Area();

        // 角度一致性
        double angleConsistency = 1 - CalculateAngleVariance(approx);

        // 综合评分（可调整权重）
        return rectMatch * 0.6 + angleConsistency * 0.3 + areaRatio * 0.1;
    }

    private static double CalculateAngleVariance(Point[] approx)
    {
        List<double> angles = new List<double>();
        for (int i = 0; i < approx.Length; i++)
        {
            Point p1 = approx[i];
            Point p2 = approx[(i + 1) % approx.Length];
            Point p3 = approx[(i + 2) % approx.Length];

            double v1x = p1.X - p2.X;
            double v1y = p1.Y - p2.Y;
            double v2x = p3.X - p2.X;
            double v2y = p3.Y - p2.Y;

            double dotProduct = v1x * v2x + v1y * v2y;
            double normV1 = Math.Sqrt(v1x * v1x + v1y * v1y);
            double normV2 = Math.Sqrt(v2x * v2x + v2y * v2y);
            double angle = Math.Acos(dotProduct / (normV1 * normV2));
            angles.Add(angle);
        }
        return angles.Variance();
    }

    private static Point[] SmoothContour(Point[] contour)
    {
        // Catmull-Rom样条插值
        List<Point> smoothed = new List<Point>();
        for (int i = 0; i < contour.Length; i++)
        {
            Point p0 = contour[(i - 1 + contour.Length) % contour.Length];
            Point p1 = contour[i];
            Point p2 = contour[(i + 1) % contour.Length];
            Point p3 = contour[(i + 2) % contour.Length];

            for (double t = 0; t < 1; t += 0.1)
            {
                double t2 = t * t;
                double t3 = t2 * t;

                int x = (int)(0.5 * ((-t3 + 2 * t2 - t) * p0.X + (3 * t3 - 5 * t2 + 2) * p1.X
                            + (-3 * t3 + 4 * t2 + t) * p2.X + (t3 - t2) * p3.X));
                int y = (int)(0.5 * ((-t3 + 2 * t2 - t) * p0.Y + (3 * t3 - 5 * t2 + 2) * p1.Y
                            + (-3 * t3 + 4 * t2 + t) * p2.Y + (t3 - t2) * p3.Y));
                smoothed.Add(new Point(x, y));
            }
        }

        // 精确多边形近似
        double epsilon = 0.005 * Cv2.ArcLength(smoothed, true);
        return Cv2.ApproxPolyDP(smoothed, epsilon, true);
    }


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

    public static double AspectRatio(this Rect rect)
    {
        return (rect.Height != 0) ? (double)rect.Width / rect.Height : double.NaN;
    }

    public static double Variance(this IEnumerable<double> values)
    {
        var list = values.ToList();
        if (list.Count == 0) return 0;
        double mean = list.Average();
        return list.Average(v => Math.Pow(v - mean, 2));
    }
}