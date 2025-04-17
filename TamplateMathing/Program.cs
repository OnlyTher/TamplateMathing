using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;

public class RobustTemplateMatcher
{
    // 配置参数
    private const double DefaultConfidenceThreshold = 0.45;
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
            int key = Cv2.WaitKey(20);

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
            var contours = GetContours(gray, safeRect);
            // ==== 新增调试输出1 ====
            Console.WriteLine($"发现轮廓数量：{contours.Length}");

            var bestContour = contours
            .OrderBy(c => CalculateCompactness(c))
            .FirstOrDefault();

            if (bestContour != null)
            {

                // ==== 新增：计算并绘制中心点 ====
                // 计算轮廓中心（局部坐标系）
                Point localCenter = CalculateContourCenter(bestContour);

                // 转换为全局坐标
                Point globalCenter = new Point(
                    localCenter.X + safeRect.X,
                    localCenter.Y + safeRect.Y
                );

                // 绘制正十字标记（绿色十字，尺寸自适应）
                int crossSize = (int)(Math.Max(template.Width, template.Height) * 0.03); // 15%的模板尺寸
                Cv2.Line(src,
                    new Point(globalCenter.X - crossSize, globalCenter.Y),
                    new Point(globalCenter.X + crossSize, globalCenter.Y),
                    Scalar.FromRgb(0, 255, 0),  // 亮绿色
                    2);                          // 线宽
                Cv2.Line(src,
                    new Point(globalCenter.X, globalCenter.Y - crossSize),
                    new Point(globalCenter.X, globalCenter.Y + crossSize),
                    Scalar.FromRgb(0, 255, 0),
                    2);

                // ==== 优化轮廓 ====
                var preciseContour = RefineContour(bestContour, gray)
                    .Select(p => new Point(p.X + safeRect.X, p.Y + safeRect.Y))
                    .ToArray();

                if (preciseContour.Length > 1)
                {
                    // 精确对齐首尾点（无误差补偿）
                    preciseContour[preciseContour.Length  - 1] = preciseContour[0];
                }

                // ==== 关键修改：确保只绘制一个轮廓 ====
                if (preciseContour.Length > 2) // 确保是有效多边形
                {
                    // 使用单一轮廓绘制模式
                    Cv2.DrawContours(
                        image: src,
                        contours: new[] { preciseContour },
                        contourIdx: 0,
                        color: Scalar.Red,
                        thickness: 2,
                        lineType: LineTypes.AntiAlias);

                    // 标记首尾点
                    Cv2.Circle(src, preciseContour[0], 5, Scalar.Blue, -1);
                    Cv2.Circle(src, preciseContour[preciseContour.Length - 1], 3, Scalar.Green, -1);

                    Console.WriteLine($"最终轮廓点数：{preciseContour.Length}，首点：{preciseContour[0]},末点：{preciseContour[preciseContour.Length - 1]}");
                }

            }
        }
    }

    // ===== 新增向量结构体 =====
    private struct Vector2
    {
        public float X;
        public float Y;

        public Vector2(float x, float y)
        {
            X = x;
            Y = y;
        }

        public static Vector2 operator +(Vector2 a, Vector2 b) =>
            new Vector2(a.X + b.X, a.Y + b.Y);
    }


    // 修改 RefineContour 方法（保留凹点关键修改）
    private static Point[] RefineContour(Point[] contour, Mat gray)
    {
        if (contour == null || contour.Length < 3) return contour ?? Array.Empty<Point>();

        // 步骤1：亚像素优化（保持原逻辑）
        var points = contour.Select(p => new Point2f(p.X, p.Y)).ToArray();
        Cv2.CornerSubPix(gray, points, new Size(3, 3), new Size(-1, -1),
            new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 5, 0.01));

        // 步骤2：插值优化（保持原SmoothContour逻辑）
        var smoothed = SmoothContour(contour);

        // ==== 关键修复1：精确闭合验证 ====
        if (smoothed.Length > 0 && smoothed[0] != smoothed[smoothed.Length - 1])
        {
            smoothed = smoothed.Concat(new[] { smoothed[0] }).ToArray();
        }

        // ==== 关键修复2：暴力去重 ====
        List<Point> cleaned = new List<Point>();
        foreach (var p in smoothed)
        {
            if (cleaned.Count == 0 || p != cleaned[cleaned.Count - 1])
                cleaned.Add(p);
        }

        // ==== 关键修复3：强制四像素闭合 ====
        if (cleaned.Count > 1)
        {
            var first = cleaned[0];
            var last = cleaned[cleaned.Count - 1];

            if (Math.Abs(first.X - last.X) + Math.Abs(first.Y - last.Y) > 4)
            {
                cleaned.Add(first);
            }
        }
        else if (cleaned.Count == 1)
        {
            // 单点保护：复制点形成线段
            cleaned.Add(cleaned[0]);
        }

        return cleaned.ToArray();
    }

    #region 核心算法模块
    private static Point[][] GetContours(Mat grayImage, Rect safeRect)
    {
        Cv2.ImShow("1. Original Gray", grayImage);
        Cv2.WaitKey(50); // 快速调试模式

        using (var validatedImage = Ensure8UC1(grayImage))
        using (var processed = new Mat())
        {
            double sizeFactor = Math.Sqrt(safeRect.Area()) / 100.0; // 基准尺寸100x100

            Cv2.MedianBlur(validatedImage, processed, 7);  // 孔径7-》5平衡降噪与细节保留

            using (var bilateralTemp = new Mat())
            {
                Cv2.BilateralFilter(processed, bilateralTemp, 7, 120, 120); // 增强空间域滤波
                bilateralTemp.CopyTo(processed);
            }
            //using (var clahe = Cv2.CreateCLAHE(1.2, new Size(12, 12))) // 自适应直方图均衡
            //{
            //    clahe.Apply(processed, processed);
            //}
            //Cv2.ImShow("1.B CreateCLAHE", processed);
            //Cv2.WaitKey(0); // 快速调试模式
            // ===== [边缘增强阶段] =====
            // 修改点2：改进的Canny阈值计算
            CalculateCannyThresholds(processed, out double low, out double high, 0.7);
            Cv2.Canny(processed, processed, low, high);
            Cv2.ImShow("2. Canny Edges", processed);
            Cv2.WaitKey(50);

            // ===== [线段连接阶段] =====
            // 修改点3：十字特征定向连接
            using (var connectedEdges = new Mat())
            {
                // 使用旋转矩形核进行多方向连接
                var rotatedKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 6));
                Cv2.MorphologyEx(processed, connectedEdges, MorphTypes.Close, rotatedKernel, iterations: 1);
                Cv2.Rotate(rotatedKernel, rotatedKernel, RotateFlags.Rotate90Clockwise);
                Cv2.MorphologyEx(connectedEdges, connectedEdges, MorphTypes.Close, rotatedKernel, iterations: 1);

                // 最终使用交叉核强化连接
                var crossKernel = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(5, 5));
                Cv2.MorphologyEx(connectedEdges, connectedEdges, MorphTypes.Close, crossKernel, iterations: 2);
                connectedEdges.CopyTo(processed);
            }
            Cv2.ImShow("3. Connected Edges", processed);
            Cv2.WaitKey(50);

            // ===== [轮廓提取阶段] =====
            // 修改点4：使用树形结构保留层级关系
            Cv2.FindContours(processed, out Point[][] contours, out HierarchyIndex[] hierarchy,
                RetrievalModes.Tree,  // 重要！保留轮廓层级
                ContourApproximationModes.ApproxNone); // 保留所有轮廓点

            // ===== [十字特征筛选] =====
            List<Point[]> crossCandidates = new List<Point[]>();
            foreach (var contour in contours)
            {
                // 改为基于ROI尺寸的动态阈值
                int minPoints = (int)(20 * Math.Max(1, sizeFactor)); // 最小点数15~动态调整
                if (contour.Length < minPoints)
                {
                    Console.WriteLine($"轮廓点数不足：{contour.Length}/{minPoints}");
                    continue;
                }

                // 计算旋转矩形特征
                var currentRotatedRect = Cv2.MinAreaRect(contour);

                // 特征1：宽高比过滤（0.8-1.2范围）
                double aspectRatio = currentRotatedRect.Size.Width / currentRotatedRect.Size.Height;
                // 改为尺寸自适应容差
                double aspectTolerance = 0.5 * (1 + 1 / sizeFactor); // ROI越小容差越大
                double targetRatio = 1.0;
                if (Math.Abs(aspectRatio - targetRatio) > aspectTolerance)
                {
                    Console.WriteLine($"宽高比超限：{aspectRatio:F2} (容差±{aspectTolerance:F2})");
                    continue;
                }

                // 动态中心区域范围 0.4
                double centerRatio = 0.4 - 0.1 * (1 / sizeFactor); // ROI越小检测区域越大
                var centerROI = new Rect(
                    (int)(currentRotatedRect.Center.X - currentRotatedRect.Size.Width * centerRatio),
                    (int)(currentRotatedRect.Center.Y - currentRotatedRect.Size.Height * centerRatio),
                    (int)(currentRotatedRect.Size.Width * centerRatio * 2),
                    (int)(currentRotatedRect.Size.Height * centerRatio * 2)
                );

                // 降低检测密度要求
                int requiredPoints = (int)(contour.Length * 0.15); // 只需10%的点在中心区
                if (contour.Count(p => centerROI.Contains(p)) < requiredPoints)
                {
                    Console.WriteLine($"中心点不足：{requiredPoints}");
                    continue;
                }

                // 动态方向要求 2：3
                int minDirections = sizeFactor < 0.8 ? 3 : 4; // 小ROI只需2个方向
                if (CountRadialLines(contour, currentRotatedRect.Center) < minDirections)
                {
                    Console.WriteLine($"辐射方向不足：{minDirections}");
                    continue;
                }

                // 修改角度检测容差（原ANGLE_TOLERANCE）
                double angleTolerance = Math.PI / (4 + sizeFactor * 2); // ROI越小容差越大

                crossCandidates.Add(contour);
            }

            // ===== [最优轮廓选择] =====
            var bestContour = crossCandidates
                .OrderBy(c => CalculateCompactness(c))  // 按紧凑度排序
                .ThenBy(c => GetDistance(Cv2.BoundingRect(c).GetCenter(), _selectedROI.GetCenter()))
                .FirstOrDefault();  // 强制取第一个

            // ===== [轮廓后处理] =====
            // 强制生成闭合轮廓
            if (bestContour != null)
            {
                // 多边形近似简化轮廓
                var approx = Cv2.ApproxPolyDP(bestContour,
                    0.005 * Cv2.ArcLength(bestContour, true),  // 动态epsilon
                    true);

                return new[] { approx };
            }

            return new Point[0][];
        }
    }


    // 添加紧凑度计算（放在类中）
    private static double CalculateCompactness(Point[] contour)
    {
        double perimeter = Cv2.ArcLength(contour, true);
        double area = Cv2.ContourArea(contour);
        return area > 0 ? Math.Pow(perimeter, 2) / (4 * Math.PI * area) : double.MaxValue;
    }

    private static double GetDistance(Point2f p1, Point2f p2)
    {
        return Math.Sqrt(Math.Pow(p1.X - p2.X, 2) + Math.Pow(p1.Y - p2.Y, 2));
    }

    // 修改点8：辐射状线段计数
    private static int CountRadialLines(Point[] contour, Point2f center)
    {
        const double ANGLE_TOLERANCE = Math.PI / 8; // 收紧角度容差
        var angleBins = new Dictionary<double, bool>
    {
        { 0, false },          // 右
        { Math.PI/2, false },  // 下
        { Math.PI, false },    // 左
        { 3*Math.PI/2, false } // 上
    };

        // 增加采样密度
        foreach (var p in contour.Where((_, i) => i % 5 == 0))
        {
            double dx = p.X - center.X;
            double dy = p.Y - center.Y;
            if (Math.Abs(dx) < 2 && Math.Abs(dy) < 2) continue; // 忽略中心点

            double angle = Math.Atan2(dy, dx);
            angle = (angle + 2 * Math.PI) % (2 * Math.PI);

            foreach (var key in angleBins.Keys.ToList())
            {
                double diff = Math.Abs(angle - key);
                diff = Math.Min(diff, 2 * Math.PI - diff);

                if (diff < ANGLE_TOLERANCE)
                {
                    angleBins[key] = true;
                    break; // 避免重复计数
                }
            }
        }

        return angleBins.Count(pair => pair.Value);
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

                high = Math.Min(maxVal * 0.8, medianVal * 3.4); // 提高高阈值
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

    private static Point[] SmoothContour(Point[] contour)
    {
        List<Point> smoothed = new List<Point>();
        const double step = 0.07; // 高密度插值

        for (int i = 0; i < contour.Length; i++)
        {
            Point p0 = contour[(i - 1 + contour.Length) % contour.Length];
            Point p1 = contour[i];
            Point p2 = contour[(i + 1) % contour.Length];
            Point p3 = contour[(i + 2) % contour.Length];

            for (double t = 0; t <= 1; t += step)
            {
                double t2 = t * t;
                double t3 = t2 * t;

                int x = (int)(0.5 * (
                    (-t3 + 2 * t2 - t) * p0.X +
                    (3 * t3 - 5 * t2 + 2) * p1.X +
                    (-3 * t3 + 4 * t2 + t) * p2.X +
                    (t3 - t2) * p3.X));

                int y = (int)(0.5 * (
                    (-t3 + 2 * t2 - t) * p0.Y +
                    (3 * t3 - 5 * t2 + 2) * p1.Y +
                    (-3 * t3 + 4 * t2 + t) * p2.Y +
                    (t3 - t2) * p3.Y));

                smoothed.Add(new Point(x, y));
            }
        }

        return smoothed.ToArray();
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

    // 扩展方法（添加到Extensions类）
    public static Point2f GetCenter(this Rect rect)
    {
        return new Point2f(rect.X + rect.Width / 2f, rect.Y + rect.Height / 2f);
    }
}