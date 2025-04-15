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

            if (contours.Length > 0)
            {
                var mainContour = contours.First();

                // ==== 新增：计算并绘制中心点 ====
                // 计算轮廓中心（局部坐标系）
                Point localCenter = CalculateContourCenter(mainContour);

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

                // 亚像素级轮廓优化（保留凹点）
                var preciseContour = RefineContour(mainContour, gray);

                // 凹点检测增强
                var concavePoints = DetectConcavePoints(preciseContour);

                // 转换坐标系
                var globalContour = preciseContour
                    .Select(p => new Point(p.X + safeRect.X, p.Y + safeRect.Y))
                    .ToArray();


                // 绘制完整轮廓（红色）
                Cv2.Polylines(src, new[] { globalContour }, true, Scalar.Red, 2);

            }
        }
    }

    // 基于凸性缺陷的凹点检测
    // 优化凹点检测阈值（修改 DetectConcavePoints 方法）
    // 增强版凹点检测
    private static Point[] DetectConcavePoints(Point[] contour)
    {
        // 参数校验增强
        if (contour == null || contour.Length < 40) // 降低最小轮廓长度要求
        {
            Console.WriteLine($"轮廓过短：{contour?.Length ?? 0}点");
            return Array.Empty<Point>();
        }

        // 获取凸包索引时增加异常处理
        int[] hullIndices;
        try
        {
            hullIndices = Cv2.ConvexHullIndices(contour, clockwise: true);

            // 新增凸包索引验证
            if (!IsValidHullIndices(hullIndices, contour.Length))
            {
                Console.WriteLine("无效的凸包索引");
                return Array.Empty<Point>();
            }

            if (hullIndices.Length < 4) // 凸包至少需要4个点
            {
                Console.WriteLine($"无效凸包点数：{hullIndices.Length}");
                return Array.Empty<Point>();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"凸包计算异常：{ex.Message}");
            return Array.Empty<Point>();
        }

        // 计算凸性缺陷
        var defects = Cv2.ConvexityDefects(contour, hullIndices);
        if (defects == null || defects.Length == 0)
        {
            Console.WriteLine("未检测到凸包缺陷");
            return Array.Empty<Point>();
        }

        // 调试输出
        Console.WriteLine($"发现凸包缺陷数量：{defects.Length}");
        foreach (var d in defects.Take(3))
        {
            Console.WriteLine($"缺陷数据：[{d[0]},{d[1]},{d[2]},{d[3]}]");
        }

        // 动态参数计算
        double contourArea = Cv2.ContourArea(contour);
        double sizeFactor = Math.Sqrt(contourArea) / 20.0;

        // 统一匿名类型定义
        return defects
            .Select(d =>
            {
                // 定义统一返回类型
                var errorResult = new
                {
                    Valid = false,
                    Point = new Point(0, 0),
                    Depth = 0f,
                    Angle = 0.0
                };

                try
                {
                    // 解析缺陷参数
                    int farIndex = (int)d[2];

                    // 索引有效性验证
                    if (farIndex < 0 || farIndex >= contour.Length)
                    {
                        Console.WriteLine($"跳过无效索引：{farIndex}/{contour.Length}");
                        return errorResult;
                    }

                    // 计算动态阈值
                    float depth = (float)d[3] / 256;
                    double depthThreshold = 3.0 + sizeFactor * 2;
                    double angleThreshold = Math.PI * 0.45;

                    // 计算角度特征
                    double angle = CalculateAngleAtPoint(contour, farIndex);

                    // 验证条件
                    bool validDepth = depth > depthThreshold;
                    bool validAngle = angle > angleThreshold;
                    bool isEdge = IsEdgePoint(contour[farIndex], contour);

                    return new
                    {
                        Valid = validDepth && validAngle && !isEdge,
                        Point = contour[farIndex],
                        Depth = depth,
                        Angle = angle
                    };
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"处理缺陷时发生异常：{ex.Message}");
                    return errorResult;
                }
            })
            .Where(p => p.Valid) // 显式过滤有效点
            .OrderByDescending(p => p.Depth)
            .Take(4)
            .Select(p => p.Point)
            .ToArray();
    }

    private static bool IsValidHullIndices(int[] hullIndices, int contourLength)
    {
        if (hullIndices.Length < 3) return false;
        return hullIndices.All(idx => idx >= 0 && idx < contourLength) &&
               hullIndices.SequenceEqual(hullIndices.OrderBy(x => x));
    }

    // 新增：计算轮廓点的内角
    private static double CalculateAngleAtPoint(Point[] contour, int index)
    {

        // 新增防御性校验
        if (contour == null || contour.Length == 0)
        {
            Console.WriteLine("错误：空轮廓输入");
            return 0;
        }

        const int window = 3; // 检查前后3个点的平均方向
        Vector2 prevDirSum = new Vector2();
        Vector2 nextDirSum = new Vector2();

        // 安全索引计算（核心修复）
        int safeIndex(int i) => (i % contour.Length + contour.Length) % contour.Length;

        for (int i = -window; i <= window; i++)
        {
            int prevIdx = (index + i - 1 + contour.Length) % contour.Length;
            int currIdx = (index + i + contour.Length) % contour.Length;
            int nextIdx = (index + i + 1) % contour.Length;

            prevDirSum += new Vector2(
                contour[currIdx].X - contour[prevIdx].X,
                contour[currIdx].Y - contour[prevIdx].Y);

            nextDirSum += new Vector2(
                contour[nextIdx].X - contour[currIdx].X,
                contour[nextIdx].Y - contour[currIdx].Y);
        }

        double dot = prevDirSum.X * nextDirSum.X + prevDirSum.Y * nextDirSum.Y;
        double mag1 = Math.Sqrt(prevDirSum.X * prevDirSum.X + prevDirSum.Y * prevDirSum.Y);
        double mag2 = Math.Sqrt(nextDirSum.X * nextDirSum.X + nextDirSum.Y * nextDirSum.Y);

        return Math.Acos(dot / (mag1 * mag2));
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

    // 新增辅助方法：过滤位于轮廓边缘的伪凹点
    private static bool IsEdgePoint(Point p, Point[] contour)
    {
        // 检查点是否在轮廓的前5%或后5%范围内
        int edgeThreshold = (int)(contour.Length * 0.05);
        int index = Array.IndexOf(contour, p);

        return index < edgeThreshold ||
               index > contour.Length - edgeThreshold;
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

    // 修改 RefineContour 方法（保留凹点关键修改）
    private static Point[] RefineContour(Point[] contour, Mat gray)
    {
        // 步骤1：亚像素优化（降低迭代次数）
        var points = contour.Select(p => new Point2f(p.X, p.Y)).ToArray();
        Cv2.CornerSubPix(gray, points, new Size(3, 3), new Size(-1, -1),
            new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 5, 0.01));

        // 步骤2：强制保留原始凹点
        var concaveAnchors = DetectConcavePoints(contour)
            .Select(p => new Point2f(p.X, p.Y))
            .ToArray();

        // 合并优化点和凹点锚点
        var mergedPoints = points.Concat(concaveAnchors).ToArray();

        // 步骤3：凹点敏感型插值
        var smoothed = SmoothContour(mergedPoints.Select(p => new Point((int)p.X, (int)p.Y)).ToArray());

        // 步骤4：凹点补偿多边形近似
        double epsilon = 0.002 * Cv2.ArcLength(smoothed, true); // 提高精度
        var approx = Cv2.ApproxPolyDP(smoothed, epsilon, true);

        // 最终轮廓 = 近似结果 + 原始凹点（双重保障）
        return approx
            .Concat(concaveAnchors.Select(p => new Point((int)p.X, (int)p.Y)))
            .GroupBy(p => new { p.X, p.Y })
            .Select(g => g.First())
            .ToArray();
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
            Cv2.WaitKey(50);

            // ===== [线段连接阶段] =====
            // 修改点3：十字特征定向连接
            using (var connectedEdges = new Mat())
            {
                // 水平方向连接（4像素长度阈值）
                var hKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(6, 1));
                Cv2.MorphologyEx(processed, connectedEdges, MorphTypes.Close, hKernel, iterations: 2);

                // 垂直方向连接（5像素长度阈值）
                var vKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(1, 5));
                Cv2.MorphologyEx(connectedEdges, connectedEdges, MorphTypes.Close, vKernel, iterations: 2);

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
                int minPoints = (int)(15 * Math.Max(1, sizeFactor)); // 最小点数15~动态调整
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

                // 动态中心区域范围
                double centerRatio = 0.4 - 0.1 * (1 / sizeFactor); // ROI越小检测区域越大
                var centerROI = new Rect(
                    (int)(currentRotatedRect.Center.X - currentRotatedRect.Size.Width * centerRatio),
                    (int)(currentRotatedRect.Center.Y - currentRotatedRect.Size.Height * centerRatio),
                    (int)(currentRotatedRect.Size.Width * centerRatio * 2),
                    (int)(currentRotatedRect.Size.Height * centerRatio * 2)
                );

                // 降低检测密度要求
                int requiredPoints = (int)(contour.Length * 0.1); // 只需10%的点在中心区
                if (contour.Count(p => centerROI.Contains(p)) < requiredPoints)
                {
                    Console.WriteLine($"中心点不足：{requiredPoints}");
                    continue;
                }

                // 动态方向要求
                int minDirections = sizeFactor < 0.8 ? 2 : 3; // 小ROI只需2个方向
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
            // 修改点6：多维度评分系统
            // 修改为完整Lambda表达式：
            var bestContour = crossCandidates
               .Select(c =>
               {
                   var currentRotatedRect = Cv2.MinAreaRect(c);
                   return new
                   {
                       Contour = c,
                       SizeScore = 1 / (1 + Math.Exp(-sizeFactor)),
                       LineScore = CountRadialLines(c, currentRotatedRect.Center) * (2 - sizeFactor),
                       FillRatio = CalculateFillRatio(c) * 1.2
                   };
               })
               .OrderByDescending(x =>
                    x.SizeScore * 0.4 +
                    x.LineScore * 0.3 +
                    x.FillRatio * 0.3)
               .FirstOrDefault()?.Contour;
            // ===== [轮廓后处理] =====
            // 修改点7：强制生成闭合轮廓
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
}