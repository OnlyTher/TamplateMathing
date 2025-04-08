using System;
using System.Linq;
using OpenCvSharp;

class InteractiveROIMatcher
{
    private static Mat _sourceImage;
    private static Mat _displayImage;
    private static Point _roiStart;
    private static Rect _selectedROI;
    private static bool _isSelecting;
    private static Point _currentMousePos = new Point(0, 0);

    static void Main(string[] args)
    {
        string sourcePath = "source.jpg";
        string templatePath = "template.jpg";
        string outputPath = "result.jpg";
        //Console.WriteLine("当前工作目录: " + System.IO.Directory.GetCurrentDirectory());

        //读取源图像
        _sourceImage = Cv2.ImRead(sourcePath, ImreadModes.Color);
        if (_sourceImage.Empty())
        {
            Console.WriteLine("无法加载源图像");
            return;
        }

        const string windowName = "Select ROI (Drag mouse to select, ENTER to confirm)";
        Cv2.NamedWindow(windowName, WindowFlags.AutoSize);

        // 鼠标回调注册 手动绘制模板
        Cv2.SetMouseCallback(windowName, OnMouse);

        _displayImage = _sourceImage.Clone();

        while (true)
        {
            Cv2.ImShow(windowName, _displayImage);
            int key = Cv2.WaitKey(10);

            if (key == 27) break; // ESC退出

            if (key == 13 && _selectedROI.Width > 0 && _selectedROI.Height > 0) // ENTER确认
            {
                ProcessWithSelectedROI(templatePath, outputPath);
                break;
            }

            if (_isSelecting)
            {
                _displayImage = _sourceImage.Clone();
                Cv2.Rectangle(_displayImage, _roiStart, _currentMousePos, Scalar.Red, 2);
            }
        }

        Cv2.DestroyAllWindows();
    }

    // 修正后的鼠标回调方法（必须使用这个精确签名）
    private static void OnMouse(MouseEventTypes @event, int x, int y, MouseEventFlags flags, IntPtr userdata)
    {
        _currentMousePos = new Point(x, y); // 更新鼠标位置
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
                Cv2.Rectangle(_displayImage, _selectedROI, Scalar.Green, 2);
                Cv2.PutText(_displayImage, $"ROI Selected (Press ENTER)",
                    new Point(10, 30),
                    HersheyFonts.HersheyComplex, 0.3, Scalar.Red, 1);
                break;
        }
    }

    private static void ProcessWithSelectedROI(string templatePath, string outputPath)
    {
        using (Mat sourceROI = new Mat(_sourceImage, _selectedROI))
        {
            // 提示用户保存模板
            Console.WriteLine("是否保存当前ROI为模板？(Y/N)");
            if (Console.ReadKey().Key == ConsoleKey.Y)
            {
                // 保存模板图像
                Cv2.ImWrite(templatePath, sourceROI);
                Console.WriteLine($"\n模板已保存至: {templatePath}");
            }
        }

            using (Mat template = Cv2.ImRead(templatePath, ImreadModes.Color))
        {
            if (template.Empty())
            {
                Console.WriteLine("无法加载模板图像");
                return;
            }

            using (Mat sourceROI = new Mat(_sourceImage, _selectedROI))
            {
                const string roiWindowName = "ROI Preview (Press any key to continue)";
                Cv2.NamedWindow(roiWindowName, WindowFlags.AutoSize);

                Mat visualROI = sourceROI.Clone();
                Cv2.PutText(visualROI, $"Size: {sourceROI.Width}x{sourceROI.Height}",
                    new Point(5, 60),
                    HersheyFonts.HersheySimplex, 0.3, Scalar.White, 1);
                Cv2.ImShow(roiWindowName, visualROI);

                Point matchLoc;
                double confidence;
                if (!MatchTemplate(sourceROI, template, out matchLoc, out confidence))
                {
                    Console.WriteLine("未找到匹配");
                    Cv2.WaitKey(0);
                    return;
                }

                DrawResult(_sourceImage, template, _selectedROI, matchLoc, confidence);
                Cv2.ImWrite(outputPath, _sourceImage);

                Console.WriteLine($"匹配完成！置信度: {confidence:P2}");
                Cv2.ImShow("Final Result", _sourceImage);
                //Cv2.WaitKey(0);
          
                Cv2.DestroyWindow(roiWindowName);
            }
        }
    }

    private static bool MatchTemplate(Mat sourceROI, Mat template, out Point matchLoc, out double confidence)
    {
        using (Mat sourceGray = new Mat())
        using (Mat templateGray = new Mat())
        using (Mat result = new Mat())
        {
            // 灰度转换提升匹配效率
            Cv2.CvtColor(sourceROI, sourceGray, ColorConversionCodes.BGR2GRAY);
            Cv2.CvtColor(template, templateGray, ColorConversionCodes.BGR2GRAY);

            // 使用归一化相关系数匹配法
            Cv2.MatchTemplate(
                sourceGray,
                templateGray,
                result,
                TemplateMatchModes.CCoeffNormed
            );

            // 获取最大匹配值和位置
            result.MinMaxLoc(out _, out confidence, out _, out matchLoc);


            return confidence > 0.75;   // 置信度阈值
        }
    }

    private static void DrawResult(Mat source, Mat template, Rect roi, Point matchLoc, double confidence)
    {
        // 计算实际匹配位置
        Point actualLoc = new Point(
            roi.X + matchLoc.X,
            roi.Y + matchLoc.Y
        );

        Rect resultRect = new Rect(
            actualLoc.X,
            actualLoc.Y,
            template.Width,
            template.Height
        );
        //Cv2.Rectangle(source, resultRect, Scalar.Red, 2);
        //Cv2.Rectangle(source, roi, Scalar.Green, 1);

        Cv2.PutText(source, $"Confidence: {confidence:P2}",
            new Point(actualLoc.X - 10, actualLoc.Y - 20),
            HersheyFonts.HersheyComplex, 0.4, Scalar.Red, 1);

        // 1. 绘制模板轮廓
        DrawTemplateContour(source, template, actualLoc);

        // 2. 计算轮廓中心点并绘制十字
        Point center = CalculateContourCenter(template);
        Point centerInSource = new Point(
            actualLoc.X + center.X,
            actualLoc.Y + center.Y
        );

        // 绘制正十字（长度10像素，颜色黄色）
        Cv2.DrawMarker(
            source,
            centerInSource,
            Scalar.Yellow,
            MarkerTypes.Cross,
            10,  // 十字大小
            1    // 线宽
        );
    }

    // 计算模板轮廓的中心点
    private static Point CalculateContourCenter(Mat template)
    {
        using (Mat templateGray = new Mat())
        {
            Cv2.CvtColor(template, templateGray, ColorConversionCodes.BGR2GRAY);
            Point[][] contours = GetContours(templateGray);

            if (contours.Length == 0)
                return new Point(template.Width / 2, template.Height / 2); // 无轮廓时返回图像中心

            // 取最大轮廓计算中心
            var largestContour = contours
                .OrderByDescending(c => Cv2.ContourArea(c))
                .First();

            Moments m = Cv2.Moments(largestContour);
            int cx = (int)(m.M10 / m.M00);
            int cy = (int)(m.M01 / m.M00);
            return new Point(cx, cy);
        }
    }

    //绘制模板轮廓（增强可视化效果）
    private static void DrawTemplateContour(Mat source, Mat template, Point location)
    {
        using (Mat templateGray = new Mat())
        {
            Cv2.CvtColor(template, templateGray, ColorConversionCodes.BGR2GRAY);
            Point[][] contours = GetContours(templateGray);

            foreach (var contour in contours)
            {
                Point[] transformed = contour.Select(p =>
                    new Point(p.X + location.X, p.Y + location.Y)
                ).ToArray();

                Cv2.Polylines(
                    source,
                    new[] { transformed },
                    true,
                    Scalar.Blue,
                    2
                );
            }
        }
    }

    private static Point[][] GetContours(Mat image)
    {
        using (Mat edges = new Mat())
        {
            // 通过Canny边缘检测提取轮廓
            Cv2.Canny(image, edges, 100, 200);
            Cv2.FindContours(
                edges,
                out Point[][] contours,
                out _,
                RetrievalModes.External,
                ContourApproximationModes.ApproxSimple
            );
            return contours.Where(c => Cv2.ContourArea(c) > 50).ToArray();
        }
    }
}