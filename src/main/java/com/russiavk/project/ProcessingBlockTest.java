package com.russiavk.project;

import org.intel.rs.frame.DepthFrame;
import org.intel.rs.frame.FrameList;
import org.intel.rs.frame.VideoFrame;
import org.intel.rs.option.CameraOption;
import org.intel.rs.pipeline.Config;
import org.intel.rs.pipeline.Pipeline;
import org.intel.rs.pipeline.PipelineProfile;
import org.intel.rs.processing.Align;
import org.intel.rs.processing.Colorizer;
import org.intel.rs.types.Format;
import org.intel.rs.types.Option;
import org.intel.rs.types.Stream;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class ProcessingBlockTest {
	static {
		System.load("C:\\opencv_java451.dll");
	}
	private volatile boolean running = true;

	public static void main(String[] args) {

		// new ProcessingBlockTest().runTest();
	}

	private static final NDArray change(NDArray position, NDArray arr1, NDArray arr2, NDArray arr3, int index,
			int length) {

		arr1.set(new NDIndex(index + ",0"), position.get(new NDIndex(index + ",0")).sub(212));
		arr1.set(new NDIndex(index + ",1"), 120 - position.get(new NDIndex(index + ",1")).getInt());
		arr1.set(new NDIndex(index + ",2"), position.get(new NDIndex(index + ",2")).sub(length));
		arr2.set(new NDIndex(index + ",0"), NDArrays.muli(position.get(new NDIndex(index + ",2")),
				arr1.get(new NDIndex(index + ",0")).div(length)));
		arr2.set(new NDIndex(index + ",1"), NDArrays.muli(position.get(new NDIndex(index + ",2")),
				arr1.get(new NDIndex(index + ",1")).div(length)));
		arr2.set(new NDIndex(index + ",2"), position.get(new NDIndex(index + ",2")));
		arr3.set(new NDIndex(index + ",0"), arr2.get(new NDIndex(index + ",0")));
		arr3.set(new NDIndex(index + ",1"), arr2.get(new NDIndex(index + ",2")));
		arr3.set(new NDIndex(index + ",2"), arr2.get(new NDIndex(index + ",1")));
		return arr3;
	}

	private static final NDArray transform(NDManager manager, NDArray nd, int opreation, int number, int value,
			int... DataType) {
		int index = 0;
		switch (DataType.length) {
		case 1:
			long[] longArr = new long[(int) nd.size()];
			for (long i : nd.toLongArray()) {
				switch (opreation) {
				case 1:
					if (i > number)
						i = value;
					break;
				case 2:
					if (i < number)
						i = value;
					break;
				case 3:
					if (i == number)
						i = value;
					break;
				}
				longArr[index] = i;
				index++;
			}
			return manager.create(longArr);
		default:
			int[] intArr = new int[(int) nd.size()];
			for (int i : nd.toIntArray()) {
				switch (opreation) {
				case 1:
					if (i > number)
						i = value;
					break;
				case 2:
					if (i < number)
						i = value;
					break;
				case 3:
					if (i == number)
						i = value;
					break;
				}
				intArr[index] = i;
				index++;
			}
			return manager.create(intArr);
		}

	}

	@SuppressWarnings("resource")
	public final void runTest() {
		Align align = new Align(Stream.Depth);
		Pipeline pipeline = new Pipeline();
		Colorizer colorizer = new Colorizer();
		System.out.println("setting up camera...");

		Runtime.getRuntime().addShutdownHook(new Thread(() -> {
			// shutdown camera
			running = false;

			pipeline.stop();
			System.out.println("camera has been shutdown!");
		}));

		// create camera
		Config cfg = new Config();
		cfg.enableStream(Stream.Depth, 424, 240, Format.Z16, 30);
		cfg.enableStream(Stream.Color, 424, 240, Format.Bgr8, 30);
		@SuppressWarnings("unused")
		PipelineProfile pp = pipeline.start(cfg);

		// set color scheme settings
		CameraOption colorScheme = colorizer.getOptions().get(Option.ColorScheme);
		colorScheme.setValue(2);
		System.out.println("camera has been started!");

		// setting up thread to read data
		Thread thread = new Thread(() -> {
			while (running) {
				// 等待一对连续的帧（包含深度和颜色）
				FrameList frames = pipeline.waitForFrames();
				FrameList alignedFrames = align.process(frames);
				// 对齐彩色图和深度图
				VideoFrame colorFrame = alignedFrames.getColorFrame();
				DepthFrame depthFrame = alignedFrames.getDepthFrame();
				if (depthFrame == null || colorFrame == null) {
					continue;
				}
				try (NDManager manager = NDManager.newBaseManager()) {
					// 将图片转换为numpy数组
					NDArray depth_image = manager.create((long) depthFrame.getDataSize());
					NDArray color_image = manager.create((long) colorFrame.getDataSize());
					// 过滤掉太远和太近的数据
					depth_image = transform(manager, depth_image, 1, 250, 0, 1);// depth_image.set(new
																				// NDIndex("depth_image>250"), 0);
					depth_image = transform(manager, depth_image, 2, 150, 0, 1);// depth_image.set(new
																				// NDIndex("depth_image<150"), 0);
					// 更改数据类型
					depth_image = manager.create(new Shape(depth_image.toLongArray()), DataType.UINT8);

					color_image = manager.create(new Shape(color_image.toLongArray()), DataType.UINT8);

					Mat mat_color_image = new Mat();
					mat_color_image.put(0, 0, color_image.toUint8Array()); // mat_color_image.fromArray(color_image.toIntArray());
					MatOfInt mat_color_gray = new MatOfInt();
					Imgproc.cvtColor(mat_color_image, mat_color_gray, Imgproc.COLOR_BGR2GRAY);
					NDArray color_gray = manager.create(mat_color_gray.toArray());
					// 创建4个4*3的数组，所有元素都为0
					NDArray pos = manager.zeros(new Shape(4, 3));
					NDArray pos_1 = manager.zeros(new Shape(4, 3));
					NDArray pos_2 = manager.zeros(new Shape(4, 3));
					NDArray pos_3 = manager.zeros(new Shape(4, 3));
					// 备份深度图，对备份图进行二值化
					NDArray depth_copy = null;
					depth_image.copyTo(depth_copy);

					depth_copy = transform(manager, depth_copy, 1, 0, 1);// depth_copy.set(new
																			// NDIndex("depth_copy>0"), 1);
					// 将灰度图和二值化深度图相乘，过滤掉灰度图中距离过远和过近的点（将值设为0）
					NDArray color_gray1 = NDArrays.muli(color_gray, depth_copy);
					color_gray1 = transform(manager, color_gray1, 3, 0, 255);// color_gray1.set(new
																				// NDIndex("color_gray1==0"), 255);
					MatOfInt mat_color_gray1 = new MatOfInt();
					mat_color_gray1.fromArray(color_gray1.toIntArray());
					Mat color_gray_gs = new Mat();
					// 对图片进行降噪，使用的高斯模糊，卷积核为5*5
					Imgproc.GaussianBlur(mat_color_gray1, color_gray_gs, new Size(5, 5), 0);
					MinMaxLocResult result = Core.minMaxLoc(color_gray_gs);
					double p1gray = result.minLoc.x;
					double p0gray = result.minLoc.y;
					pos.set(new NDIndex("0,0"), result.minLoc.y);
					pos.set(new NDIndex("0,1"), result.minLoc.x);
					pos.set(new NDIndex("0,2"), depth_image.get(new NDIndex(p0gray + "," + p1gray)));
					Imgproc.circle(mat_color_gray1, result.minLoc, 7, new Scalar(255, 0, 0), 2);
					Imgcodecs.imwrite("e:\\picture\\color_gray_circle.png", mat_color_gray1);
					System.out.println(change(pos, pos_1, pos_2, pos_3, 0, 201));
					int key = HighGui.waitKey(1);
					if (key != 0 & 0xFF == (char) 'q' || key == 27) {
						HighGui.destroyAllWindows();
						break;
					}
				}
			}
		});
		thread.start();
	}

}