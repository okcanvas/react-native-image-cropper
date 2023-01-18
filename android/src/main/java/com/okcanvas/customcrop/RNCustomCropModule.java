
package com.okcanvas.customcrop;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.provider.MediaStore;
import android.util.Base64;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.WritableMap;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgcodecs.*;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import org.opencv.calib3d.Calib3d;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;


public class RNCustomCropModule extends ReactContextBaseJavaModule {

  private final ReactApplicationContext reactContext;

  public RNCustomCropModule(ReactApplicationContext reactContext) {
    super(reactContext);
    OpenCVLoader.initDebug();
    this.reactContext = reactContext;
  }

  @Override
  public String getName() {
    return "CustomCropManager";
  }


  @ReactMethod
  public void imageSize(String imageUri, Callback callback) {
    Mat src;

    Boolean isBase64 = org.apache.commons.codec.binary.Base64.isBase64(imageUri.getBytes());
    if (isBase64) {
      byte[] decodedString = Base64.decode(imageUri.replace("data:image/jpg;base64,", ""), Base64.DEFAULT);
      Bitmap decodedByte = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length); 
      Bitmap bmp32 = decodedByte.copy(Bitmap.Config.ARGB_8888, true);
      src = new Mat();
      Utils.bitmapToMat(bmp32, src);
    } else {
      src = Imgcodecs.imread(imageUri.replace("file://", ""), Imgproc.COLOR_BGR2RGB);
    }

    WritableMap map = Arguments.createMap();
    map.putBoolean("isBase64", isBase64);
    map.putDouble("width", src.size().width);
    map.putDouble("height", src.size().height);
    callback.invoke(null, map);
  }

  @ReactMethod
  public void cropBase64(ReadableMap points, String imageBase64, Callback callback) {
    Point tl = new Point(points.getMap("topLeft").getDouble("x"), points.getMap("topLeft").getDouble("y"));
    Point tr = new Point(points.getMap("topRight").getDouble("x"), points.getMap("topRight").getDouble("y"));
    Point bl = new Point(points.getMap("bottomLeft").getDouble("x"), points.getMap("bottomLeft").getDouble("y"));
    Point br = new Point(points.getMap("bottomRight").getDouble("x"), points.getMap("bottomRight").getDouble("y"));

    byte[] decodedString = Base64.decode(imageBase64.replace("data:image/jpg;base64,", ""), Base64.DEFAULT);
    Bitmap decodedByte = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length); 

    Mat src = new Mat();    
    Bitmap bmp32 = decodedByte.copy(Bitmap.Config.ARGB_8888, true);
    Utils.bitmapToMat(bmp32, src);


    Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2RGB, 0);

    double widthA = Math.sqrt(Math.pow(br.x - bl.x, 2) + Math.pow(br.y - bl.y, 2));
    double widthB = Math.sqrt(Math.pow(tr.x - tl.x, 2) + Math.pow(tr.y - tl.y, 2));

    double dw = Math.max(widthA, widthB);
    int maxWidth = Double.valueOf(dw).intValue();

    double heightA = Math.sqrt(Math.pow(tr.x - br.x, 2) + Math.pow(tr.y - br.y, 2));
    double heightB = Math.sqrt(Math.pow(tl.x - bl.x, 2) + Math.pow(tl.y - bl.y, 2));

    double dh = Math.max(heightA, heightB);
    int maxHeight = Double.valueOf(dh).intValue();

    Mat doc = new Mat(maxHeight, maxWidth, CvType.CV_8UC4);

    Mat src_mat = new Mat(4, 1, CvType.CV_32FC2);
    Mat dst_mat = new Mat(4, 1, CvType.CV_32FC2);

    src_mat.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y);
    dst_mat.put(0, 0, 0.0, 0.0, dw, 0.0, dw, dh, 0.0, dh);

    Mat m = Imgproc.getPerspectiveTransform(src_mat, dst_mat);

    Imgproc.warpPerspective(src, doc, m, doc.size());
    Imgproc.cvtColor(doc, doc, Imgproc.COLOR_BGR2RGB, 0);


    Bitmap bitmap = Bitmap.createBitmap(doc.cols(), doc.rows(), Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(doc, bitmap);

    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
    byte[] byteArray = byteArrayOutputStream.toByteArray();

    WritableMap map = Arguments.createMap();
    map.putString("image", "data:image/jpg;base64," + Base64.encodeToString(byteArray, Base64.DEFAULT));
    map.putDouble("width", Double.valueOf(doc.size().width).intValue());
    map.putDouble("height", Double.valueOf(doc.size().height).intValue());
    callback.invoke(null, map);

    m.release();
  }

  @ReactMethod
  public void cropFile(ReadableMap points, String imageUri, Callback callback) {
    Point tl = new Point(points.getMap("topLeft").getDouble("x"), points.getMap("topLeft").getDouble("y"));
    Point tr = new Point(points.getMap("topRight").getDouble("x"), points.getMap("topRight").getDouble("y"));
    Point bl = new Point(points.getMap("bottomLeft").getDouble("x"), points.getMap("bottomLeft").getDouble("y"));
    Point br = new Point(points.getMap("bottomRight").getDouble("x"), points.getMap("bottomRight").getDouble("y"));

    Mat src = Imgcodecs.imread(imageUri.replace("file://", ""), Imgproc.COLOR_BGR2RGB);
    Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2RGB);

    double widthA = Math.sqrt(Math.pow(br.x - bl.x, 2) + Math.pow(br.y - bl.y, 2));
    double widthB = Math.sqrt(Math.pow(tr.x - tl.x, 2) + Math.pow(tr.y - tl.y, 2));

    double dw = Math.max(widthA, widthB);
    int maxWidth = Double.valueOf(dw).intValue();

    double heightA = Math.sqrt(Math.pow(tr.x - br.x, 2) + Math.pow(tr.y - br.y, 2));
    double heightB = Math.sqrt(Math.pow(tl.x - bl.x, 2) + Math.pow(tl.y - bl.y, 2));

    double dh = Math.max(heightA, heightB);
    int maxHeight = Double.valueOf(dh).intValue();

    Mat doc = new Mat(maxHeight, maxWidth, CvType.CV_8UC4);

    Mat src_mat = new Mat(4, 1, CvType.CV_32FC2);
    Mat dst_mat = new Mat(4, 1, CvType.CV_32FC2);

    src_mat.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y);
    dst_mat.put(0, 0, 0.0, 0.0, dw, 0.0, dw, dh, 0.0, dh);

    Mat m = Imgproc.getPerspectiveTransform(src_mat, dst_mat);

    Imgproc.warpPerspective(src, doc, m, doc.size());

    Bitmap bitmap = Bitmap.createBitmap(doc.cols(), doc.rows(), Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(doc, bitmap, true);
    

    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
    byte[] byteArray = byteArrayOutputStream.toByteArray();

    WritableMap map = Arguments.createMap();
    map.putString("image", "data:image/jpg;base64," + Base64.encodeToString(byteArray, Base64.DEFAULT));
    map.putDouble("width", Double.valueOf(doc.size().width).intValue());
    map.putDouble("height", Double.valueOf(doc.size().height).intValue());
    callback.invoke(null, map);

    m.release();
  }

}