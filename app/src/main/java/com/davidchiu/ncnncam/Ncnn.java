/*
 *  Created by David Chiu
 *  Dec. 28th, 2018
 *
 */
package com.davidchiu.ncnncam;

import android.content.Context;
import android.graphics.Bitmap;

import java.io.IOException;
import java.io.InputStream;

public class Ncnn
{
    public native boolean init(byte[] param, byte[] bin, byte[] words);

    public native float[] detect(Bitmap bitmap);

    static {
        System.loadLibrary("ncnn_jni");
    }


    public boolean initNcnn(Context context, String paramFile, String weightsFile, String labels) throws IOException
    {
        byte[] param = null;
        byte[] bin = null;
        byte[] words = null;

        if (paramFile == null || weightsFile == null) {
            return false;
        }
        {
            InputStream assetsInputStream;
                assetsInputStream = context.getAssets().open(paramFile);
            int available = assetsInputStream.available();
            param = new byte[available];
            int byteCode = assetsInputStream.read(param);
            assetsInputStream.close();
        }
        {
            InputStream assetsInputStream = context.getAssets().open(weightsFile);
            int available = assetsInputStream.available();
            bin = new byte[available];
            int byteCode = assetsInputStream.read(bin);
            assetsInputStream.close();
        }
        if (labels != null)
        {
            InputStream assetsInputStream = context.getAssets().open(labels);
            int available = assetsInputStream.available();
            words = new byte[available];
            int byteCode = assetsInputStream.read(words);
            assetsInputStream.close();
        }

        return init(param, bin, words);
    }
}
