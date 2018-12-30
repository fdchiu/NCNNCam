/*
 *  Created by David Chiu
 *  Dec. 28th, 2018 based on tencent's original file
 *
 */

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

// ncnn
#include "include/net.h"


#include <sys/time.h>
#include <unistd.h>
#include <mat.h>
#include <net.h>

//#include "yolov2-tiny_voc.id.h"
#include "mobilenet_yolo.id.h"
#include "include/allocator.h"
#include "include/mat.h"
#include "include/net.h"

static struct timeval tv_begin;
static struct timeval tv_end;
static double elasped;

struct Object
{
    //cv::Rect_<float> rect;
    float x, y, width, height;
    int label;
    float prob;
};

static void bench_start()
{
    gettimeofday(&tv_begin, NULL);
}

static void bench_end(const char* comment)
{
    gettimeofday(&tv_end, NULL);
    elasped = ((tv_end.tv_sec - tv_begin.tv_sec) * 1000000.0f + tv_end.tv_usec - tv_begin.tv_usec) / 1000.0f;
//     fprintf(stderr, "%.2fms   %s\n", elasped, comment);
    __android_log_print(ANDROID_LOG_DEBUG, "Ncnn", "%.2fms   %s", elasped, comment);
}

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Mat ncnn_param;
static ncnn::Mat ncnn_bin;
static std::vector<std::string> ncnn_label;
static ncnn::Net ncnnnet;

static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

#define NCNNJNI_METHOD(METHOD_NAME) \
  Java_com_davidchiu_ncnncam_Ncnn_##METHOD_NAME  // NOLINT

extern "C" {

// public native boolean Init(byte[] param, byte[] bin, byte[] words);
JNIEXPORT jboolean JNICALL  NCNNJNI_METHOD(init)(JNIEnv* env, jobject thiz, jbyteArray param, jbyteArray bin, jbyteArray words)
{
    // init param
    {
        int len = env->GetArrayLength(param);
        ncnn_param.create(len, (size_t)1u);
        env->GetByteArrayRegion(param, 0, len, (jbyte*)ncnn_param);
        int ret = ncnnnet.load_param((const unsigned char*)ncnn_param);
        __android_log_print(ANDROID_LOG_DEBUG, "yolov2ncnn", "load_param %d %d", ret, len);
    }

    // init bin
    {
        int len = env->GetArrayLength(bin);
        ncnn_bin.create(len, (size_t)1u);
        env->GetByteArrayRegion(bin, 0, len, (jbyte*)ncnn_bin);
        int ret = ncnnnet.load_model((const unsigned char*)ncnn_bin);
        __android_log_print(ANDROID_LOG_DEBUG, "yolov2ncnn", "load_model %d %d", ret, len);
    }

    if (words != NULL)
    // init words
    {
        int len = env->GetArrayLength(words);
        std::string words_buffer;
        words_buffer.resize(len);
        env->GetByteArrayRegion(words, 0, len, (jbyte*)words_buffer.data());
        ncnn_label = split_string(words_buffer, "\n");
    }

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    ncnn::set_default_option(opt);

    return JNI_TRUE;
}

JNIEXPORT jarray JNICALL NCNNJNI_METHOD(nativeDetect)(JNIEnv* env, jobject thiz, jobject bitmap)
{
    bench_start();

    // ncnn from bitmap
    ncnn::Mat in;
    {
        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, bitmap, &info);
        int width = info.width;
        int height = info.height;

        __android_log_print(ANDROID_LOG_DEBUG, "yolov2ncnn", "image size: %dx%d", width, height);
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
            return NULL;

        void* indata;
        AndroidBitmap_lockPixels(env, bitmap, &indata);
//        in = ncnn::Mat::from_pixels_resize((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2RGB, origin_w, origin_h, width, height);

        in = ncnn::Mat::from_pixels((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2RGB/*ncnn::Mat::PIXEL_BGR*/, width, height);
        AndroidBitmap_unlockPixels(env, bitmap);
    }

    // ncnn_net
    std::vector<float> cls_scores;
    {
        // 减去均值和乘上比例
    const float mean_vals[3] = {0.5f, 0.5f, 0.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    in.substract_mean_normalize(0, norm_vals);
    in.substract_mean_normalize(mean_vals, 0);

        //__android_log_print(ANDROID_LOG_DEBUG, "yolov2TinyJniIn", "yolov2_predict_has_input3, in[0][0]: %f; in[250][250]: %f", in.row(0)[0], in.row(250)[250]);

        ncnn::Extractor ex = ncnnnet.create_extractor();
        ex.input(mobilenet_yolo_param_id::BLOB_data, in);
        ncnn::Mat out;
        ex.extract(mobilenet_yolo_param_id::BLOB_detection_out, out);
        __android_log_print(ANDROID_LOG_DEBUG, "NcnnCam", "ncnn out: %dx%d;", out.w, out.h);
        int output_wsize = out.w;
        int output_hsize = out.h;

        jfloat output[output_wsize * output_hsize+1];
        output[0] = out.w;
        for(int i = 0; i< out.h; i++) {
            for (int j = 0; j < out.w; j++) {
                output[i*output_wsize + j +1] = out.row(i)[j];
            }
        }

#if 1
        jfloatArray jOutputData = env->NewFloatArray(output_wsize*output_hsize+1);
        if (jOutputData == NULL) return NULL;
        env->SetFloatArrayRegion(jOutputData, 0,  output_wsize * output_hsize+1,output);  // copy

        return jOutputData;
#else
    //const std::string& word = squeezenet_words[top_class];
    //char tmp[32];
    //sprintf(tmp, "%.3f", max_score);
    //std::string result_str = std::string(word.c_str() + 10) + " = " + tmp;
    std::string result_str = std::string("finished");
        int img_w = 640;
        int img_h = 480;

        if (0) {
            for (int i = 0; i < out.h; i++) {
                const float *values = out.row(i);

                Object object;
                object.label = values[0];
                object.prob = values[1];
                object.x = values[2] * img_w;
                object.y = values[3] * img_h;
                object.width = values[4] * img_w - object.x;
                object.height = values[5] * img_h - object.y;
                __android_log_print(ANDROID_LOG_DEBUG, "yolov2", "%.2f %.2f %.2f %.2f %.2f %.2f",
                                    values[0], values[1],
                                    object.x, object.y, object.width, object.height);
                //objects.push_back(object);
            }
        }

    // +10 to skip leading n03179701
    jstring result = env->NewStringUTF(result_str.c_str());

    bench_end("detect");

    return result;

#endif


    }
  }
}
