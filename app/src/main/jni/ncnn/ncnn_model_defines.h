//
// Created by David Chiu on 12/31/18.
//

#ifndef NCNNCAM_NCNN_MODEL_DEFINES_H
#define NCNNCAM_NCNN_MODEL_DEFINES_H

#define STRINGIZE_AUX(a) #a
#define STRINGIZE(a) STRINGIZE_AUX(a)
#define INCLUDE_MACRO(file) STRINGIZE(file)

#ifdef HEADER_FILE
#include INCLUDE_MACRO(HEADER_FILE)
#warning "ncnn Header defined"
#endif

//#include "mobilenet_yolo.id.h"

static int ncnn_input = NCNN_INPUT;
static int ncnn_output = NCNN_OUTPUT;

#endif //NCNNCAM_NCNN_MODEL_DEFINES_H
