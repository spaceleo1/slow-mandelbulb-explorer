#ifndef GLOBAL_H
#define GLOBAL_H

#include "vec3d.h"

const int W = 1000;
const int H = 1000;
const float shift = 0.05;
const float angleShift = 0.05;

const int blocks = 10;
const int threads = 1024;

const int MAX_RAYMARCH_ITERS = 64;
const int MAX_MANDELBULB_ITERS = 32;

const float MANDELBULB_ESCAPE_RADIUS = 2;
const int MANDELBULB_POWER = 8;

#endif