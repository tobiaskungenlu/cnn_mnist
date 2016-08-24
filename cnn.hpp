//
//  cnn.hpp
//  mnsit
//
//  Created by 陆闻韬 on 16/4/17.
//  Copyright © 2016年 tobias_lu. All rights reserved.
//

#ifndef cnn_hpp
#define cnn_hpp

#include <stdio.h>

#endif
#include <iostream>
#include  <fstream>
#include <assert.h>
using namespace std;

#define width_image_cnn   32 //normalized image size
#define height_image_cnn  32

#define width_c1          28
#define height_c1         28
#define width_s2          14
#define height_s2         14
#define width_c3          10
#define height_c3         10
#define width_s4          5
#define height_s4         5
#define width_c5          1
#define height_c5         1
#define width_output_cnn  1
#define height_output_cnn 1
#define size_single_image

#define width_kernel      5//normalized conv kernel
#define height_kernel     5

#define width_sample_pooling      2//normalized pooling
#define height_sample_pooling     2

#define num_input_cnn     1 //num of maps of every layer
#define num_map_c1        6
#define num_map_s2        6
#define num_map_c3        16
#define num_map_s4        16
#define num_map_c5        120
#define num_output_cnn    10


#define patterns_train_cnn   60000 //训练模式对数(总数)
#define patterns_test_cnn    10000 //测试模式对数(总数)
#define iterations_cnn       30 //最大训练次数
#define accuracy_rate_cnn    0.95 //要求达到的准确率

#define ita_cnn              0.01//ita

#define len_weight_c1        150//5 * 5 * 6 = 150
#define len_bias_c1          6 //6  maps
#define len_weight_s2        6 // 1 * 6 = 6
#define len_bias_s2          6 // 6 maps
//#define len_weight_c3        1500 //5*5*3*6 + 5*5*4+9 + 5*5*6 = 1500
#define len_weight_c3        2400 //5*5*6*16
#define len_bias_c3          16 // 16 maps
#define len_weight_s4        16 // 1 * 16
#define len_bias_s4          16 //16 maps
#define len_weight_c5        48000 // 5*5*16*120
#define len_bias_c5          120 // 120 maps
#define len_weight_out_cnn   1200 // 120 *10
#define len_bias_out_cnn     10// 10 maps

#define len_input_map_all    1024 // 32 * 32
#define len_c1_map_all       4704 //28 * 28 *6
#define len_s2_map_all       1176 //14*14*6
#define len_c3_map_all       1600 // 10 * 10 * 16
#define len_s4_map_all       400  // 5 * 5 *16;
#define len_c5_map_all       120  // 1 * 1 * 120
#define len_output_map_all   10   //1 * 1 *10





class cnn
{
public:
    cnn();
    ~cnn();
    
    void initial();
    bool train();
    //int predict;
    //bool readModel(const char * file);
    
    
protected:
    //initial
    void getData();
    bool initialWeightsAndBias();
    void uniform_rand(float* data, int length, float  min, float max);//uniform distribution
    float uniform_rand(float min, float max);
    
    //functions and tools
    float activation_tanh(float x);
    float activation_tanh_derivative(float x);
    float mse(float x , float y);
    float mse_derivative(float x, float y);
    void Kronecker_2(const float a[],float *ret, int a_len);
    //float dot_product(float *a, float *b, int len);
    void expand(float *src, float *dst, int len_src, int len_dst);
    
    
    //forward process
    bool forward_c1();
    bool forward_s2();
    bool forward_c3();
    bool forward_s4();
    bool forward_c5();
    bool forward_output();
    
    //backword process
    bool backward_output();
    bool backward_c5();
    bool backward_s4();
    bool backward_c3();
    bool backward_s2();
    bool backward_c1();
    //update
    bool update();
    
    
private:
    float* data_input_train;
    float* data_label_train;
    float* data_input_test;
    float* data_label_test;
    float* data_single_image;
    float* data_single_label;
    
    float weight_c1[len_weight_c1];
    float bias_c1[len_bias_c1];
    float weight_s2[len_weight_s2];
    float bias_s2[len_bias_s2];
    float weight_c3[len_weight_c3];
    float bias_c3[len_bias_c3];
    float weight_s4[len_weight_s4];
    float bias_s4[len_bias_s4];
    float weight_c5[len_weight_c5];
    float bias_c5[len_bias_c5];
    float weight_out[len_weight_out_cnn];
    float bias_out[len_bias_out_cnn];
    
    
    float map_c1_out[width_c1 * height_c1 * num_map_c1];
    float map_s2_out[width_s2 * height_s2 * num_map_s2];
    float map_c3_out[width_c3 * height_c3 * num_map_c3];
    float map_s4_out[width_s4 * height_s4 * num_map_s4];
    float map_c5_out[width_c5 * height_c5 * num_map_c5];
    float map_output_out[width_output_cnn * height_output_cnn * num_output_cnn];

    
    
    
    
    float delta_output_cnn[len_output_map_all];
    float delta_c5[len_c5_map_all];
    float delta_s4[len_s4_map_all];
    float delta_c3[len_c3_map_all];
    float delta_s2[len_s2_map_all];
    float delta_c1[len_c1_map_all];
    
    
    
    
    
    /*
    constexpr static int Matrix_s2c3[num_map_s2][num_map_c3] =
    {
        1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1,
        1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1,
        1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1,
        0, 1, 1, 1, 0, 0 ,1, 1, 1, 1, 0, 0, 1, 0, 1, 1,
        0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
        0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,
    };
   
   */
    
    
    
    
    
};





