//
//  cnn.cpp
//  mnsit
//
//  Created by 陆闻韬 on 16/4/17.
//  Copyright © 2016年 tobias_lu. All rights reserved.
//

#include "cnn.hpp"
#include <random>
#include <time.h>
#include <cmath>
using namespace std;

cnn::cnn()
{
    data_input_train = nullptr;
    data_label_train = nullptr;
    data_input_test = nullptr;
    data_label_test = nullptr;
    data_single_image = nullptr;
    data_single_label = nullptr;
    
}

cnn::~cnn()
{
    delete [] data_input_train;
    delete [] data_label_train;
    delete [] data_input_test;
    delete [] data_label_test;
    
}

static int reverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void loadMnistImage(const char* file, float * data, int num_image)
{
    const int width_src_image = 28;
    const int height_src_image = 28;
    const int x_padding = 2;
    const int y_padding = 2;
    const float scale_min = -0.1;
    const float scale_max = 1.175;
    
    ifstream f(file, ios::binary);
    assert(f.is_open());
    
    int magic_number =0;
    int number_of_images = 0;
    int n_rows = 0;//行
    int n_cols = 0; //列
    
    f.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    f.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    assert(number_of_images == num_image);
    f.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    f.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);
    assert(n_rows == height_src_image && n_cols == width_src_image);
    
    //cout<<magic_number<<" "<<number_of_images<<endl;
    int size_of_image = width_image_cnn * height_image_cnn;
    
    for (int i = 0; i < number_of_images; ++i)
    {
        int addr = size_of_image * i ;
       
        
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                unsigned char temp = 0;
                f.read((char*)&temp, sizeof(temp));
                data[addr + width_image_cnn * (r + y_padding) + c + x_padding] = (temp / 255.0) * (scale_max - scale_min) + scale_min;
               // cout<<addr + width_image_cnn * (r + y_padding) + c + x_padding<<endl;
             
            }
        }
    }
  
    
    
}

void loadMnistLabel(const char* filename, float * data, int num_label)
{
    const float scale_min = -0.9;
    const float scale_max = 0.9;
    
    std::ifstream file(filename, std::ios::binary);
    assert(file.is_open());
    
    int magic_number = 0;
    int number_of_images = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    assert(number_of_images == num_label);
    
    for (int i = 0; i < number_of_images; ++i) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        data[i * num_output_cnn + temp] = scale_max;
    }
    
}

void cnn::getData()
{
    const char* filename_image = "/Users/Tobias_Lu/Documents/data/mnist/train-images.idx3-ubyte";
    const char* filename_label = "/Users/Tobias_Lu/Documents/data/mnist/train-labels.idx1-ubyte";
    const char* filename_image_test = "/Users/Tobias_Lu/Documents/data/mnist/t10k-images.idx3-ubyte";
    const char* filename_label_test = "/Users/Tobias_Lu/Documents/data/mnist//t10k-labels.idx1-ubyte";
    
    loadMnistImage(filename_image, data_input_train, patterns_train_cnn);
    loadMnistLabel(filename_label, data_label_train, patterns_train_cnn);
    
    loadMnistImage(filename_image_test, data_input_test, patterns_test_cnn);
    loadMnistLabel(filename_label_test, data_label_test, patterns_test_cnn);
    
    
}

void initial_value(float* data, float ini, int length)
{
    for (int i = 0; i < length; i++)
    {
        data[i] = ini;
    }
}


float cnn::uniform_rand(float min, float max)
{
    srand((unsigned)time(NULL));
    int seed = rand() % 30;
    static mt19937 gen(seed);
    uniform_real_distribution<float> dst(min, max);
    return dst(gen);
}



void cnn::uniform_rand(float *data, int length, float min, float max)
{
    for (int i =0 ; i < length; i++)
    {
        data[i] = uniform_rand(min, max);
        
    }
}



bool cnn::initialWeightsAndBias()
{
    
    
    int fan_in;
    
    //c1
    fan_in = num_input_cnn * width_kernel * height_kernel;
    uniform_rand(weight_c1, len_weight_c1, -1.0/fan_in, 1.0/fan_in); //uniform distribution for weights
    initial_value(bias_c1, 0.0, len_bias_c1); // 0.0 for bias
    
   // for (int i = 0 ; i < len_weight_c1;i++)
   // {
   //     cout<<weight_c1[i]<<" ";
   // }
    
    //s2
    fan_in = num_map_c1 * width_kernel * height_kernel ;
    uniform_rand(weight_s2, len_weight_s2, -1.0/fan_in, 1.0/fan_in);
    initial_value(bias_s2, 0.0, len_bias_s2);
    
    //c3
    fan_in = num_map_s2 * width_kernel * height_kernel;
    uniform_rand(weight_c3, len_weight_c3, -1.0/fan_in, 1.0/fan_in);
    initial_value(bias_c3, 0.0, len_bias_c3);
    
    //s4
    fan_in = num_map_c3 * width_kernel * height_kernel;
    uniform_rand(weight_s4, len_weight_s4, -1.0/fan_in, 1.0/fan_in);
    initial_value(bias_s4, 0.0, len_bias_s4);
    
    //c5
    fan_in = num_map_s4 * width_kernel * height_kernel;
    uniform_rand(weight_c5, len_weight_c5, -1.0/fan_in, 1.0/fan_in);
    initial_value(bias_c5, 0.0, len_bias_c5);
    
    //out
    fan_in = num_map_c5 * width_kernel * height_kernel;
    uniform_rand(weight_out, len_weight_out_cnn, -1.0/fan_in, 1.0/fan_in);
    initial_value(bias_out, 0.0, len_bias_out_cnn);
    
    return true;
    
    
    
    
    
}

void cnn::initial()
{
    int length;
    
    length = width_image_cnn * height_image_cnn * patterns_train_cnn;
    data_input_train = new float[length];
    initial_value(data_input_train, -0.1, length);
    
    length = num_output_cnn * patterns_train_cnn;
    data_label_train = new float[length];
    initial_value(data_label_train, -0.9, length);
    
    length = width_image_cnn * height_image_cnn * patterns_test_cnn;
    data_input_test = new float[length];
    initial_value(data_input_test, -.01, length);
    
    length = num_output_cnn * patterns_test_cnn;
    data_label_test = new float[length];
    initial_value(data_label_test, -0.9, length);
    
    
    initialWeightsAndBias();
    getData();
    
    cout<<"Initialized! "<<endl;
    
    
    
    
}


float cnn::activation_tanh(float x)//use 1.7159tanh(x)
{
    float a = 1.7159;
    //float s = 2/3;
   // x = s * x;
    float ep  = exp(x);
    float em = exp(-x);
    float ret = (ep - em)/ (ep+ em);
    return a * ret;
}

float cnn::activation_tanh_derivative(float x)// use 1.719* tanh(x)
{
    float a = 1.7159;
    //float s = 2/3;
    //x = s * x;
    float ep  = exp(x);
    float em = exp(-x);
    float t = (ep - em)/(ep+ em);
    float ret = a * (1 - t * t) ;
    return ret;
    
}
/*
float cnn::activation_tanh(float x)
{
    float ep = std::exp(x);
    float em = std::exp(-x);
    
    return (ep - em) / (ep + em);
}

float cnn::activation_tanh_derivative(float x)
{
    return (1.0 - x * x);
}
*/


float cnn::mse(float  x , float y)
{
    return ((y-x)*(y-x)) / 2;
}

float cnn::mse_derivative(float x, float y)
{
    return y-x;
}

float* cnn::Kronecker_2(const float a[],int a_len)
{
    float* ret;
    int x = a_len * 2;
    ret = new float[x * x];
    for (int i = 0; i< x;i++)
    {
        for (int j = 0 ; j < x; j++)
        {
            int a_x = i  / 2;
            int a_y = j / 2;
            ret[i * x + j] = a[a_x * a_len + a_y] * 1.0;
        }
    }
    
    
    return ret;
}

bool cnn::train()
{
    for (int i = 0; i <patterns_train_cnn; i++)
    {
        data_single_image = data_input_train + i * len_input_map_all;
        data_single_label = data_label_train + i * len_input_map_all;
        
        //forward
        forward_c1();
        forward_s2();
        forward_c3();
        forward_s4();
        forward_c5();
        forward_output();
        
        //backward
        
        
        cin.get();
    }
    return true;
}




//forward
bool cnn::forward_c1()
{
    float image_input[width_image_cnn][height_image_cnn];
    
    for (int i = 0; i < height_image_cnn; i ++)
    {
        for (int j = 0 ; j< width_image_cnn; j++)
        {
            image_input[i][j] = data_single_image[i * height_image_cnn + j];
            //cout<<image_input[i][j]<<" ";
        }
    }
    
    int i = 0;
    while ( i < num_map_c1)
    {
        float* temp_kernel = weight_c1 + i * width_kernel * height_kernel;
        float* temp_c1_out = map_c1_out + i * height_c1 * width_c1;
        
        for (int r = 0; r < height_image_cnn - height_kernel + 1; r++)
        {
            for (int c = 0 ; c < width_image_cnn - width_kernel +1; c++)
            {
                int pos = r * (height_image_cnn - height_kernel + 1) + c;
                int pos_x = pos / 32;
                int pos_y = pos % 32;
               // cout<<pos<<" ";
                float temp = 0.0;
                for (int p = 0; p < height_kernel;p++)
                {
                    for (int q =0; q<width_kernel;q++)
                    {
                        
                        temp += image_input[pos_x + p][pos_y + q] * temp_kernel[p * height_kernel + q];
                        
                    }
                }
                //if (temp == 0) cout<<temp<<" ";
                temp_c1_out[pos] = activation_tanh(temp + bias_c1[i]);
                //if (!isnumber(temp_c1_out[pos])) cout<<temp_c1_out[pos]<<" ";
               // cout<<pos<<" ";
               // if (temp_c1_out[pos] != 0)
               // {
                //cout<<temp_c1_out[pos]<<" ";
           
               // }
            }
        }
        i++;
    }
   /*
    for (int i= 0 ; i < width_c1 * height_c1 * num_map_c1; i++)
    {
        cout<<map_c1_out[i]<<" ";
    }
    */
       return true;
}

bool cnn::forward_s2()
{
 
    int i = 0;
    float* c1_map_addr;
    float* temp_s2_out;
    while (i < num_map_s2)
    {
        c1_map_addr = map_c1_out + i * width_c1 * height_c1;
        temp_s2_out = map_s2_out + i * width_s2 * height_s2;
        float map_input[width_c1][height_c1];
        for (int j = 0 ; j < height_c1;j++)
        {
            for (int r = 0; r< width_c1;r++)
            {
                map_input[j][r] = c1_map_addr[j * height_c1 + r];
            }
        }
        
        
        float temp ;
        int pos = 0;
        for (int p = 0; p < height_c1; p+=2)
        {
            for (int q =0 ; q < width_c1; q+=2)
            {
                temp = map_input[q][p] + map_input[q+1][p] + map_input[q][p+1] + map_input[q+1][p+1];
               
              //  cout<<pos<<" ";
                temp_s2_out[pos] = activation_tanh(temp / 4 + bias_s2[i]);// mean pooling
                //cout<<i * width_s2 * height_s2 + pos<<" ";
                pos +=1;
            }
        }
        
        i++;
    }
    /*
    for (int i = 0 ; i < width_s2 * height_s2 * num_map_s2; i ++)
    {
        cout<<map_s2_out[i]<<" ";
    }
    */
    return true;
}



bool cnn::forward_c3()
{
    
    float map_s2_1[width_s2][height_s2];
    float map_s2_2[width_s2][height_s2];
    float map_s2_3[width_s2][height_s2];
    float map_s2_4[width_s2][height_s2];
    float map_s2_5[width_s2][height_s2];
    float map_s2_6[width_s2][height_s2];
    
    float *addr_in_s2 = map_s2_out + 0 * width_s2 * height_s2;
    for (int i = 0 ; i < height_s2;i++)
    {
        for (int j = 0 ; j < width_s2;j++)
        {
            map_s2_1[i][j] = addr_in_s2[i * height_s2 + j];
        }
    }
    
    addr_in_s2 = map_s2_out + 1 * width_s2 * height_s2;
    for (int i = 0 ; i < height_s2;i++)
    {
        for (int j = 0 ; j < width_s2;j++)
        {
            map_s2_2[i][j] = addr_in_s2[i * height_s2 + j];
        }
    }
    
    addr_in_s2 = map_s2_out + 2 * width_s2 * height_s2;
    for (int i = 0 ; i < height_s2;i++)
    {
        for (int j = 0 ; j < width_s2;j++)
        {
            map_s2_3[i][j] = addr_in_s2[i * height_s2 + j];
        }
    }
    
    addr_in_s2 = map_s2_out + 3 * width_s2 * height_s2;
    for (int i = 0 ; i < height_s2;i++)
    {
        for (int j = 0 ; j < width_s2;j++)
        {
            map_s2_4[i][j] = addr_in_s2[i * height_s2 + j];
        }
    }
 
    addr_in_s2 = map_s2_out + 4 * width_s2 * height_s2;
    for (int i = 0 ; i < height_s2;i++)
    {
        for (int j = 0 ; j < width_s2;j++)
        {
            map_s2_5[i][j] = addr_in_s2[i * height_s2 + j];
        }
    }
    
    addr_in_s2 = map_s2_out + 5 * width_s2 * height_s2;
    for (int i = 0 ; i < height_s2;i++)
    {
        for (int j = 0 ; j < width_s2;j++)
        {
            map_s2_6[i][j] = addr_in_s2[i * height_s2 + j];
            //cout<<map_s2_6[i][j]<<" ";
        }
     //   cout<<endl;
    }
    
    //cout<<map_s2_out[1175];
    
    int i = 0;
    while (i < num_map_c3)
    {
        float *addr_out_c3 = map_c3_out + i * width_c3 * height_c3;
       //float *addr_in_s2 = map_s2_out + i * width_s2 * height_s4 * num_map_s2;
        
        float kernel_1[width_kernel][height_kernel];//6 kernel in all 16 set
        float kernel_2[width_kernel][height_kernel];
        float kernel_3[width_kernel][height_kernel];
        float kernel_4[width_kernel][height_kernel];
        float kernel_5[width_kernel][height_kernel];
        float kernel_6[width_kernel][height_kernel];
        
        float *addr_weight_c3 = weight_c3 + i * width_kernel * height_kernel * num_map_s2;
        for (int j = 0 ; j < height_kernel;j++)
        {
            for (int k = 0 ; k < width_kernel;k++)
            {
                kernel_1[j][k] = addr_weight_c3[j * height_kernel + k];
            }
        }
        
        addr_weight_c3 =
        weight_c3 + i * width_kernel * height_kernel * num_map_s2 + 1 * width_kernel * height_kernel;
        for (int j = 0 ; j < height_kernel;j++)
        {
            for (int k = 0 ; k < width_kernel;k++)
            {
                kernel_2[j][k] = addr_weight_c3[j * height_kernel + k];
            }
        }
        
        addr_weight_c3 =
        weight_c3 + i * width_kernel * height_kernel * num_map_s2 + 2 * width_kernel * height_kernel;
        for (int j = 0 ; j < height_kernel;j++)
        {
            for (int k = 0 ; k < width_kernel;k++)
            {
                kernel_3[j][k] = addr_weight_c3[j * height_kernel + k];
            }
        }
        
        addr_weight_c3 =
        weight_c3 + i * width_kernel * height_kernel * num_map_s2 + 3 * width_kernel * height_kernel;
        for (int j = 0 ; j < height_kernel;j++)
        {
            for (int k = 0 ; k < width_kernel;k++)
            {
                kernel_4[j][k] = addr_weight_c3[j * height_kernel + k];
            }
        }
        
        addr_weight_c3 =
        weight_c3 + i * width_kernel * height_kernel * num_map_s2 + 4 * width_kernel * height_kernel;
        for (int j = 0 ; j < height_kernel;j++)
        {
            for (int k = 0 ; k < width_kernel;k++)
            {
                kernel_5[j][k] = addr_weight_c3[j * height_kernel + k];
            }
        }
        
        addr_weight_c3 =
        weight_c3 + i * width_kernel * height_kernel * num_map_s2 + 5 * width_kernel * height_kernel;
        for (int j = 0 ; j < height_kernel;j++)
        {
            for (int k = 0 ; k < width_kernel;k++)
            {
                kernel_6[j][k] = addr_weight_c3[j * height_kernel + k];
            }
        }
        
        
        /*
        float temp_out_1[width_c3][height_c3];
        float temp_out_2[width_c3][height_c3];
        float temp_out_3[width_c3][height_c3];
        float temp_out_4[width_c3][height_c3];
        float temp_out_5[width_c3][height_c3];
        float temp_out_6[width_c3][height_c3];
        */
        for (int p = 0 ; p < height_c3; p ++)
        {
            for (int q = 0 ; q < width_c3; q++)
            {
                float temp1 = 0.0;
                float temp2 = 0.0;
                float temp3 = 0.0;
                float temp4 = 0.0;
                float temp5 = 0.0;
                float temp6 = 0.0;
                for (int x = 0 ; x < height_kernel; x++)
                {
                    for (int y = 0; y < width_kernel;y++)
                    {
                        temp1 += map_s2_1[p + x][q + y] * kernel_1[x][y];
                        temp2 += map_s2_2[p + x][q + y] * kernel_2[x][y];
                        temp3 += map_s2_3[p + x][q + y] * kernel_3[x][y];
                        temp4 += map_s2_4[p + x][q + y] * kernel_4[x][y];
                        temp5 += map_s2_5[p + x][q + y] * kernel_5[x][y];
                        temp6 += map_s2_6[p + x][q + y] * kernel_6[x][y];
                    }
                }
                float temp_sum = temp1+temp2+temp3+temp4+temp5+temp6+bias_c3[i];
                addr_out_c3[p * height_c3 + q] = activation_tanh(temp_sum);
                
            }
        }
        
        
        
        i++;
    }
    
    //for (int j =0;j < width_c3 * height_c3 * num_map_c3; j++)
    //{
      //  cout<<map_c3_out[j]<<" ";
   // }
    
    
    
    
    
    
    return true;
    
}

bool cnn::forward_s4()
{
    int i = 0 ;
    float* c3_map_addr;
    float* temp_s4_out;
    while (i < num_map_s4)
    {
        c3_map_addr = map_c3_out + i * width_c3 * height_c3 ;
        temp_s4_out = map_s4_out + i * width_s4 * height_s4;
        
        float map_cur[width_c3][height_c3];
        for (int p = 0 ; p < height_c3; p++)
        {
            for (int q = 0 ; q < width_c3 ; q ++)
            {
                map_cur[p][q] = c3_map_addr[p * height_c3 + q];
            }
        }
        
        float temp ;
        int pos = 0;
        for (int p = 0; p < height_c3; p+=2)
        {
            for (int q =0 ; q < width_c3; q+=2)
            {
                temp = map_cur[q][p] + map_cur[q+1][p] + map_cur[q][p+1] + map_cur[q+1][p+1];
                
                //  cout<<pos<<" ";
                temp_s4_out[pos] = activation_tanh(temp / 4 + bias_s4[i]);// mean pooling
                //cout<<i * width_s2 * height_s2 + pos<<" ";
                pos +=1;
            }
        }
        
        
        
        i++;
    }
    
    //for (int j = 0 ; j < width_s4 * height_s4 * num_map_s4; j++)
   // {
    //    cout<<map_s4_out[j]<<" ";
   // }
   // cout<<map_s4_out[width_s4 * height_s4 * num_map_s4];
    
    
    
    
    
    return true;
}

/*
float** one_two(float* input, int width, int height)
{
    //float ret[5][5];
    //ret=(float**) new float*[height];
    float ret[5][5];
    //ret = (float**)malloc(width * sizeof(float*));
    for (int i = 0 ; i < height; i++)
        for (int j =0;j<width;j++)
        {
          //  ret[i]=new float[width];
            //ret[i] = (float*)malloc(height * sizeof(float));
            ret[i][j] = input[i* height + j];
        }
    return ;
}
*/

bool cnn::forward_c5()
{
    float* addr_input_s4 =map_s4_out;
    
    
    float map_s4_1[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_1[i][j] = addr_input_s4[i * height_kernel +j];
           // cout<<map_s4_1[i][j]<<" ";
        }
    }
    
    
   
    //cout<<"input done"<<endl;
    /*
    addr_input_s4 += 1 * width_s4 * height_s4;
    float** map_s4_2 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 2 * width_s4 * height_s4;
    float** map_s4_3 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 3 * width_s4 * height_s4;
    float** map_s4_4 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 4 * width_s4 * height_s4;
    float** map_s4_5 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 5 * width_s4 * height_s4;
    float** map_s4_6 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 6 * width_s4 * height_s4;
    float** map_s4_7 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 7 * width_s4 * height_s4;
    float** map_s4_8 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 8 * width_s4 * height_s4;
    float** map_s4_9 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 9 * width_s4 * height_s4;
    float** map_s4_10 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 10 * width_s4 * height_s4;
    float** map_s4_11 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 11 * width_s4 * height_s4;
    float** map_s4_12 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 12 * width_s4 * height_s4;
    float** map_s4_13 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 13 * width_s4 * height_s4;
    float** map_s4_14 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 14 * width_s4 * height_s4;
    float** map_s4_15 = one_two(addr_input_s4, width_s4, height_s4);
    
    addr_input_s4 += 15 * width_s4 * height_s4;
    float** map_s4_16 = one_two(addr_input_s4, width_s4, height_s4);
    */
    addr_input_s4 +=  width_s4 * height_s4;
    
    
    float map_s4_2[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_2[i][j] = addr_input_s4[i * height_kernel +j];
            
        }
    }
    
  
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_3[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_3[i][j] = addr_input_s4[i * height_kernel +j];
            
        }
    }
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_4[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_4[i][j] = addr_input_s4[i * height_kernel +j];
            
        }
    }
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_5[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_5[i][j] = addr_input_s4[i * height_kernel +j];
            
        }
    }
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_6[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_6[i][j] = addr_input_s4[i * height_kernel +j];
            
        }
    }
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_7[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_7[i][j] = addr_input_s4[i * height_kernel +j];
            
        }
    }
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_8[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_8[i][j] = addr_input_s4[i * height_kernel +j];
        }
    }
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_9[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_9[i][j] = addr_input_s4[i * height_kernel +j];
        }
    }
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_10[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_10[i][j] = addr_input_s4[i * height_kernel +j];
        }
    }
    
    addr_input_s4 += width_s4 * height_s4;
    float map_s4_11[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_11[i][j] = addr_input_s4[i * height_kernel +j];
        }
    }
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_12[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_12[i][j] = addr_input_s4[i * height_kernel +j];
        }
    }
    
    addr_input_s4 += width_s4 * height_s4;
    float map_s4_13[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_13[i][j] = addr_input_s4[i * height_kernel +j];
        }
    }
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_14[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_14[i][j] = addr_input_s4[i * height_kernel +j];
        }
    }
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_15[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_15[i][j] = addr_input_s4[i * height_kernel +j];
        }
    }
    
    addr_input_s4 +=  width_s4 * height_s4;
    float map_s4_16[width_kernel][height_kernel];
    for (int i = 0 ; i < height_kernel;i++)
    {
        for (int j = 0 ;  j< width_kernel;j++)
        {
            map_s4_16[i][j] = addr_input_s4[i * height_kernel +j];
        }
    }
    
    
    
    
   
    
    
    int i = 0;
    while (i < num_map_c5)
    {
        float* addr_weight_c5 = weight_c5 + i * width_kernel * height_kernel * num_map_s4 ;
        float kernel_1 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_1[p][q] = addr_weight_c5[p * height_kernel + q];
               // cout<<kernel_1[p][q]<<" ";
            }
        }
        
        
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 1 * width_kernel * height_kernel;
        
        
        
        
        float kernel_2 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_2[p][q] = addr_weight_c5[p * height_kernel + q];
                //cout<<kernel_2[p][q]<<" ";
            }
        }
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 2 * width_kernel * height_kernel;
        float kernel_3 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_3[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 3 * width_kernel * height_kernel;
        float kernel_4 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_4[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 4 * width_kernel * height_kernel;
        float kernel_5 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_5[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 5 * width_kernel * height_kernel;
        float kernel_6 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_6[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 6 * width_kernel * height_kernel;
        float kernel_7 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_7[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 7 * width_kernel * height_kernel;
        float kernel_8 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_8[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 8 * width_kernel * height_kernel;
        float kernel_9 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_9[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 9 * width_kernel * height_kernel;
        float kernel_10 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_10[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 10 * width_kernel * height_kernel;
        float kernel_11 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_11[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 11 * width_kernel * height_kernel;
        float kernel_12 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_12[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 12 * width_kernel * height_kernel;
        float kernel_13 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_13[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 13 * width_kernel * height_kernel;
        float kernel_14 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_14[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 14 * width_kernel * height_kernel;
        float kernel_15 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_15[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        
        addr_weight_c5 =
        weight_c5 + i * width_kernel * height_kernel
        * num_map_s4+ 15 * width_kernel * height_kernel;
        float kernel_16 [width_kernel][height_kernel];
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel; q++)
            {
                kernel_16[p][q] = addr_weight_c5[p * height_kernel + q];
            }
        }
        
        float temp = 0.0;
        
        
        for (int p = 0 ; p < height_kernel; p++)
        {
            for (int q = 0 ; q < width_kernel ; q++)
            {
               temp += map_s4_1[p][q] * kernel_1[p][q]
                +map_s4_2[p][q] * kernel_2[p][q]
                +map_s4_3[p][q] * kernel_3[p][q]
                +map_s4_4[p][q] * kernel_4[p][q]
                +map_s4_5[p][q] * kernel_5[p][q]
                +map_s4_6[p][q] * kernel_6[p][q]
                +map_s4_7[p][q] * kernel_7[p][q]
                +map_s4_8[p][q] * kernel_8[p][q]
                +map_s4_9[p][q] * kernel_9[p][q]
                +map_s4_10[p][q] * kernel_10[p][q]
                +map_s4_11[p][q] * kernel_11[p][q]
                +map_s4_12[p][q] * kernel_12[p][q]
                +map_s4_13[p][q] * kernel_13[p][q]
                +map_s4_14[p][q] * kernel_14[p][q]
                +map_s4_15[p][q] * kernel_15[p][q]
                +map_s4_16[p][q] * kernel_16[p][q];
                
                /*
                cout<<map_s4_1[p][q]<<" "<<kernel_1[p][q]<<endl;
                cout<<map_s4_2[p][q]<<" "<<kernel_2[p][q]<<endl;
                cout<<map_s4_3[p][q]<<" "<<kernel_3[p][q]<<endl;
                cout<<map_s4_4[p][q]<<" "<<kernel_4[p][q]<<endl;
                cout<<map_s4_5[p][q]<<" "<<kernel_5[p][q]<<endl;
                cout<<map_s4_6[p][q]<<" "<<kernel_6[p][q]<<endl;
                cout<<map_s4_7[p][q]<<" "<<kernel_7[p][q]<<endl;
                cout<<map_s4_8[p][q]<<" "<<kernel_8[p][q]<<endl;
                cout<<map_s4_9[p][q]<<" "<<kernel_9[p][q]<<endl;
                cout<<map_s4_10[p][q]<<" "<<kernel_10[p][q]<<endl;
                cout<<map_s4_11[p][q]<<" "<<kernel_11[p][q]<<endl;
                cout<<map_s4_12[p][q]<<" "<<kernel_12[p][q]<<endl;
                cout<<map_s4_13[p][q]<<" "<<kernel_13[p][q]<<endl;
                cout<<map_s4_14[p][q]<<" "<<kernel_14[p][q]<<endl;
                cout<<map_s4_15[p][q]<<" "<<kernel_15[p][q]<<endl;
                cout<<map_s4_16[p][q]<<" "<<kernel_16[p][q]<<endl;
                cout<<temp<<endl;
                cin.get();
                */
            }
        }
       // cout<<temp<<" ";
        map_c5_out[i] = activation_tanh(temp + bias_c5[i]);
       // cout<<map_c5_out[i]<<" ";
        
        
        
        i++;
    }
 


    return true;
}

bool cnn::forward_output()
{
    int i = 0;
    while (i < num_output_cnn)
    {
        float* addr_weight_output = weight_out + i * num_map_c5;
        float temp = 0.0;
        for (int j = 0; j < num_map_c5; j++)
        {
            temp += map_c5_out[j] * addr_weight_output[j];
        }
        temp += bias_out[i];
        map_output_out[i] = activation_tanh(temp);
        i++;
        
    }
    
    for (int j = 0 ; j < 10; j++)
    {
        cout<<map_output_out[j]<<" ";
    }
    
    
    
    
    
    return true;
}


bool cnn::backward_output()
{
    float de_dy[len_output_map_all];
    float dy_da[len_output_map_all];
    
    //initial_value(de_dy, 0.0, len_output_map_all);
    //initial_value(dy_da, 0.0, len_output_map_all);
    
    for (int i = 0 ; i < len_output_map_all; i++)
    {
        de_dy[i] = mse_derivative(data_single_label[i], map_output_out[i]);
        dy_da[i] = activation_tanh_derivative(map_output_out[i]);
        delta_output_cnn[i] = de_dy[i] * dy_da[i];
    }
    
    return true;
}



